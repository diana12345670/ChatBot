import os
import logging
import json
from pathlib import Path
import secrets
from datetime import datetime, timedelta, timezone
import re
import hashlib
import hmac
from collections import deque
import requests
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import PlainTextResponse, HTMLResponse

from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


logger = logging.getLogger("whatsapp-bot")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


# Persisted storage (best-effort). On some Railway deployments, filesystem may reset.
# For Render, use /tmp directory for persistent storage
default_path = "/tmp/storage.json" if os.environ.get("RENDER") else "storage.json"
_STORAGE_PATH = Path(os.environ.get("CONFIG_PATH", default_path))


def _load_storage() -> dict:
    if not _STORAGE_PATH.exists():
        return {}
    try:
        return json.loads(_STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read storage.json")
        return {}


def _save_storage(storage: dict) -> None:
    _STORAGE_PATH.write_text(json.dumps(storage, ensure_ascii=False, indent=2), encoding="utf-8")


_storage: dict = _load_storage()
_storage.setdefault("login_codes", {})
_storage.setdefault("clients", {})
_storage.setdefault("accounts", {})


_MAX_ACCOUNTS = int(os.environ.get("MAX_ACCOUNTS", "20"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _dt_to_str(dt: datetime | None) -> str | None:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_or_none(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else None


def _require_admin_master_key(request: Request) -> None:
    required = os.environ.get("ADMIN_MASTER_KEY")
    if not required:
        raise HTTPException(status_code=500, detail="Missing ADMIN_MASTER_KEY")
    provided = request.headers.get("x-admin-key") or request.query_params.get("key")
    if provided != required:
        raise HTTPException(status_code=403, detail="Forbidden")


def _get_client(client_id: str) -> dict | None:
    return (_storage.get("clients") or {}).get(client_id)


def _get_account(account_id: str) -> dict | None:
    return (_storage.get("accounts") or {}).get(account_id)


def _find_account_by_email(email: str) -> tuple[str, dict] | None:
    for aid, acc in ((_storage.get("accounts") or {}).items()):
        if (acc.get("email") or "").lower() == email.lower():
            return aid, acc
    return None


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return salt, dk.hex()


def _verify_password(password: str, salt: str, pw_hash: str) -> bool:
    _, h = _hash_password(password, salt=salt)
    return hmac.compare_digest(h, pw_hash)


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def _is_client_active(client: dict) -> bool:
    exp = _parse_dt(client.get("expires_at"))
    if exp is None:
        return True
    return _utcnow() < exp


def _client_setting(client: dict, key: str, default=None):
    if key in client and client.get(key) not in (None, ""):
        return client.get(key)
    return os.environ.get(key, default)


def _ensure_storage_saved() -> None:
    _save_storage(_storage)


def _is_account_active(acc: dict) -> bool:
    exp = _parse_dt(acc.get("expires_at"))
    if exp is None:
        return True
    return _utcnow() < exp


# Memory is kept in-process. On Railway, redeploy/restart will reset it.
# Keyed by (client_id, whatsapp_from_phone)
_conversation_memory: dict[str, deque[dict]] = {}


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_history(client_id: str, from_phone: str, max_messages: int) -> deque[dict]:
    key = f"{client_id}:{from_phone}"
    history = _conversation_memory.get(key)
    if history is None:
        history = deque(maxlen=max_messages)
        _conversation_memory[key] = history
        return history
    if history.maxlen != max_messages:
        history = deque(list(history), maxlen=max_messages)
        _conversation_memory[key] = history
    return history


def _build_openai_messages(system_prompt: str, history: deque[dict], user_text: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, *list(history), {"role": "user", "content": user_text}]


def send_whatsapp_text(*, token: str, phone_number_id: str, graph_version: str, to_phone: str, text: str) -> None:

    url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"WhatsApp send failed: {r.status_code} {r.text}")


def extract_incoming_messages(body: dict) -> list[tuple[str, str]]:
    messages_out: list[tuple[str, str]] = []

    entry = body.get("entry") or []
    for e in entry:
        changes = e.get("changes") or []
        for c in changes:
            value = (c.get("value") or {})
            messages = value.get("messages") or []
            for m in messages:
                from_phone = m.get("from")
                msg_type = m.get("type")
                if not from_phone or msg_type != "text":
                    continue
                text = ((m.get("text") or {}).get("body") or "").strip()
                if not text:
                    continue
                messages_out.append((from_phone, text))

    return messages_out


@app.get("/health")
def health():
    return {"status": "healthy", "service": "whatsapp-bot"}


@app.get("/debug/files")
def debug_files():
    """Debug endpoint to check if HTML files are found"""
    files_status = {}
    
    # Check site.html
    site_paths = [
        Path(__file__).parent / "site.html",
        Path("site.html"),
        Path("/app/site.html"),
    ]
    
    files_status["site.html"] = {}
    for path in site_paths:
        files_status["site.html"][str(path)] = path.exists()
    
    # Check super_admin.html
    admin_paths = [
        Path(__file__).parent / "super_admin.html",
        Path("super_admin.html"),
        Path("/app/super_admin.html"),
    ]
    
    files_status["super_admin.html"] = {}
    for path in admin_paths:
        files_status["super_admin.html"][str(path)] = path.exists()
    
    return files_status


@app.get("/")
def admin_home():
    # Try multiple possible paths for site.html
    possible_paths = [
        Path(__file__).parent / "site.html",  # Same directory as main.py
        Path("site.html"),  # Current working directory
        Path("/app/site.html"),  # Railway container path
    ]
    
    html_path = None
    for path in possible_paths:
        if path.exists():
            html_path = path
            break
    
    if not html_path:
        # Return a simple HTML error page if file not found
        return HTMLResponse(
            content="""
            <html>
                <head><title>File Not Found</title></head>
                <body>
                    <h1>site.html not found</h1>
                    <p>Checked paths:</p>
                    <ul>
                        {"".join(f"<li>{p} - {'✓' if p.exists() else '✗'}</li>" for p in possible_paths)}
                    </ul>
                </body>
            </html>
            """,
            status_code=500,
            headers={"Content-Type": "text/html; charset=utf-8"}
        )
    
    try:
        html_content = html_path.read_text(encoding="utf-8")
        # Debug: print first 100 chars to verify it's HTML
        print(f"DEBUG: HTML content starts with: {html_content[:100]}")
        
        return HTMLResponse(
            content=html_content,
            headers={"Content-Type": "text/html; charset=utf-8"}
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Error reading HTML: {str(e)}</h1></body></html>",
            status_code=500,
            headers={"Content-Type": "text/html; charset=utf-8"}
        )


def _get_session_account_id(request: Request) -> str | None:
    return request.cookies.get("sid")


@app.post("/api/signup")
async def signup(request: Request, response: Response):
    body = await request.json()
    email = str(body.get("email") or "").strip().lower()
    password = str(body.get("password") or "").strip()
    code = str(body.get("code") or "").strip()

    if not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="invalid_email")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="weak_password")
    if _find_account_by_email(email):
        raise HTTPException(status_code=409, detail="email_in_use")

    if len((_storage.get("accounts") or {})) >= _MAX_ACCOUNTS:
        raise HTTPException(status_code=429, detail="max_accounts_reached")

    codes = _storage.get("login_codes") or {}
    record = codes.get(code)
    if not record:
        raise HTTPException(status_code=401, detail="invalid_code")

    code_exp = _parse_dt(record.get("expires_at"))
    if code_exp is not None and _utcnow() >= code_exp:
        raise HTTPException(status_code=401, detail="expired_code")

    client_id = record.get("client_id")
    if not client_id:
        raise HTTPException(status_code=401, detail="invalid_code")

    client = _get_client(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="invalid_code")

    account_id = secrets.token_urlsafe(16)
    salt, pw_hash = _hash_password(password)
    now = _utcnow()

    _storage["accounts"][account_id] = {
        "email": email,
        "pw_salt": salt,
        "pw_hash": pw_hash,
        "client_id": client_id,
        "plan": client.get("plan"),
        "created_at": _dt_to_str(now),
        "expires_at": client.get("expires_at"),
    }

    # bind client ownership to this account
    client["account_id"] = account_id
    _storage["clients"][client_id] = client
    _ensure_storage_saved()

    cookie_secure = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
    response.set_cookie(
        key="sid",
        value=account_id,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 400,
    )
    return {"ok": True}


@app.post("/api/login")
async def login(request: Request, response: Response):
    body = await request.json()
    email = str(body.get("email") or "").strip().lower()
    password = str(body.get("password") or "").strip()
    found = _find_account_by_email(email)
    if not found:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    account_id, acc = found
    if not _verify_password(password, acc.get("pw_salt") or "", acc.get("pw_hash") or ""):
        raise HTTPException(status_code=401, detail="invalid_credentials")

    cookie_secure = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
    response.set_cookie(
        key="sid",
        value=account_id,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 400,
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("sid")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    account_id = _get_session_account_id(request)
    if not account_id:
        return {"authenticated": False}
    acc = _get_account(account_id)
    if not acc:
        return {"authenticated": False}
    active = _is_account_active(acc)
    return {
        "authenticated": True,
        "account_id": account_id,
        "active": active,
        "plan": acc.get("plan"),
        "expires_at": acc.get("expires_at"),
    }


@app.get("/api/client/config")
def get_client_config(request: Request):
    account_id = _get_session_account_id(request)
    if not account_id:
        raise HTTPException(status_code=401, detail="not_authenticated")
    acc = _get_account(account_id)
    if not acc:
        raise HTTPException(status_code=401, detail="not_authenticated")
    if not _is_account_active(acc):
        raise HTTPException(status_code=402, detail="plan_expired")

    client_id = acc.get("client_id")
    client = _get_client(client_id) if client_id else None
    if not client:
        raise HTTPException(status_code=401, detail="not_authenticated")

    return {
        "system_prompt": _client_setting(client, "SYSTEM_PROMPT", ""),
        "whatsapp_token": _client_setting(client, "WHATSAPP_TOKEN", ""),
        "whatsapp_phone_number_id": _client_setting(client, "WHATSAPP_PHONE_NUMBER_ID", ""),
        "whatsapp_graph_version": _client_setting(client, "WHATSAPP_GRAPH_VERSION", "v20.0"),
        "memory_max_messages": int(_client_setting(client, "MEMORY_MAX_MESSAGES", "10")),
        "plan": client.get("plan"),
        "expires_at": client.get("expires_at"),
    }


@app.post("/api/client/config")
async def set_client_config(request: Request):
    account_id = _get_session_account_id(request)
    if not account_id:
        raise HTTPException(status_code=401, detail="not_authenticated")
    acc = _get_account(account_id)
    if not acc:
        raise HTTPException(status_code=401, detail="not_authenticated")
    if not _is_account_active(acc):
        raise HTTPException(status_code=402, detail="plan_expired")

    client_id = acc.get("client_id")
    client = _get_client(client_id) if client_id else None
    if not client:
        raise HTTPException(status_code=401, detail="not_authenticated")

    body = await request.json()

    def _s(v):
        if v is None:
            return ""
        return str(v)

    client["SYSTEM_PROMPT"] = _s(body.get("system_prompt"))
    client["WHATSAPP_TOKEN"] = _s(body.get("whatsapp_token"))
    client["WHATSAPP_PHONE_NUMBER_ID"] = _s(body.get("whatsapp_phone_number_id"))
    client["WHATSAPP_GRAPH_VERSION"] = _s(body.get("whatsapp_graph_version") or "v20.0")
    client["MEMORY_MAX_MESSAGES"] = str(int(body.get("memory_max_messages") or 10))

    _storage["clients"][client_id] = client

    # sync plan and expiry onto account
    acc["plan"] = client.get("plan")
    acc["expires_at"] = client.get("expires_at")
    _storage["accounts"][account_id] = acc

    _ensure_storage_saved()
    return {"ok": True}


@app.get("/admin")
def super_admin_page():
    # Try multiple possible paths for super_admin.html
    possible_paths = [
        Path(__file__).parent / "super_admin.html",  # Same directory as main.py
        Path("super_admin.html"),  # Current working directory
        Path("/app/super_admin.html"),  # Railway container path
    ]
    
    html_path = None
    for path in possible_paths:
        if path.exists():
            html_path = path
            break
    
    if not html_path:
        raise HTTPException(status_code=500, detail="super_admin.html not found in any expected location")
    
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Content-Type": "text/html; charset=utf-8"}
    )


@app.get("/api/admin/metrics")
def admin_metrics(request: Request):
    _require_admin_master_key(request)
    clients = _storage.get("clients") or {}
    active = 0
    for c in clients.values():
        if _is_client_active(c):
            active += 1
    return {
        "clients_total": len(clients),
        "clients_active": active,
        "codes_total": len((_storage.get("login_codes") or {})),
        "storage_path": str(_STORAGE_PATH),
    }


@app.post("/api/admin/codes")
async def admin_create_code(request: Request):
    _require_admin_master_key(request)
    body = await request.json()

    plan = str(body.get("plan") or "basic").strip().lower()
    days = int(body.get("days") or 30)
    client_name = str(body.get("client_name") or "").strip()

    if plan not in ("basic", "pro"):
        raise HTTPException(status_code=400, detail="invalid_plan")

    client_id = secrets.token_urlsafe(16)
    now = _utcnow()
    expires_at = now + timedelta(days=days)

    memory = 15 if plan == "basic" else 30

    _storage["clients"][client_id] = {
        "client_name": client_name,
        "plan": plan,
        "created_at": _dt_to_str(now),
        "expires_at": _dt_to_str(expires_at),
        "MEMORY_MAX_MESSAGES": str(memory),
        "SYSTEM_PROMPT": "",
        "WHATSAPP_TOKEN": "",
        "WHATSAPP_PHONE_NUMBER_ID": "",
        "WHATSAPP_GRAPH_VERSION": "v20.0",
    }

    code = secrets.token_urlsafe(10)
    _storage["login_codes"][code] = {
        "client_id": client_id,
        "plan": plan,
        "created_at": _dt_to_str(now),
        "expires_at": _dt_to_str(expires_at),
    }

    _ensure_storage_saved()
    return {"ok": True, "code": code, "client_id": client_id, "expires_at": _dt_to_str(expires_at), "plan": plan}


@app.get("/api/admin/codes")
def admin_list_codes(request: Request):
    _require_admin_master_key(request)
    out = []
    for code, rec in ((_storage.get("login_codes") or {}).items()):
        out.append({"code": code, **rec})
    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"codes": out}


@app.post("/api/admin/clients/renew")
async def admin_renew_client(request: Request):
    _require_admin_master_key(request)
    body = await request.json()
    client_id = str(body.get("client_id") or "").strip()
    days = int(body.get("days") or 30)
    if not client_id:
        raise HTTPException(status_code=400, detail="missing_client_id")
    client = _get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="client_not_found")

    now = _utcnow()
    current_exp = _parse_dt(client.get("expires_at"))
    base = current_exp if (current_exp and current_exp > now) else now
    new_exp = base + timedelta(days=days)
    client["expires_at"] = _dt_to_str(new_exp)

    # sync expiry to account if linked
    account_id = client.get("account_id")
    if account_id and _get_account(account_id):
        _storage["accounts"][account_id]["expires_at"] = client["expires_at"]

    _storage["clients"][client_id] = client
    _ensure_storage_saved()
    return {"ok": True, "client_id": client_id, "expires_at": client.get("expires_at")}


@app.post("/api/admin/clients/plan")
async def admin_change_plan(request: Request):
    _require_admin_master_key(request)
    body = await request.json()
    client_id = str(body.get("client_id") or "").strip()
    plan = str(body.get("plan") or "").strip().lower()
    if not client_id:
        raise HTTPException(status_code=400, detail="missing_client_id")
    if plan not in ("basic", "pro"):
        raise HTTPException(status_code=400, detail="invalid_plan")
    client = _get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="client_not_found")

    client["plan"] = plan
    # default memory per plan, but don't override if user configured a custom value
    default_memory = 15 if plan == "basic" else 30
    existing_memory = str(client.get("MEMORY_MAX_MESSAGES") or "").strip()
    if existing_memory in ("", "10", "15", "30"):
        client["MEMORY_MAX_MESSAGES"] = str(default_memory)

    # sync plan to account if linked
    account_id = client.get("account_id")
    if account_id and _get_account(account_id):
        _storage["accounts"][account_id]["plan"] = plan

    _storage["clients"][client_id] = client
    _ensure_storage_saved()
    return {"ok": True, "client_id": client_id, "plan": plan, "memory_max_messages": client.get("MEMORY_MAX_MESSAGES")}


@app.post("/api/admin/codes/for-client")
async def admin_create_code_for_client(request: Request):
    _require_admin_master_key(request)
    body = await request.json()
    client_id = str(body.get("client_id") or "").strip()
    days = int(body.get("days") or 30)
    if not client_id:
        raise HTTPException(status_code=400, detail="missing_client_id")
    client = _get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="client_not_found")

    now = _utcnow()
    expires_at = now + timedelta(days=days)

    code = secrets.token_urlsafe(10)
    _storage["login_codes"][code] = {
        "client_id": client_id,
        "plan": client.get("plan"),
        "created_at": _dt_to_str(now),
        "expires_at": _dt_to_str(expires_at),
    }
    _ensure_storage_saved()
    return {"ok": True, "code": code, "client_id": client_id, "expires_at": _dt_to_str(expires_at), "plan": client.get("plan")}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "whatsapp-bot"}


@app.get("/webhook")
def verify_webhook(
    hub_mode: str | None = None,
    hub_challenge: str | None = None,
    hub_verify_token: str | None = None,
):
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN")
    if not verify_token:
        raise HTTPException(status_code=500, detail="Missing WHATSAPP_VERIFY_TOKEN")

    if hub_mode == "subscribe" and hub_verify_token == verify_token and hub_challenge:
        return PlainTextResponse(content=hub_challenge)

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    logger.info("Webhook received")

    incoming = extract_incoming_messages(body)
    if not incoming:
        return {"status": "ignored"}

    meta_phone_number_id = ((body.get("entry") or [{}])[0].get("changes") or [{}])[0].get("value")
    meta_phone_number_id = (meta_phone_number_id or {}).get("metadata") or {}
    meta_phone_number_id = meta_phone_number_id.get("phone_number_id")

    # Find which client owns this phone_number_id
    client_id = None
    for cid, c in ((_storage.get("clients") or {}).items()):
        if c.get("WHATSAPP_PHONE_NUMBER_ID") and c.get("WHATSAPP_PHONE_NUMBER_ID") == meta_phone_number_id:
            client_id = cid
            break

    if not client_id:
        logger.warning("No client mapped for incoming phone_number_id=%s", meta_phone_number_id)
        return {"status": "unmapped"}

    client = _get_client(client_id)
    if not client or not _is_client_active(client):
        logger.warning("Client inactive/expired client_id=%s", client_id)
        return {"status": "inactive"}

    token = _client_setting(client, "WHATSAPP_TOKEN", "")
    phone_number_id = _client_setting(client, "WHATSAPP_PHONE_NUMBER_ID", "")
    graph_version = _client_setting(client, "WHATSAPP_GRAPH_VERSION", "v20.0")
    system_prompt = _client_setting(client, "SYSTEM_PROMPT", "Você é um assistente útil e objetivo. Responda em pt-BR.")
    max_messages = int(_client_setting(client, "MEMORY_MAX_MESSAGES", "10"))

    if not token or not phone_number_id:
        logger.warning("Missing WhatsApp credentials for client_id=%s", client_id)
        return {"status": "missing_credentials"}

    for from_phone, text in incoming:
        try:
            history = _get_history(client_id, from_phone, max_messages)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=_build_openai_messages(system_prompt, history, text),
            )
            answer = (resp.choices[0].message.content or "").strip() or "Desculpe, não consegui responder agora."

            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": answer})

            send_whatsapp_text(
                token=token,
                phone_number_id=phone_number_id,
                graph_version=graph_version,
                to_phone=from_phone,
                text=answer,
            )
        except Exception as exc:
            logger.exception("Failed processing message from=%s", from_phone)
            # tenta responder com uma msg genérica no WhatsApp; se falhar, só ignora
            try:
                send_whatsapp_text(
                    token=token,
                    phone_number_id=phone_number_id,
                    graph_version=graph_version,
                    to_phone=from_phone,
                    text="Tive um problema para responder agora. Tente novamente em instantes.",
                )
            except Exception:
                pass
            # Return 200 anyway to reduce webhook retries; error is logged.
            continue

    return {"status": "ok"}
