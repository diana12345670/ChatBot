# WhatsApp ChatBot with Multi-Account Support

Sistema de chatbot para WhatsApp com IA, suporte a múltiplas contas e painel administrativo.

## 🚀 Deploy

### Railway
1. Fork este repositório
2. Conecte ao Railway
3. Configure as variáveis de ambiente em Settings → Variables
4. Deploy automático

### Render
1. Fork este repositório
2. Conecte ao Render
3. Use o arquivo `render.yaml` para configuração
4. Configure as variáveis de ambiente

## ⚙️ Variáveis de Ambiente

### Obrigatórias
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
WHATSAPP_VERIFY_TOKEN=your-webhook-verify-token-here
ADMIN_MASTER_KEY=ABEL2011
```

### Opcionais
```bash
MAX_ACCOUNTS=20
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
COOKIE_SECURE=true
CONFIG_PATH=storage.json
PORT=8000  # Railway: 8000, Render: 10000
```

## 📋 Configuração do WhatsApp

1. Acesse [Meta for Developers](https://developers.facebook.com/apps/)
2. Selecione seu App WhatsApp
3. Vá para WhatsApp → Configuration → Webhook
4. Configure:
   - **Callback URL**: `https://seu-domínio.com/webhook`
   - **Verify Token**: Use o mesmo valor de `WHATSAPP_VERIFY_TOKEN`
5. Inscreva os campos:
   - `messages`
   - `message_deliveries`
   - `message_reads`

## 🔧 Funcionalidades

- ✅ Multi-conta (até 20 clientes)
- ✅ IA com OpenAI GPT-4o-mini
- ✅ Memória contextual por conversa
- ✅ Painel administrativo
- ✅ Autenticação segura
- ✅ Webhook verification
- ✅ Health check endpoint

## 📁 Estrutura

```
├── main.py              # Backend FastAPI
├── site.html            # Frontend público
├── super_admin.html     # Painel admin
├── requirements.txt     # Dependências Python
├── Dockerfile          # Configuração Docker
├── railway.toml        # Configuração Railway
├── render.yaml         # Configuração Render
├── .env.example        # Exemplo de variáveis
└── storage.json        # Armazenamento local
```

## 🌐 Endpoints

### Públicos
- `GET /` - Site principal
- `GET /health` - Health check

### WhatsApp
- `GET /webhook` - Verificação do webhook
- `POST /webhook` - Recebimento de mensagens

### API
- `POST /api/signup` - Criar conta
- `POST /api/login` - Login
- `GET /api/config` - Configurações do cliente
- `POST /api/config` - Salvar configurações

### Admin
- `GET /admin` - Painel administrativo
- `GET /api/admin/metrics` - Métricas
- `POST /api/admin/codes` - Gerar código
- `GET /api/admin/codes` - Listar códigos

## 🛠️ Desenvolvimento Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis
export OPENAI_API_KEY=your-key
export WHATSAPP_VERIFY_TOKEN=your-token
export ADMIN_MASTER_KEY=ABEL2011
export COOKIE_SECURE=false

# Executar
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Planos

- **Básico**: R$ 69,99/mês - 15 mensagens de memória
- **Pro**: R$ 99,99/mês - 50 mensagens de memória

## 📞 Suporte

WhatsApp: +55 71 996086559
