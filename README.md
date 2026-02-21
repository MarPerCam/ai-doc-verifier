# ai-doc-verifier (v3)

Sistema de verificação automática de documentos de importação/exportação.

## Visão geral

O **ai-doc-verifier** recebe **BL** + **Invoice** (e **Packing List** opcional), extrai os dados via **Gemini**, compara automaticamente e devolve um relatório com divergências.

### Stack
- **Frontend:** HTML + JavaScript puro
- **Backend:** Python + Flask
- **IA:** Gemini (Google)
- **Cache:** SQLite (`cache.db`)
  - Cache por **hash do arquivo**
  - Cache por **workflow** (combinação de hashes BL + Invoice + Packing)

---

## Funcionalidades

- Upload de documentos (BL/Invoice/Packing)
- Extração estruturada via Gemini
- Comparação automática entre documentos
- Cache:
  - **document_cache:** extração por hash do arquivo
  - **workflow_cache:** relatório final por hash do workflow
- **Reverificar**:
  - apaga **somente** o cache dos documentos enviados (por hash)
  - apaga **somente** o cache do workflow atual
  - reprocessa com Gemini e salva novamente no cache

---

## Estrutura do projeto (principal)pasta /docs..
ai-doc-verifier - v3/ ├─ frontend/ │  └─ index.html └─ backend/ ├─ api_server.py ├─ ai_backend_gemini.py ├─ database.py ├─ utils.py ├─ cache.db ├─ requirements.txt └─ .env
---

## Requisitos

- Python 3.10+ (recomendado)
- Chave de API do true

---

## Configuração (.env)

Crie/edite `backend/.env`:

```env
GEMINI_API_KEY=SEU_TOKEN_AQUI
CACHE_ENABLED=true
CACHE_TTL_DAYS=9=

Como rodar (Windows)
1) Backend (Flask)
Copiar código
Bat
cd "C:\Users\Usuário\OneDrive\Documentos\Agentes de IA\ai-doc-verifier - v3\backend"
.\venv\Scripts\activate
pip install -r requirements.txt
python api_server.py
Teste:
http://127.0.0.1:5000/api/health
2) Frontend (HTML/JS)
Em outro terminal:
Copiar código
Bat
cd "C:\Users\Usuário\OneDrive\Documentos\Agentes de IA\ai-doc-verifier - v3\frontend"
python -m http.server 5500
Abra:
http://127.0.0.1:5500/index.html
Evite abrir o HTML “duplo clique” (file://). Use o servidor para não sofrer com CORS/paths.
Endpoints
Health
GET /api/health
Retorna status do servidor e se a IA está ativa.
Processamento completo (cache-aware)
POST /api/process-complete?force=0|1
multipart/form-data:
bl (obrigatório)
invoice (obrigatório)
packing (opcional)
Regras:
force=0:
se existir workflow cache, retorna cached=true e não chama Gemini
force=1:
ignora caches e sobrescreve resultados no cache
Resposta inclui:
report
cached
workflow_hash
files (paths no servidor)
Reverificar (apaga somente cache dos arquivos atuais)
POST /api/reverify
multipart/form-data:
bl (obrigatório)
invoice (obrigatório)
packing (opcional)
Comportamento:
recebe os arquivos novamente
calcula hash
apaga somente document_cache para os hashes enviados
apaga somente workflow_cache do workflow atual
re-extrai via Gemini (force=True)
salva novamente no cache
Relatórios
GET /api/reports
Lista relatórios salvos em backend/outputs/.
GET /api/reports/<filename>
Baixa um relatório .json.
Como o cache funciona
document_cache: chave (hash do arquivo + doc_type)
evita re-extrair o mesmo documento
workflow_cache: chave (hash do workflow)
evita recomputar relatório inteiro
O botão Reverificar existe para forçar a re-extração e comparação apenas do conjunto atual, sem limpar cache.db.
Troubleshooting (rápido)
404 no navegador
O backend não precisa ter rota / por padrão. Use:
http://127.0.0.1:5000/api/health
Frontend dá 404
Você rodou o http.server na pasta errada.
Abra:
http://127.0.0.1:5500/index.html
Erro de CORS
Use o frontend via python -m http.server
Confirme que o backend está em http://127.0.0.1:5000
Próximos passos (ideias)
Botões no frontend: “Processar” e “Reverificar”
Upload múltiplo com preview e validação de extensão/tamanho
Página de histórico de relatórios (listar /api/reports)
Testes automatizados (pytest) para cache HIT/MISS e reverificação
