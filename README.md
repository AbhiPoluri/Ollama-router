# Ollama Model Router

A lightweight FastAPI proxy that sits between your AI tools and Ollama — automatically routing requests to the right model based on task complexity.

## What it does

- **Auto-discovers** all models running in your Ollama instance
- **Routes intelligently** — simple questions hit the fast model, complex tasks hit the smart model
- **Live dashboard** at `http://localhost:4002` showing routing decisions in real time
- **Drop-in compatible** with OpenAI API, Ollama native API, Open WebUI, OpenCode, and Page Assist
- **Injects `think: false`** for Qwen reasoning models to skip unnecessary chain-of-thought on fast queries

## Architecture

```
OpenCode / Open WebUI / Page Assist
              ↓
    localhost:4001 (router proxy)
         ↙          ↘
  Fast Model      Smart Model
 (llama3.2:3b)  (qwen3.5:35b-a3b)
              ↓
    Ollama @ Mac Studio
   192.168.86.41:11434
```

## Routing Logic

The router scores only your **last user message** (not the full context) to avoid false positives from file contents injected by tools like OpenCode.

| Trigger | Lane |
|---|---|
| Message under 25 words | Fast |
| Simple keywords: "what is", "briefly", "quick" | Fast |
| Code in message (```, `def`, `class`, `import`) | Smart |
| Message over 150 words | Smart |
| Power keywords: "debug", "refactor", "analyze", "implement" | Smart |
| Default | Smart |

## Setup

### Requirements

- Python 3.11+
- Ollama running (local or remote)

### Install

```bash
git clone https://github.com/AbhiPoluri/ollama-router
cd ollama-router
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx
```

### Configure

Edit the top of `router.py`:

```python
OLLAMA_BASE  = "http://localhost:11434"   # your Ollama URL
SMART_MODEL  = "qwen3.5:35b-a3b"         # your smart model
PROXY_PORT   = 4001
DASH_PORT    = 4002
```

### Run

```bash
python router.py
```

Or use the start script:

```bash
bash start-router.sh
```

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /api/chat` | Ollama native chat |
| `POST /api/generate` | Ollama native generate |
| `GET /api/tags` | Model list (proxied from Ollama) |
| `GET /api/version` | Ollama version |
| `GET /stats` | Router stats JSON |
| `GET /` (port 4002) | Live dashboard |

## Connecting your tools

**Open WebUI:** Settings → Connections → Ollama URL → `http://localhost:4001`

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://localhost:4001/v1" }
    }
  }
}
```

**Page Assist:** Set Ollama URL to `http://localhost:4001`

## Dashboard

Open `http://localhost:4002` to see live routing decisions — which model each request hit, why, and the full prompt preview.

## License

MIT
