#!/usr/bin/env python3
"""
Model Router Proxy
Auto-discovers all Ollama models, routes by complexity.
Proxy:    http://localhost:4001
Dashboard: http://localhost:4002
"""

import re
import threading
from collections import deque, defaultdict
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_BASE  = "http://192.168.86.41:11434"
SMART_MODEL  = "qwen3.5:35b-a3b"   # override: always use this for smart lane
PROXY_PORT   = 4001
DASH_PORT    = 4002

# ── Capability profiles ───────────────────────────────────────────────────────
# Maps model name substrings to capability tags (good/bad)
CAPABILITY_PROFILES = {
    "qwen3.5":   {"good": ["coding", "reasoning", "math", "long-context"],
                  "bad":  ["web-search", "real-time"]},
    "qwen2.5":   {"good": ["coding", "reasoning", "math"],
                  "bad":  ["web-search", "real-time"]},
    "llama3.2":  {"good": ["chat", "summarization", "tool-use", "web-search"],
                  "bad":  ["deep-reasoning"]},
    "llama3":    {"good": ["chat", "summarization", "tool-use"],
                  "bad":  ["deep-reasoning"]},
    "mistral":   {"good": ["chat", "tool-use", "web-search"],
                  "bad":  ["deep-reasoning"]},
}

# Task → capability mapping for routing
TASK_CAPABILITY = {
    "web-search":     ["web search", "current events", "news", "today", "latest",
                       "real-time", "live", "price", "weather", "score"],
    "coding":         ["```", "def ", "class ", "import ", "debug", "refactor",
                       "implement", "function", "error", "traceback", "bug", "fix"],
    "reasoning":      ["analyze", "compare", "design pattern", "architecture",
                       "algorithm", "explain in detail", "step by step"],
    "math":           ["calculate", "solve", "equation", "proof", "derivative",
                       "integral", "formula"],
    "chat":           ["hello", "hi", "what is", "what's", "who is", "briefly",
                       "quick", "simple", "yes or no"],
    "summarization":  ["summarize", "summary", "tldr", "brief overview"],
}

def get_capabilities(name: str) -> dict:
    """Return capability profile for a model."""
    name_lower = name.lower()
    for key, caps in CAPABILITY_PROFILES.items():
        if key in name_lower:
            return caps
    return {"good": ["chat"], "bad": []}

def detect_task(text: str) -> str:
    """Detect the primary task type from prompt text."""
    text_lower = text.lower()
    scores = {task: sum(1 for kw in kws if kw in text_lower)
              for task, kws in TASK_CAPABILITY.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "chat"

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "models":       [],          # [{name, size_b, lane}]
    "fast_model":   "",
    "smart_model":  SMART_MODEL,
    "history":      deque(maxlen=30),
    "counts":       defaultdict(int),   # model_name → count
}

# ── Model discovery ───────────────────────────────────────────────────────────
def parse_param_count(name: str) -> float:
    """Estimate parameter count (billions) from model name."""
    name_lower = name.lower()
    # MoE active params: treat as smaller (use active param count)
    moe = re.search(r"a(\d+(?:\.\d+)?)b", name_lower)
    if moe:
        return float(moe.group(1))
    # Standard size
    m = re.search(r"(\d+(?:\.\d+)?)b", name_lower)
    if m:
        return float(m.group(1))
    return 7.0  # fallback assumption

def fetch_models() -> list:
    """Fetch and sort models from Ollama by size."""
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = r.json().get("models", [])
        result = []
        for m in models:
            name   = m["name"]
            size_b = m.get("size", 0)
            params = parse_param_count(name)
            caps   = get_capabilities(name)
            result.append({
                "name":   name,
                "size_b": size_b,
                "params": params,
                "good":   caps["good"],
                "bad":    caps["bad"],
            })
        result.sort(key=lambda x: x["params"])
        return result
    except Exception:
        return []

def refresh_models():
    """Update state with latest model list and assign fast/smart lanes."""
    models = fetch_models()
    if not models:
        return

    # Smart lane: use configured SMART_MODEL if available, else largest
    smart_names = [m["name"] for m in models]
    smart = SMART_MODEL if SMART_MODEL in smart_names else models[-1]["name"]

    # Fast lane: smallest model that isn't the smart model
    fast_candidates = [m["name"] for m in models if m["name"] != smart]
    fast = fast_candidates[0] if fast_candidates else smart

    # Annotate with lane
    for m in models:
        if m["name"] == smart:
            m["lane"] = "smart"
        elif m["name"] == fast:
            m["lane"] = "fast"
        else:
            m["lane"] = "other"

    state["models"]      = models
    state["fast_model"]  = fast
    state["smart_model"] = smart

# ── Routing logic ─────────────────────────────────────────────────────────────
SMART_KEYWORDS = [
    "debug", "refactor", "implement", "algorithm", "architecture",
    "optimize", "async", "concurrency", "design pattern", "system design",
    "explain in detail", "step by step", "complex", "advanced",
    "write a", "build a", "create a function", "analyze", "review",
    "compare", "error", "bug", "fix", "traceback", "exception",
    "class ", "def ", "import ", "```",
]

FAST_KEYWORDS = [
    "what is", "what's", "briefly", "short", "quick", "simple",
    "define", "list", "name a few", "example of",
    "yes or no", "true or false",
]

def score_prompt(messages: list) -> tuple[str, str]:
    """Returns (model_name, reason) using capability-aware routing."""
    fast  = state["fast_model"]
    smart = state["smart_model"]
    models = {m["name"]: m for m in state["models"]}

    # Score only the last user message — avoids code context in system/prior
    # messages (e.g. OpenCode) always triggering the smart lane
    user_msgs = [m for m in messages if m.get("role") == "user"]
    last = user_msgs[-1] if user_msgs else (messages[-1] if messages else {})
    last_text = (last.get("content", "") if isinstance(last.get("content"), str) else "").lower()

    # Full context still used for word-count / task detection
    full_text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in messages
    ).lower()

    word_count = len(last_text.split())
    task       = detect_task(last_text)

    # Check if smart model is bad at this task — find a better candidate
    smart_caps = models.get(smart, {})
    if task in smart_caps.get("bad", []):
        for m in state["models"]:
            if task in m.get("good", []) and m["name"] != smart:
                return m["name"], f"capability: {task}"
        return fast, f"smart bad at {task}"

    # Fast model triggers
    fast_score  = sum(1 for kw in FAST_KEYWORDS if kw in last_text)
    smart_score = sum(1 for kw in SMART_KEYWORDS if kw in last_text)

    # Code in the last user message → smart (not just in context)
    if "```" in last_text or "def " in last_text or "class " in last_text or "import " in last_text:
        return smart, "contains code"
    if word_count > 150:
        return smart, f"long prompt ({word_count}w)"
    if smart_score > fast_score and smart_score > 0:
        return smart, f"task: {task}"
    if fast_score > 0 and word_count < 60:
        return fast, "simple query"
    if word_count < 25:
        return fast, f"short prompt ({word_count}w)"
    return smart, f"default → smart"

# ── Proxy app (port 4001) ─────────────────────────────────────────────────────
proxy = FastAPI()
proxy.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@proxy.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body     = await request.json()
    messages = body.get("messages", [])
    model, reason = score_prompt(messages)

    lane = "smart" if model == state["smart_model"] else "fast"
    state["counts"][model] += 1

    preview = ""
    if messages:
        last = messages[-1].get("content", "")
        if isinstance(last, str):
            preview = last[:300] + ("…" if len(last) > 300 else "")

    state["history"].appendleft({
        "time":    datetime.now().strftime("%H:%M:%S"),
        "model":   model,
        "lane":    lane,
        "reason":  reason,
        "preview": preview,
    })

    body["model"] = model

    # Disable thinking for smart (Qwen) model
    if model == state["smart_model"]:
        body["think"] = False

    stream = body.get("stream", False)

    if stream:
        client = httpx.AsyncClient(timeout=300)
        async def _stream_completions():
            try:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()
        return StreamingResponse(_stream_completions(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_BASE}/v1/chat/completions", json=body)
            return JSONResponse(r.json())

# ── Ollama-native API endpoints (needed for Page Assist / OpenWebUI) ──────────

@proxy.get("/api/tags")
async def api_tags():
    """Proxy model list from Ollama."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_BASE}/api/tags")
        return JSONResponse(r.json())

@proxy.get("/api/version")
async def api_version():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_BASE}/api/version")
        return JSONResponse(r.json())

@proxy.post("/api/chat")
async def api_chat(request: Request):
    """Ollama /api/chat with routing applied."""
    body     = await request.json()
    messages = body.get("messages", [])
    model, reason = score_prompt(messages)

    lane = "smart" if model == state["smart_model"] else "fast"
    state["counts"][model] += 1

    preview = ""
    if messages:
        last = messages[-1].get("content", "")
        if isinstance(last, str):
            preview = last[:300] + ("…" if len(last) > 300 else "")

    state["history"].appendleft({
        "time":    datetime.now().strftime("%H:%M:%S"),
        "model":   model,
        "lane":    lane,
        "reason":  reason,
        "preview": preview,
    })

    body["model"] = model
    if model == state["smart_model"]:
        body["think"] = False

    stream = body.get("stream", True)
    if stream:
        client = httpx.AsyncClient(timeout=300)
        async def _stream_chat():
            try:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/api/chat",
                    json=body, headers={"Content-Type": "application/json"},
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()
        return StreamingResponse(_stream_chat(), media_type="application/x-ndjson")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=body)
            return JSONResponse(r.json())

@proxy.post("/api/generate")
async def api_generate(request: Request):
    """Ollama /api/generate — proxy with model override."""
    body = await request.json()
    body["model"] = state["smart_model"]
    if body["model"] == state["smart_model"]:
        body["think"] = False
    stream = body.get("stream", True)
    if stream:
        client = httpx.AsyncClient(timeout=300)
        async def _stream_generate():
            try:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/api/generate",
                    json=body, headers={"Content-Type": "application/json"},
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()
        return StreamingResponse(_stream_generate(), media_type="application/x-ndjson")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_BASE}/api/generate", json=body)
            return JSONResponse(r.json())

@proxy.get("/stats")
async def get_stats():
    fast_total  = sum(v for k, v in state["counts"].items() if k == state["fast_model"])
    smart_total = sum(v for k, v in state["counts"].items() if k == state["smart_model"])
    return {
        "fast":        fast_total,
        "smart":       smart_total,
        "fast_model":  state["fast_model"],
        "smart_model": state["smart_model"],
        "models":      state["models"],
        "counts":      dict(state["counts"]),
        "history":     list(state["history"]),
    }

# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Router</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace;
    padding: 28px;
    min-height: 100vh;
  }
  header { display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }
  header h1 { font-size: 15px; letter-spacing: 3px; text-transform: uppercase; color: #7d8590; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #3fb950; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  /* ── Route cards ── */
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 22px 24px; position: relative; overflow: hidden;
  }
  .card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .card.fast::before  { background: #3fb950; }
  .card.smart::before { background: #79c0ff; }
  .card-lane  { font-size:11px; color:#7d8590; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
  .card-model { font-size:12px; color:#8b949e; margin-bottom:14px; }
  .card-count { font-size:52px; font-weight:700; line-height:1; margin-bottom:16px; }
  .card.fast  .card-count { color: #3fb950; }
  .card.smart .card-count { color: #79c0ff; }
  .card-pct { font-size:12px; color:#7d8590; margin-bottom:8px; }
  .bar-track { background:#21262d; border-radius:4px; height:6px; overflow:hidden; }
  .bar-fill  { height:100%; border-radius:4px; transition:width .6s ease; }
  .card.fast  .bar-fill { background: #3fb950; }
  .card.smart .bar-fill { background: #79c0ff; }

  /* ── All models panel ── */
  .models-panel {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 24px;
  }
  .panel-header { font-size:11px; color:#7d8590; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px; }
  .model-row {
    display: grid;
    grid-template-columns: 160px 55px 70px 1fr 80px;
    align-items: start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #21262d;
    font-size: 12px;
  }
  .model-row:last-child { border-bottom: none; }
  .model-name { color: #c9d1d9; word-break: break-all; }
  .model-params { color: #7d8590; font-size: 11px; }
  .model-requests { color: #e6edf3; font-weight: 600; text-align: right; }
  .caps { display: flex; flex-wrap: wrap; gap: 4px; }
  .cap-tag {
    display: inline-block; padding: 1px 6px; border-radius: 10px;
    font-size: 10px; font-weight: 600;
  }
  .cap-tag.good { background: #1a3a22; color: #3fb950; }
  .cap-tag.bad  { background: #3a1a1a; color: #f85149; }
  .lane-badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 10px; font-weight: 700; text-transform: uppercase; text-align: center;
  }
  .lane-badge.fast  { background: #1a3a22; color: #3fb950; }
  .lane-badge.smart { background: #1a2a3f; color: #79c0ff; }
  .lane-badge.other { background: #21262d; color: #7d8590; }

  /* ── Log ── */
  .log { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px 24px; }
  .log-header { font-size:11px; color:#7d8590; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px; }
  .entry {
    display: grid; grid-template-columns: 60px 62px 1fr auto;
    align-items: start; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #21262d;
    font-size: 12px; animation: slide-in .25s ease;
  }
  @keyframes slide-in { from{opacity:0;transform:translateY(-6px)} to{opacity:1} }
  .entry:last-child { border-bottom: none; }
  .entry-time { color:#7d8590; font-size:11px; padding-top:2px; }
  .badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:11px; font-weight:700; text-align:center;
    text-transform:uppercase; letter-spacing:.5px;
  }
  .badge.fast  { background:#1a3a22; color:#3fb950; }
  .badge.smart { background:#1a2a3f; color:#79c0ff; }
  .entry-preview { color:#c9d1d9; white-space:pre-wrap; word-break:break-word; line-height:1.5; }
  .entry-reason  { color:#484f58; font-size:11px; text-align:right; white-space:nowrap; padding-top:2px; }
  .empty { color:#484f58; font-size:13px; padding:8px 0; }
</style>
</head>
<body>
<header><div class="dot"></div><h1>Model Router</h1></header>

<div class="cards">
  <div class="card fast">
    <div class="card-lane">Fast Lane</div>
    <div class="card-model" id="fast-model">—</div>
    <div class="card-count" id="fast-count">0</div>
    <div class="card-pct" id="fast-pct">0% of requests</div>
    <div class="bar-track"><div class="bar-fill" id="fast-bar" style="width:0%"></div></div>
  </div>
  <div class="card smart">
    <div class="card-lane">Smart Lane</div>
    <div class="card-model" id="smart-model">—</div>
    <div class="card-count" id="smart-count">0</div>
    <div class="card-pct" id="smart-pct">0% of requests</div>
    <div class="bar-track"><div class="bar-fill" id="smart-bar" style="width:0%"></div></div>
  </div>
</div>

<div class="models-panel">
  <div class="panel-header">Available Models</div>
  <div id="models-list"><div class="empty">Loading…</div></div>
</div>

<div class="log">
  <div class="log-header">Live Request Feed</div>
  <div id="history"><div class="empty">Waiting for requests…</div></div>
</div>

<script>
async function refresh() {
  try {
    const res  = await fetch('http://localhost:4001/stats');
    const data = await res.json();
    const total = (data.fast + data.smart) || 1;

    document.getElementById('fast-model').textContent  = data.fast_model  || '—';
    document.getElementById('smart-model').textContent = data.smart_model || '—';
    document.getElementById('fast-count').textContent  = data.fast;
    document.getElementById('smart-count').textContent = data.smart;

    const fp = Math.round(data.fast  / total * 100);
    const sp = Math.round(data.smart / total * 100);
    document.getElementById('fast-pct').textContent  = fp + '% of requests';
    document.getElementById('smart-pct').textContent = sp + '% of requests';
    document.getElementById('fast-bar').style.width  = fp + '%';
    document.getElementById('smart-bar').style.width = sp + '%';

    // Models list
    const ml = document.getElementById('models-list');
    if (data.models && data.models.length) {
      ml.innerHTML = data.models.map(m => {
        const reqs     = data.counts[m.name] || 0;
        const gb       = m.size_b ? (m.size_b / 1e9).toFixed(1) + ' GB' : m.params + 'B';
        const goodTags = (m.good || []).map(c => `<span class="cap-tag good">${c}</span>`).join('');
        const badTags  = (m.bad  || []).map(c => `<span class="cap-tag bad">✕ ${c}</span>`).join('');
        return `<div class="model-row">
          <span class="model-name">${m.name}</span>
          <span class="model-params">${gb}</span>
          <span class="lane-badge ${m.lane}">${m.lane}</span>
          <div class="caps">${goodTags}${badTags}</div>
          <span class="model-requests">${reqs} req${reqs !== 1 ? 's' : ''}</span>
        </div>`;
      }).join('');
    } else {
      ml.innerHTML = '<div class="empty">No models found</div>';
    }

    // History
    const hist = document.getElementById('history');
    if (!data.history.length) {
      hist.innerHTML = '<div class="empty">Waiting for requests…</div>';
      return;
    }
    hist.innerHTML = data.history.map(e => `
      <div class="entry">
        <span class="entry-time">${e.time}</span>
        <span class="badge ${e.lane}">${e.lane}</span>
        <span class="entry-preview">${e.preview || '(no preview)'}</span>
        <span class="entry-reason">${e.reason}</span>
      </div>
    `).join('');
  } catch(_) {}
}

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""

# ── Dashboard app (port 4002) ──────────────────────────────────────────────────
dash = FastAPI()

@dash.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASH_HTML

# ── Launch ─────────────────────────────────────────────────────────────────────
def run_dash():
    uvicorn.run(dash, host="0.0.0.0", port=DASH_PORT, log_level="error")

if __name__ == "__main__":
    print("  Fetching models from Ollama…")
    refresh_models()
    print(f"  Fast  → {state['fast_model']}")
    print(f"  Smart → {state['smart_model']}")
    print(f"  All models: {[m['name'] for m in state['models']]}")
    print(f"  Proxy     → http://localhost:{PROXY_PORT}")
    print(f"  Dashboard → http://localhost:{DASH_PORT}")
    t = threading.Thread(target=run_dash, daemon=True)
    t.start()
    uvicorn.run(proxy, host="0.0.0.0", port=PROXY_PORT, log_level="warning")
