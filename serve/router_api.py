"""Decision router API: small model first, LLM shortlist fallback, request log + live dashboard.

  ROUTER_DIR=runs/minilm_official uvicorn serve.router_api:app --port 8000
  open http://localhost:8000/dashboard

GET  /           interactive console (serve/console.html)
POST /decide     {"text": "...", "allow_llm": true, "min_confidence": 0.95, "top_k": 5, "explain": false}
GET  /metrics    rolling metrics (?minutes=60)     GET /dashboard   operations view
GET  /health     GET /intents     GET /config

Env: LLM_PROVIDER (ollama|openai|groq|openrouter|gemini, key in OPENAI_API_KEY etc.), ROUTER_DIR, VAL_RATE (0.05|0.1|0.2, default 0.1), OLLAMA_HOST (http://localhost:11434),
     LLM_MODEL (qwen2.5:7b), LLM_TIMEOUT (10), TOP_K (5), LOG_PATH (logs/requests.jsonl), STORE_TEXT (1),
     EXAMPLES (3): nearest training examples shown to the LLM under each shortlisted label; 0 = label names only
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from loopthink.llm import KEY_ENV, OOS, ExampleIndex, build_shortlist_prompt, shortlist_llm, shortlist_message
from loopthink.router import RequestLog, Router

CFG = dict(router_dir=os.environ.get("ROUTER_DIR", "runs/minilm_official"),
           val_rate=float(os.environ.get("VAL_RATE", "0.1")),
           provider=os.environ.get("LLM_PROVIDER", "ollama"),
           ollama=os.environ.get("OLLAMA_HOST") or os.environ.get("LLM_BASE_URL"),
           llm_model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
           llm_timeout=float(os.environ.get("LLM_TIMEOUT", "10")),
           top_k=int(os.environ.get("TOP_K", "5")),
           examples=int(os.environ.get("EXAMPLES", "3")),
           log_path=os.environ.get("LOG_PATH", "logs/requests.jsonl"),
           store_text=os.environ.get("STORE_TEXT", "1") == "1")

app = FastAPI(title="loopthink router", version="0.3.0",
              description="A 22M classifier answers most queries in milliseconds; uncertain ones go to an LLM.")
_router: Router | None = None
_log: RequestLog | None = None


_ex: ExampleIndex | None = None
# Reference figures measured on CLINC150 with a local 7B LLM (results/REPORT.md), used by the console to
# estimate what a session would have cost if every request had gone to the LLM.
REFERENCE = dict(llm_only_ms=455, llm_only_cost_per_1k=0.099, escalated_ms=755, escalated_cost_per_1k=0.082,
                 accuracy=95.2, llm_only_accuracy=78.7)


def _llm_available() -> bool:
    if not CFG["llm_model"]:
        return False
    if CFG["provider"] == "ollama":
        return True
    return bool(os.environ.get(KEY_ENV.get(CFG["provider"], "LLM_API_KEY")) or os.environ.get("LLM_API_KEY"))


def router() -> Router:
    global _router, _ex
    if _router is None:
        if CFG["examples"] > 0:
            try:
                _ex = ExampleIndex.from_run(CFG["router_dir"])
            except Exception as e:   # no training data next to the model: serve with label names only
                print(f"examples prompt disabled ({type(e).__name__}: {e})")
        CFG["examples_active"] = CFG["examples"] if _ex else 0
        llm = shortlist_llm(CFG["ollama"], CFG["llm_model"], CFG["llm_timeout"], CFG["provider"],
                            examples=_ex, n_examples=CFG["examples"]) if _llm_available() else None
        _router = Router.from_dir(CFG["router_dir"], CFG["val_rate"], llm, CFG["top_k"])
    return _router


def log() -> RequestLog:
    global _log
    if _log is None:
        _log = RequestLog(CFG["log_path"] or None, store_text=CFG["store_text"])
    return _log


class DecideReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    allow_llm: bool = Field(True, description="false = never call the LLM (latency/cost cap)")
    min_confidence: float | None = Field(None, ge=0.0, le=1.0, description="escalate at or below this confidence; "
                                         "default is the threshold validated offline")
    top_k: int | None = Field(None, ge=2, le=10, description="shortlist size shown to the LLM")
    explain: bool = Field(False, description="also return the prompt an escalated request sends to the LLM")


@app.post("/decide")
def decide(req: DecideReq):
    r = router()
    out = r.decide(req.text, req.allow_llm, req.min_confidence, req.top_k)
    log().add(req.text, out)
    if req.explain and out["escalate"]:
        cands = [t["intent"] for t in out["top"]] + [OOS]
        ex = _ex.lookup(req.text, cands, CFG["examples"]) if _ex is not None and CFG["examples"] > 0 else None
        out = dict(out, llm_prompt=dict(system=build_shortlist_prompt(bool(ex)), user=shortlist_message(req.text, cands, ex)))
    return out


@app.get("/config")
def config():
    r = router()
    return dict(intents=len(r.intents), base_model=r.meta.get("base_model"), llm_connected=r.llm is not None,
                llm_model=CFG["llm_model"] if r.llm is not None else None, min_confidence=round(-r.threshold, 3),
                presets=r.presets, top_k=r.top_k, **r.info, examples_per_label=CFG.get("examples_active", 0), reference=REFERENCE)


@app.get("/", include_in_schema=False)
def console():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "console.html"), media_type="text/html")


@app.get("/health")
def health():
    r = router()
    return dict(status="ok", intents=len(r.intents), llm_model=CFG["llm_model"] if r.llm is not None else None, llm_provider=CFG["provider"],
                examples_per_label=CFG.get("examples_active", 0), **r.meta)


@app.get("/intents")
def intents():
    return router().intents


@app.get("/metrics")
def metrics(minutes: float | None = None):
    return log().metrics(minutes)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD


DASHBOARD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Router dashboard</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--acc:#60a5fa}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}.v{font-size:26px;font-weight:600;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:6px 4px;border-bottom:1px solid var(--line)}
form{display:flex;gap:8px;margin-top:8px}input{flex:1;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{padding:8px 14px;border:0;border-radius:8px;background:var(--acc);color:#fff;cursor:pointer}pre{white-space:pre-wrap;font-size:13px}
</style></head><body><main>
<h1>Decision router</h1><div class="mut" id="sub">loading…</div>
<div class="grid">
<div class="card"><div class="k">Requests</div><div class="v" id="n">–</div></div>
<div class="card"><div class="k">Sent to LLM</div><div class="v" id="esc">–</div></div>
<div class="card"><div class="k">Out of scope</div><div class="v" id="oos">–</div></div>
<div class="card"><div class="k">Latency p50 / p95</div><div class="v" id="lat">–</div></div>
<div class="card"><div class="k">LLM failures</div><div class="v" id="fail">–</div></div>
</div>
<div class="card"><div class="k">Try a query</div>
<form id="f"><input id="q" placeholder="e.g. remind me to call the dentist tomorrow" autocomplete="off"><button>Decide</button></form><pre id="res"></pre></div>
<div class="grid"><div class="card"><div class="k">Answered by</div><table id="src"></table></div>
<div class="card"><div class="k">Top answers</div><table id="top"></table></div></div>
<script>
const pct=x=>x==null?'–':(100*x).toFixed(1)+'%';
function rows(el,pairs){el.innerHTML=pairs.map(([a,b])=>`<tr><td>${a}</td><td style="text-align:right">${b}</td></tr>`).join('')}
async function load(){try{
 const h=await (await fetch('health')).json();document.getElementById('sub').textContent=`${h.base_model} · ${h.intents} intents · escalate below ${(-h.threshold).toFixed(2)} confidence · fallback ${h.llm_model||'not connected'}`;
 const m=await (await fetch('metrics')).json();document.getElementById('n').textContent=m.requests??0;
 if(!m.requests)return;document.getElementById('esc').textContent=pct(m.escalation_rate);document.getElementById('oos').textContent=pct(m.out_of_scope_rate);
 document.getElementById('lat').textContent=`${m.latency_ms.p50} / ${m.latency_ms.p95} ms`;document.getElementById('fail').textContent=pct(m.llm_failure_rate);
 rows(document.getElementById('src'),Object.entries(m.sources));rows(document.getElementById('top'),m.top_answers);
}catch(e){document.getElementById('sub').textContent='cannot reach API: '+e}}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const t=document.getElementById('q').value;if(!t)return;
 const r=await (await fetch('decide',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:t})})).json();
 document.getElementById('res').textContent=`${r.answer}  ·  ${r.source}  ·  ${r.latency_ms} ms\\n${r.reason}\\ntop: ${r.top.map(x=>x.intent+' '+x.prob).join(', ')}`;load()};
load();setInterval(load,5000);
</script></main></body></html>"""
