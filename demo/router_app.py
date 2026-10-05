"""Gradio demo of the decision router: a 22.8M classifier answers; uncertain queries go to an LLM.

  ROUTER_DIR=runs/minilm_clinc_official_s0 python demo/router_app.py

Fallback LLM (optional): set LLM_PROVIDER (ollama|groq|openai|openrouter|gemini) and LLM_MODEL, with the key
in GROQ_API_KEY etc. Without one, the demo still shows which queries would be sent to an LLM.
On a Hugging Face Space the model sits in ./model (see scripts/push_space.py) and the key is a Space secret.
"""
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):          # Space root, or the repository root when run locally
    if os.path.isdir(os.path.join(p, "loopthink")):
        sys.path.insert(0, p)

from loopthink.llm import KEY_ENV, OOS, ExampleIndex, shortlist_llm  # noqa: E402
from loopthink.router import Router  # noqa: E402

ROUTER_DIR = os.environ.get("ROUTER_DIR", os.path.join(HERE, "model"))
PROVIDER = os.environ.get("LLM_PROVIDER") or ("groq" if os.environ.get("GROQ_API_KEY") else "")
MODEL = os.environ.get("LLM_MODEL") or {"groq": "openai/gpt-oss-120b", "ollama": "qwen2.5:7b"}.get(PROVIDER, "")
N_EXAMPLES = int(os.environ.get("EXAMPLES", "3"))
REPO = "https://github.com/preranam05/loopthink"
EXAMPLES = ["book me a flight to boston on friday", "is my flight to denver on time",
            "set a timer for ten minutes", "wake me up at 6am", "remind me to call the dentist tomorrow",
            "play some jazz in the kitchen", "what should i cook with eggs and spinach",
            "what's the best way to grow tomatoes on a balcony", "who painted the ceiling of the sistine chapel"]

_router = None
_llm_note = ""


def router():
    global _router, _llm_note
    if _router is None:
        llm = None
        if PROVIDER and MODEL and (PROVIDER == "ollama" or os.environ.get(KEY_ENV.get(PROVIDER, "LLM_API_KEY"))
                                   or os.environ.get("LLM_API_KEY")):
            try:
                ex = ExampleIndex.from_run(ROUTER_DIR) if N_EXAMPLES > 0 else None
            except Exception:
                ex = None
            llm = shortlist_llm(os.environ.get("OLLAMA_HOST") or os.environ.get("LLM_BASE_URL"), MODEL,
                                float(os.environ.get("LLM_TIMEOUT", "20")), PROVIDER, examples=ex, n_examples=N_EXAMPLES)
            _llm_note = f"Fallback LLM: {MODEL}" + (f", shown {N_EXAMPLES} training examples per candidate" if ex else "")
        else:
            _llm_note = "No fallback LLM is connected to this demo, so it only shows which queries would be sent to one."
        _router = Router.from_dir(ROUTER_DIR, float(os.environ.get("VAL_RATE", "0.1")), llm,
                                  int(os.environ.get("TOP_K", "5")))
    return _router


def render(out, llm_note=""):
    """Markdown for one decision (the dict returned by Router.decide). Pure, so it can be tested."""
    ans, src = out["answer"], out["source"]
    conf = f"{out['confidence']:.2f}"
    if src == "model":
        head = f"## `{ans}`\n**Answered by the small classifier** in {out.get('model_ms', 0):.0f} ms. No LLM call."
    elif src == "llm" and ans == OOS:
        head = (f"## Not something this assistant handles\nThe classifier was unsure (confidence {conf}), so the "
                f"query went to the LLM, which declined all of the candidates. LLM call: {out.get('llm_ms') or 0:.0f} ms.")
    elif src == "llm":
        head = (f"## `{ans}`\n**Answered by the LLM.** The classifier was unsure (confidence {conf}) and sent its "
                f"shortlist. LLM call: {out.get('llm_ms') or 0:.0f} ms.")
    elif src == "model_fallback":
        head = (f"## `{ans}`\n**The LLM call failed, so the classifier's answer is returned.** The classifier was "
                f"unsure (confidence {conf}); the service degrades instead of erroring.")
    else:
        head = (f"## Would be sent to the LLM\nThe classifier is unsure (confidence {conf}). Its best guess is "
                f"`{ans}`, but this may not be a request the assistant supports.")
    rows = "\n".join(f"| `{t['intent']}` | {t['prob']:.3f} |" for t in out["top"])
    table = "| Classifier's top guesses | Probability |\n|---|---|\n" + rows
    note = f"\n\n<sub>{html.escape(out.get('reason') or '')}{' · ' + html.escape(llm_note) if llm_note else ''}</sub>"
    return head + "\n\n" + table + note


def run(text):
    text = (text or "").strip()[:500]
    if not text:
        return ""
    r = router()
    return render(r.decide(text), _llm_note)


def build():
    import gradio as gr
    with gr.Blocks(title="loopthink router") as demo:
        gr.Markdown(
            "# When can a small model decide, and when does it need an LLM?\n"
            "A 22.8M-parameter classifier (fine-tuned MiniLM, 150 intents from CLINC150) answers most queries in a "
            "few milliseconds. When its confidence is below a threshold set on validation data, it hands the query "
            "and its top five guesses to an LLM, which picks one or says the request is out of scope.\n\n"
            f"Try a routine request, then something the assistant was never built for. "
            f"Code, measurements and the ideas that did not work: [{REPO.split('//')[1]}]({REPO})")
        txt = gr.Textbox(label="Type a request", lines=2, placeholder="e.g. remind me to call the dentist tomorrow")
        btn = gr.Button("Decide", variant="primary")
        out = gr.Markdown()
        gr.Examples(EXAMPLES, txt)
        btn.click(run, txt, out)
        txt.submit(run, txt, out)
    return demo


if __name__ == "__main__":
    build().launch()
