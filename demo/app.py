"""Gradio Space: live loops-used counter + "I don't know this" indicator.
Expects the model folder (model.pt + tokenizer.json) next to this file under ./model,
or MODEL_DIR env var. On a HF Space, requirements: torch, tokenizers, gradio, numpy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gradio as gr
import torch

from loopthink.engine import AdaptiveClassifier
from loopthink.train_utils import get_device

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
clf = AdaptiveClassifier.from_run(MODEL_DIR, get_device())
EXAMPLES = ["book me a flight to boston on friday", "is my flight to denver on time",
            "set a timer for ten minutes", "wake me up at 6am", "how do i freeze my card",
            "what's the best way to grow tomatoes on a balcony", "who painted the ceiling of the sistine chapel"]


def run(text, max_loops, tau):
    if not text.strip():
        return "", "", ""
    d = clf.classify([text], int(max_loops), float(tau))[0]
    bar = "█" * d.loops_used + "░" * (int(max_loops) - d.loops_used)
    loops = f"### 🔁 Loops used: {d.loops_used} / {int(max_loops)}\n`{bar}`"
    if d.converged:
        verdict = f"## ✅ `{d.intent}`\nconfidence {d.confidence:.2f} · margin {d.margin:.2f}"
    else:
        verdict = (f"## 🤷 I don't know this one\nIt never settled on an answer within {int(max_loops)} loops. "
                   f"Best guess was `{d.intent}` ({d.confidence:.2f}).")
    top = "\n".join(f"- `{n}` {p:.3f}" for n, p in d.top3)
    return verdict, loops, top


with gr.Blocks(title="loopthink") as demo:
    gr.Markdown("# loopthink\nA ~27M-parameter model that decides how long to think — and notices "
                "when a question doesn't fit any answer it knows. Trained on CLINC150's 150 intents.")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Ask something", lines=2)
            ml = gr.Slider(2, 32, value=16, step=1, label="Max loops (compute budget)")
            tau = gr.Slider(0.0, 0.95, value=0.5, step=0.05, label="τ — margin needed to stop early")
            btn = gr.Button("Think", variant="primary")
            gr.Examples(EXAMPLES, txt)
        with gr.Column():
            verdict, loops, top = gr.Markdown(), gr.Markdown(), gr.Markdown()
    btn.click(run, [txt, ml, tau], [verdict, loops, top])
    txt.submit(run, [txt, ml, tau], [verdict, loops, top])

if __name__ == "__main__":
    demo.launch()
