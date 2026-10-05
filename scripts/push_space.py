"""Publish the router demo as a Hugging Face Space (free CPU hardware is enough).

  python -c "from huggingface_hub import login; login()"       # once; paste a token with write access
  python scripts/push_space.py --run runs/minilm_clinc_official_s0 --space <hf-username>/loopthink-router

The Space gets the demo, the two library files it needs and the saved model. To enable the LLM fallback,
add a secret (for example GROQ_API_KEY) in the Space settings; never put a key in a file.
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

CARD = """---
title: loopthink router
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: A small classifier decides; unsure queries go to an LLM
---

Demo for [loopthink](https://github.com/preranam05/loopthink): a 22.8M-parameter intent classifier answers
most queries itself and escalates the uncertain ones to an LLM with a shortlist. Measurements are in the
repository's `results/REPORT.md`.

The classifier is `sentence-transformers/all-MiniLM-L6-v2` fine-tuned on CLINC150 (Larson et al., EMNLP 2019,
CC BY 3.0). `model/examples.json` contains CLINC150 training queries, which the LLM prompt quotes.
"""

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="a run saved by scripts/encoder_baseline.py --save_model")
p.add_argument("--space", required=True, help="<hf-username>/<space-name>")
p.add_argument("--private", action="store_true")
p.add_argument("--dry_run", default="", help="write the Space folder here instead of uploading")
a = p.parse_args()

for f in ("router.json", "head.pt", "encoder"):
    if not os.path.exists(os.path.join(a.run, f)):
        sys.exit(f"{a.run} has no {f}. Train with: python scripts/encoder_baseline.py --setup official --out {a.run} --save_model")
if not os.path.exists(os.path.join(a.run, "examples.json")):
    from loopthink.llm import ExampleIndex
    ExampleIndex.from_run(a.run, ROOT)        # writes examples.json next to the model

d = a.dry_run or tempfile.mkdtemp()
os.makedirs(os.path.join(d, "loopthink"), exist_ok=True)
shutil.copy(os.path.join(ROOT, "demo", "router_app.py"), os.path.join(d, "app.py"))
open(os.path.join(d, "loopthink", "__init__.py"), "w").close()
for f in ("llm.py", "router.py"):
    shutil.copy(os.path.join(ROOT, "loopthink", f), os.path.join(d, "loopthink", f))
os.makedirs(os.path.join(d, "model"), exist_ok=True)
for f in ("router.json", "head.pt", "examples.json"):
    shutil.copy(os.path.join(a.run, f), os.path.join(d, "model", f))
shutil.copytree(os.path.join(a.run, "encoder"), os.path.join(d, "model", "encoder"), dirs_exist_ok=True)
open(os.path.join(d, "requirements.txt"), "w").write(
    "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch\ntransformers\nnumpy\nscikit-learn\ncertifi\n")
open(os.path.join(d, "README.md"), "w").write(CARD)
if a.dry_run:
    sys.exit(f"wrote {d}")

from huggingface_hub import HfApi  # noqa: E402
api = HfApi()
api.create_repo(a.space, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
api.upload_folder(folder_path=d, repo_id=a.space, repo_type="space")
print("space:", f"https://huggingface.co/spaces/{a.space}")
