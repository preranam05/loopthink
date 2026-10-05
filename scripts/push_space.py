"""Publish the router and its console as a Hugging Face Space (free CPU hardware is enough).

  python -c "from huggingface_hub import login; login()"       # once; paste a token with write access
  python scripts/push_space.py --run runs/minilm_clinc_official_s0 --space <hf-username>/loopthink-router

The Space runs the same API as serve/router_api.py, with the console at its root. It gets the two library
files it needs and the saved model. To enable the LLM fallback, add a secret named GROQ_API_KEY in the Space
settings; never put a key in a file. Request text is not stored on the Space.
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
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Small model first. LLM only when needed.
---

Live console for [loopthink](https://github.com/preranam05/loopthink): a 22.8M-parameter intent classifier answers
most requests itself and escalates the uncertain ones to an LLM with a shortlist. Measurements are in the
repository's `results/REPORT.md`.

The classifier is `sentence-transformers/all-MiniLM-L6-v2` fine-tuned on CLINC150 (Larson et al., EMNLP 2019,
CC BY 3.0). `model/examples.json` contains CLINC150 training queries, which the LLM prompt quotes.
"""

DOCKERFILE = """FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \\
    pip install --no-cache-dir transformers numpy scikit-learn certifi fastapi "uvicorn[standard]" pydantic
COPY --chown=user . /app
USER user
ENV ROUTER_DIR=/app/model VAL_RATE=0.1 EXAMPLES=3 LLM_PROVIDER=groq LLM_MODEL=openai/gpt-oss-120b LLM_TIMEOUT=20 \\
    LOG_PATH=/tmp/requests.jsonl STORE_TEXT=0 HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["uvicorn", "serve.router_api:app", "--host", "0.0.0.0", "--port", "7860"]
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
for sub in ("loopthink", "serve", "model"):
    os.makedirs(os.path.join(d, sub), exist_ok=True)
open(os.path.join(d, "loopthink", "__init__.py"), "w").close()
for f in ("llm.py", "router.py"):
    shutil.copy(os.path.join(ROOT, "loopthink", f), os.path.join(d, "loopthink", f))
for f in ("router_api.py", "console.html"):
    shutil.copy(os.path.join(ROOT, "serve", f), os.path.join(d, "serve", f))
for f in ("router.json", "head.pt", "examples.json"):
    shutil.copy(os.path.join(a.run, f), os.path.join(d, "model", f))
shutil.copytree(os.path.join(a.run, "encoder"), os.path.join(d, "model", "encoder"), dirs_exist_ok=True)
open(os.path.join(d, "Dockerfile"), "w").write(DOCKERFILE)
open(os.path.join(d, "README.md"), "w").write(CARD)
if a.dry_run:
    sys.exit(f"wrote {d}")

from huggingface_hub import HfApi  # noqa: E402
api = HfApi()
api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
api.upload_folder(folder_path=d, repo_id=a.space, repo_type="space")
print("space:", f"https://huggingface.co/spaces/{a.space}")
