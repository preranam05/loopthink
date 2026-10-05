"""Upload a trained run to the Hugging Face Hub (model repo) and optionally the Space.

  huggingface-cli login
  python scripts/push_to_hub.py --run runs/ft_official --repo <you>/loopthink-27m --space <you>/loopthink-demo
"""
import argparse, os, shutil, sys, tempfile
from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--space", default="")
p.add_argument("--private", action="store_true")
a = p.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    for f in ("model.pt", "tokenizer.json", "train_log.json"):
        if os.path.exists(os.path.join(a.run, f)):
            shutil.copy(os.path.join(a.run, f), d)
    if os.path.isdir(os.path.join(a.run, "eval")):
        shutil.copytree(os.path.join(a.run, "eval"), os.path.join(d, "eval"),
                        ignore=shutil.ignore_patterns("*.npz"))
    shutil.copy(os.path.join(root, "MODEL_CARD.md"), os.path.join(d, "README.md"))
    api.upload_folder(folder_path=d, repo_id=a.repo)
print("model:", f"https://huggingface.co/{a.repo}")
if a.space:
    api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(root, "loopthink"), os.path.join(d, "loopthink"))
        shutil.copy(os.path.join(root, "demo", "app.py"), os.path.join(d, "app.py"))
        # app.py adds '..' to sys.path; at Space root the package sits beside it, which also works
        os.makedirs(os.path.join(d, "model"))
        for f in ("model.pt", "tokenizer.json"):
            shutil.copy(os.path.join(a.run, f), os.path.join(d, "model", f))
        open(os.path.join(d, "requirements.txt"), "w").write("torch\ntokenizers\nnumpy\n")
        api.upload_folder(folder_path=d, repo_id=a.space, repo_type="space")
    print("space:", f"https://huggingface.co/spaces/{a.space}")
