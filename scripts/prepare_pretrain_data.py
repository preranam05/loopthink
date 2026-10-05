"""Stream FineWeb-Edu, train a 16k BPE tokenizer, and write uint16 token shards.

  python scripts/prepare_pretrain_data.py --out /kaggle/working/tok --tokens 1e9
Writes: tokenizer.json, train.bin, val.bin, meta.json   (~2 GB for 1B tokens)
Tip: run this once as its own Kaggle notebook and save the output as a Kaggle Dataset.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loopthink.data import BOS, EOS, train_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
p.add_argument("--subset", default="sample-10BT")
p.add_argument("--vocab_size", type=int, default=16384)
p.add_argument("--tok_docs", type=int, default=200_000, help="docs used to train the tokenizer")
p.add_argument("--tokens", type=float, default=1e9)
p.add_argument("--val_tokens", type=float, default=5e6)
p.add_argument("--batch_docs", type=int, default=1000)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from datasets import load_dataset
ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True)
tok_path = os.path.join(a.out, "tokenizer.json")
if not os.path.exists(tok_path):
    print("training tokenizer...")
    train_tokenizer((r["text"] for r in itertools.islice(ds, a.tok_docs)), a.vocab_size, tok_path)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(tok_path)
eos = tok.token_to_id(EOS)
# skip the tokenizer docs so train data is disjoint from tokenizer data (not required, just clean)
it = iter(ds.skip(a.tok_docs))
t0 = time.time()
for name, target in (("val.bin", a.val_tokens), ("train.bin", a.tokens)):
    n = 0
    with open(os.path.join(a.out, name), "wb") as f:
        while n < target:
            docs = [r["text"] for r in itertools.islice(it, a.batch_docs)]
            if not docs:
                break
            ids = [e.ids + [eos] for e in tok.encode_batch(docs)]
            arr = np.concatenate([np.asarray(x, dtype=np.uint16) for x in ids])
            arr.tofile(f)
            n += len(arr)
            if name == "train.bin" and (n // a.batch_docs) % 50 == 0:
                print(f"{n/1e6:.0f}M tokens  {(time.time()-t0)/60:.1f} min", flush=True)
    print(name, n)
json.dump(dict(vocab_size=a.vocab_size, dataset=a.dataset, subset=a.subset, train_tokens=a.tokens), open(os.path.join(a.out, "meta.json"), "w"))
