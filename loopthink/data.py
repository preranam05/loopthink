"""Tokenizer, CLINC150 loading/splits, and memmapped token shards for pretraining."""
from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


# ----------------------------------------------------------------------------- tokenizer
def train_tokenizer(text_iter, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[PAD, BOS, EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(text_iter, trainer=trainer)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def encode_batch(tok, texts, max_len: int, device="cpu"):
    """Right-padded [BOS]+tokens. Returns (idx LongTensor B,T ; pad_mask BoolTensor B,T)."""
    bos = tok.token_to_id(BOS)
    ids = [[bos] + e.ids[: max_len - 1] for e in tok.encode_batch(list(texts))]
    T = max(len(x) for x in ids)
    idx = torch.zeros(len(ids), T, dtype=torch.long)
    pm = torch.zeros(len(ids), T, dtype=torch.bool)
    for i, x in enumerate(ids):
        idx[i, : len(x)] = torch.tensor(x)
        pm[i, : len(x)] = True
    return idx.to(device), pm.to(device)


# ----------------------------------------------------------------------------- CLINC150
def load_clinc(path_or_dir: str = "data") -> dict:
    path = path_or_dir if path_or_dir.endswith(".json") else os.path.join(path_or_dir, "data_full.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(CLINC_URL, path)
    with open(path) as f:
        return json.load(f)


def label_text(intent: str) -> str:
    return intent.replace("_", " ")


@dataclass
class IntentSplit:
    """Everything a fine-tune/eval run needs. `unknown_*` = queries whose intent is NOT in `intents`."""
    name: str
    intents: list[str]
    train: list[tuple[str, int]]
    val: list[tuple[str, int]]
    test: list[tuple[str, int]]
    unknown_val: list[str]
    unknown_test: list[str]
    heldout_intents: list[str]

    def to_meta(self):
        return dict(name=self.name, intents=self.intents, heldout_intents=self.heldout_intents)


def make_split(raw: dict, setup: str = "official", n_heldout: int = 30, seed: int = 0) -> IntentSplit:
    all_intents = sorted({l for _, l in raw["train"]})
    if setup == "official":
        known, held = all_intents, []
    elif setup == "heldout":
        rng = random.Random(seed)
        held = sorted(rng.sample(all_intents, n_heldout))
        known = [i for i in all_intents if i not in held]
    else:
        raise ValueError(setup)
    ix = {k: i for i, k in enumerate(known)}

    def keep(rows):
        return [(t, ix[l]) for t, l in rows if l in ix]

    if setup == "official":
        unk_val = [t for t, _ in raw["oos_val"]]
        unk_test = [t for t, _ in raw["oos_test"]]
    else:  # held-out intents are the unknowns (harder, near-OOS)
        unk_val = [t for t, l in raw["val"] if l in held]
        unk_test = [t for t, l in raw["test"] if l in held]
    return IntentSplit(setup, known, keep(raw["train"]), keep(raw["val"]), keep(raw["test"]),
                       unk_val, unk_test, held)


# ----------------------------------------------------------------------------- pretraining shards
class TokenShard:
    """Random windows from a flat uint16 token file (written by scripts/prepare_pretrain_data.py)."""

    def __init__(self, path: str, seq_len: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size: int, device="cpu"):
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i: i + self.seq_len + 1].astype(np.int64) for i in ix])
        x = torch.from_numpy(x)
        return x[:, :-1].to(device, non_blocking=True), x[:, 1:].to(device, non_blocking=True)

    def rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, st):
        self.rng.bit_generator.state = st


def write_shard(token_lists, path: str):
    arr = np.concatenate([np.asarray(t, dtype=np.uint16) for t in token_lists])
    arr.tofile(path)
    return len(arr)
