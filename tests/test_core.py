import json
import os

import numpy as np
import pytest
import torch

from loopthink.data import encode_batch, train_tokenizer, load_tokenizer
from loopthink.engine import AdaptiveClassifier
from loopthink.evaluate import adaptive_exit, collect, signals
from loopthink.model import PRESETS, ModelConfig, build_model

TEXTS = ["book a flight to paris", "what is my balance", "set an alarm for 7", "freeze my card",
         "play some jazz", "how tall is everest", "remind me to call mom", "is my flight delayed"] * 3


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("tok") / "tok.json")
    train_tokenizer(iter(TEXTS * 20), 300, p)
    return load_tokenizer(p)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    m = build_model(ModelConfig(vocab_size=tok.get_vocab_size(), **PRESETS["tiny"]))
    m.set_label_table(torch.randn(10, 128))
    return m.eval()


def test_param_matched_baseline():
    looped = build_model(ModelConfig(**PRESETS["looped-27m"])).num_params()
    dense = build_model(ModelConfig(**PRESETS["dense-6"])).num_params()
    assert abs(looped - dense) / looped < 0.05


def test_eval_is_deterministic_and_batch_independent(model, tok):
    idx, pm = encode_batch(tok, TEXTS, 32)
    le = model.intent_keys()
    a = model.classify_trajectory(idx, pm, 6, le)[-1][1]
    b = model.classify_trajectory(idx[:3], pm[:3, :], 6, le)[-1][1]
    assert torch.allclose(a[:3], b, atol=1e-4)


def test_truncated_bptt_still_trains_prelude(tok):
    m = build_model(ModelConfig(vocab_size=tok.get_vocab_size(), **PRESETS["tiny"])).train()
    idx = torch.randint(3, tok.get_vocab_size(), (2, 12))
    m.lm_loss(idx, idx, n_loops=8, grad_last_k=2).backward()
    assert m.prelude[0].attn.qkv.weight.grad.abs().sum() > 0
    assert m.core[0].attn.qkv.weight.grad.abs().sum() > 0


def test_no_injection_ablation_runs(tok):
    m = build_model(ModelConfig(vocab_size=tok.get_vocab_size(), inject=False, **PRESETS["tiny"]))
    idx = torch.randint(3, tok.get_vocab_size(), (2, 12))
    assert torch.isfinite(m.lm_loss(idx, idx, n_loops=4))


@pytest.mark.parametrize("mode", ["mask", "compact"])
def test_engine_matches_offline_adaptive_exit(model, tok, mode):
    le = model.intent_keys()
    intents = [f"i{k}" for k in range(10)]
    eng = AdaptiveClassifier(model, tok, intents, le, max_len=32)
    tau, cap = 0.02, 12
    sig = signals(collect(model, tok, TEXTS, le, cap, max_len=32))
    off = adaptive_exit(sig, tau, 2, cap)
    on = eng.classify(TEXTS, max_loops=cap, tau=tau, mode=mode)
    assert [d.loops_used for d in on] == off["loops"].tolist()
    assert [d.intent_id for d in on] == off["pred"].tolist()
    assert [d.converged for d in on] == off["converged"].tolist()


def test_pretrain_resume(tmp_path, tok):
    from loopthink import pretrain
    d = tmp_path / "data"; d.mkdir()
    rng = np.random.default_rng(0)
    for n in ("train.bin", "val.bin"):
        rng.integers(3, 300, 20000).astype(np.uint16).tofile(d / n)
    base = ["--data_dir", str(d), "--out", str(tmp_path / "run"), "--preset", "tiny", "--vocab_size", "300",
            "--seq_len", "32", "--batch_size", "4", "--grad_accum", "1", "--tokens", "3000", "--precision", "fp32",
            "--log_every", "5", "--eval_every", "1000", "--eval_batches", "2", "--warmup", "5"]
    pretrain.main(base + ["--max_steps", "10"])
    pretrain.main(base + ["--resume"])
    assert os.path.exists(tmp_path / "run" / "final.pt")
    logs = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    steps = [r["step"] for r in logs if "loss" in r]
    assert steps.count(5) == 1 and 15 in steps  # resumed from step 10 rather than restarting
