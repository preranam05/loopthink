"""Batched adaptive-exit inference.

Queries in one batch converge at different loops. Two strategies:
  * mode="mask":    keep running the full batch, freeze finished rows' answers (no compute saved;
                    the honest baseline for "adaptive exit under batching")
  * mode="compact": drop finished rows from the batch each loop (and trim padding), so later
                    loops run on fewer/shorter sequences — this is where real throughput comes from
  * mode="fixed":   run exactly max_loops for everyone (the fixed-r baseline)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import encode_batch


@dataclass
class Decision:
    intent: str
    intent_id: int
    confidence: float
    margin: float
    loops_used: int
    converged: bool
    top3: list


class AdaptiveClassifier:
    def __init__(self, model, tok, intents, label_emb, device="cpu", max_len=48):
        self.model, self.tok, self.intents = model.eval(), tok, intents
        self.label_emb, self.device, self.max_len = label_emb, device, max_len

    @classmethod
    def from_run(cls, run_dir, device="cpu"):
        from .finetune import load_classifier
        model, tok, st, emb = load_classifier(run_dir, device)
        return cls(model, tok, st["split"]["intents"], emb, device)

    @torch.no_grad()
    def classify(self, texts, max_loops=16, tau=0.5, patience=2, mode="compact") -> list[Decision]:
        m = self.model
        idx, pm = encode_batch(self.tok, texts, self.max_len, self.device)
        B = idx.shape[0]
        e = m.embed_prelude(idx, pm)
        if not m.cfg.is_looped:
            p = m.intent_logits(m.query_vec(e, pm), self.label_emb).float().softmax(-1)
            return [self._dec(p[i], 1, True) for i in range(B)]
        s = m.init_state(e)
        rows = torch.arange(B, device=self.device)          # original row id of each active row
        final_p = [None] * B
        loops = [max_loops] * B
        conv = [False] * B
        prev_top = torch.full((B,), -1, device=self.device, dtype=torch.long)
        stable = torch.zeros(B, device=self.device, dtype=torch.long)
        for t in range(1, max_loops + 1):
            s = m.core_step(s, e, pm)
            p = m.intent_logits(m.query_vec(s, pm), self.label_emb).float().softmax(-1)
            top2 = p.topk(2, dim=-1)
            top = top2.indices[:, 0]
            stable = torch.where(top == prev_top, stable + 1, torch.ones_like(stable))
            prev_top = top
            margin = top2.values[:, 0] - top2.values[:, 1]
            done = (stable >= patience) & (margin > tau) if mode != "fixed" else torch.zeros_like(stable, dtype=torch.bool)
            if t == max_loops:
                done_final = torch.ones_like(done)
            else:
                done_final = done
            for j in torch.nonzero(done_final).flatten().tolist():
                r = rows[j].item()
                if final_p[r] is None:
                    final_p[r], loops[r], conv[r] = p[j].cpu(), t, bool(done[j])
            if mode == "compact":
                keep = ~done_final
                if not keep.any():
                    break
                if not keep.all():
                    s, e, pm, rows = s[keep], e[keep], pm[keep], rows[keep]
                    prev_top, stable = prev_top[keep], stable[keep]
                    T = int(pm.sum(1).max())            # trim padding no longer needed
                    s, e, pm = s[:, :T], e[:, :T], pm[:, :T]
            elif mode == "mask" and all(fp is not None for fp in final_p):
                break  # everyone done: even the masked batch can stop
        return [self._dec(final_p[i], loops[i], conv[i]) for i in range(B)]

    def _dec(self, p, loops, conv):
        v, i = p.topk(min(3, p.numel()))
        return Decision(intent=self.intents[i[0]], intent_id=int(i[0]), confidence=float(v[0]),
                        margin=float(v[0] - v[1]) if len(v) > 1 else float(v[0]), loops_used=int(loops),
                        converged=bool(conv), top3=[(self.intents[k], round(float(x), 4)) for k, x in zip(i.tolist(), v)])
