# Literature check (Phase 0) — September 2026

**Bottom line:** the exact claim — *non-convergence of a looped transformer as an out-of-scope
detector for text, benchmarked against max-softmax and energy on held-out intents, with a measured
serving-cost story* — was not found. Three recent papers are close and must be cited.

| Paper | What it does | How we differ |
|---|---|---|
| Popescu, Sáez de Ocáriz Borde, Liò — *Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts* (arXiv 2607.20519, Jul 2026) | Confidence readouts (entropy, top-1, margin) and convergence readouts (predictive KL, logit change, hidden-state displacement) for early exit; fixed-prior depth supervision beats learned gates; harder OOD arithmetic stays uncertain longer; latency on Ouro-1.4B/2.6B | Efficiency framing on synthetic tasks + LM benchmarks. No intent classification, no OOS detection, no AUROC vs MSP/energy |
| *Recurrent-Depth VLA* (arXiv 2602.07845, Feb 2026) | Robotics policy; uses latent convergence (iteration count) as an epistemic-uncertainty proxy to shorten execution horizons | Idea of convergence-as-uncertainty is **not new** — our contribution is a rigorous text-OOD evaluation of it |
| *Domain Restriction via Multi SAE Layer Transitions* (arXiv 2605.11920, May 2026) | Depth-trajectory (layer-to-layer) signals for scope detection, incl. CLINC150; performance collapses on CLINC150 at 2B | Non-looped model; a natural comparison and a warning that 150 fine-grained intents are hard for trajectory signals |

Supporting / cautionary:
- **Graves 2016 (ACT)**: ponder time did *not* increase on random / unpredictable input. OOS queries may
  converge *fast* to a wrong attractor — `evaluate.py` reports a `direction_check` for exactly this.
- **STARS** (arXiv 2605.26733, ICML 2026): looped LMs often peak then degrade with more loops; Jacobian
  spectral-radius regularisation pushes toward stable fixed points. Relevant if test-time scaling beyond r=16 collapses.
- Ouro (Zhu et al. 2025): public looped checkpoints (1.4B/2.6B) — the obvious retrofit base, but too big for the 27M story.

Design decisions taken from this:
1. Fixed deep-supervision weights (no learned halting gate) — Popescu et al.
2. Log several convergence signals (stability+margin, KL drift, hidden-state cosine), not one.
3. Kill test checks the *direction* of the difference, and the combined detector stays in the headline.

Before writing up: search Semantic Scholar for papers citing 2602.07845 and 2607.20519.
