# Running on the MacBook Pro (M5, 16 GB)

PyTorch uses the Apple GPU through the **MPS** backend. Everything auto-detects it
(`cuda > mps > cpu`); force a device with `LOOPTHINK_DEVICE=cpu|mps`.

## Setup (once)
```bash
brew install python@3.12            # or use uv / pyenv
cd loopthink
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch tokenizers numpy scikit-learn matplotlib fastapi "uvicorn[standard]" datasets huggingface_hub gradio pytest
python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
python -m pytest -q tests/
```

## 1. Measure the machine (5 min)
```bash
caffeinate -i python scripts/speedtest.py
```
Prints tokens/sec and hours for the kill test, the 1B run and the ablations. Run the same script on a
Kaggle T4 notebook and put the 1B run wherever it's faster. The Mac has no 12-hour session cap and no
weekly quota, so it's the natural home for the kill test and ablations either way.

## 2. Defaults for 16 GB
- **Precision:** `--precision auto` = fp32 on MPS (safest). `--precision fp16` tries MPS autocast +
  loss scaling; it's experimental — only switch if fp32 is too slow, and watch for NaN losses.
- **Batch:** use `--batch_size 16 --grad_accum 4` (same 32k tokens/step as the T4 config, half the
  activation memory). Open Activity Monitor → Memory; if pressure turns yellow/red, drop to `--batch_size 8 --grad_accum 8`.
- **Data:** 1B tokens = ~2 GB on disk; the 1 TB SSD is fine. `prepare_pretrain_data.py` streams FineWeb-Edu,
  so it needs internet for ~1–2 h.

## 3. Long runs
```bash
caffeinate -i python -m loopthink.pretrain --data_dir data/tok1b --out runs/pretrain \
    --preset looped-27m --tokens 1e9 --batch_size 16 --grad_accum 4 --resume --max_hours 1000
```
- `caffeinate -i` stops the Mac sleeping; keep it **plugged in**, lid open (or clamshell with external display).
- Laptops throttle when hot: raise it off the desk; expect some slowdown in the first hour.
- Checkpoints every 30 min; if anything stops the run, re-run the same command — `--resume` picks up.

## 4. Serving benchmarks
For a model this small, single-query latency can be **lower on CPU than MPS** (GPU launch overhead);
batched throughput is usually better on MPS. Benchmark both:
```bash
LOOPTHINK_DEVICE=cpu python -m loopthink.bench --runs runs/ft_official --out bench_cpu.json
LOOPTHINK_DEVICE=mps python -m loopthink.bench --runs runs/ft_official --out bench_mps.json
```
That comparison is itself a nice finding for the write-up.
