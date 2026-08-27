# torch_nca

A PyTorch reimplementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al.), extended to grow more than one target image from a
single conditioned model. A single `CAModel` learns to grow, and optionally
sustain and self-repair, several different emoji targets, selected at
inference time by a small learned label embedding.

Each cell updates itself using only local (3x3) information -- a fixed
Sobel/identity perceive step feeds a tiny per-cell MLP -- and the whole
image emerges from many rounds of this local update rule, the same way a
single genome drives every cell in a growing organism.

## How it works

- `model.py` -- `CAModel`: perceives each cell's 3x3 neighborhood with fixed
  Sobel + identity filters, concatenates a learned embedding for the target
  label, and runs that through a 1x1-conv MLP to produce a per-cell update.
  A stochastic "fire rate" and a living-cell mask (borrowed from the original
  paper) keep growth asynchronous and let dead space stay dead.
- `data.py` -- fetches target emoji PNGs from the [noto-emoji](https://github.com/googlefonts/noto-emoji)
  repo, and holds the RGBA <-> CA-state conversion helpers.
- `training_utils.py` -- the pattern pool (`SamplePool`) that keeps the
  "Regenerating" experiment training on damaged/mature patterns instead of
  only ever starting from a fresh seed, plus figure/checkpoint helpers.
- `training.py` -- the training loop.
- `eval.py` -- loads a trained checkpoint and renders a growth (and, for the
  Regenerating experiment, damage-recovery) video per target emoji.
- `eval_interpolate.py` -- diagnostic: linearly interpolates between two
  targets' label embeddings and rolls the CA out from the in-between point,
  to check whether the model learned a structured embedding space or just
  memorized isolated points. See [`eval_interpolate.py`](eval_interpolate.py)'s
  docstring for how to read the output.

## Config

All experiment knobs live in `config.py`:

| Key | Meaning |
|---|---|
| `TARGET_EMOJIS` | list of emoji to train on; the model conditions on one shared label embedding per entry |
| `EXPERIMENT_TYPE` | `"Growing"` (grow once, no upkeep), `"Persistent"` (grow and hold), or `"Regenerating"` (grow, hold, and recover from damage) |
| `CHANNEL_N` | per-cell state size (4 visible RGBA + hidden channels) |
| `TARGET_SIZE` / `TARGET_PADDING` | target image size and the empty border padded around it for growth room |
| `POOL_SIZE` / `BATCH_SIZE` | pattern pool size and training batch size |
| `CELL_FIRE_RATE` | probability each cell applies its update on a given step (keeps updates asynchronous) |
| `TRAIN_STEPS` | number of optimizer steps `training.py` runs |

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access at runtime (to fetch the target emoji PNGs from
GitHub) and `ffmpeg` on `PATH` (used by `moviepy` in `eval.py`).

### Kaggle

```python
!git clone https://github.com/juliejyyang/torch_nca.git
%cd torch_nca
!pip install -q moviepy   # torch/numpy/pillow/requests/tqdm/matplotlib are preinstalled
```

Turn on **Internet** and a **GPU accelerator** in the notebook's settings
before running training.

## Running

```bash
python training.py         # trains for config.TRAIN_STEPS steps, checkpointing to train_log/
python eval.py              # renders a growth/regeneration video per target emoji
python eval_interpolate.py  # embedding-space structure diagnostic (see above)
```

`eval.py` and `eval_interpolate.py` read the most recent checkpoint in
`train_log/` unless you pass one explicitly (`python eval.py train_log/8000.pt`).
Outputs -- checkpoints, loss/pool figures, and rendered videos -- all land
in `train_log/`.
