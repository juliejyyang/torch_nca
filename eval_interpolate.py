"""Test whether the label embedding learned a structured space or just three
isolated points.

For every pair of target emojis, linearly interpolate their learned
embedding vectors and roll the CA out from each interpolated point (never an
embedding the model was actually trained on, except at the endpoints).

  - Structured space: intermediate alphas produce a coherent, alive pattern
    that visibly morphs between the two flowers (or at least degrades
    gracefully).
  - Isolated points: the pattern dies out or turns to noise/garbage for any
    alpha strictly between 0 and 1, and only looks right exactly at the two
    trained endpoints.

Writes train_log/interp_grid.jpg (rows = emoji pairs, columns = alpha 0..1)
and prints the fraction of living cells at each alpha as a quick numeric
signal -- a fraction near zero mid-row means the pattern collapsed.

Usage:
    python eval_interpolate.py [checkpoint] [--steps N] [--alphas N]
"""
import argparse
import glob
import itertools
import os

import numpy as np
import torch

from config import *
from data import make_seed, to_alpha, to_rgb
from model import CAModel
from media_utils import imwrite, tile2d, zoom


def latest_checkpoint():
    paths = sorted(glob.glob('train_log/*.pt'))
    if not paths:
        raise FileNotFoundError('no checkpoints found in train_log/; run training.py first')
    return paths[-1]


def rollout(model, embed, size, steps, device):
    x = make_seed(size, device=device)
    with torch.no_grad():
        for _ in range(steps):
            x = model.step_with_embed(x, embed)
    return x


def frame(x):
    rgb = to_rgb(x)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return zoom(rgb, 4)


def alive_fraction(x):
    return (to_alpha(x) > 0.1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', default=None)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--alphas', type=int, default=7,
                         help='interpolation points per pair, including both endpoints')
    parser.add_argument('--out', default='train_log/interp_grid.jpg')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = args.checkpoint or latest_checkpoint()
    print('loading', ckpt)

    model = CAModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    size = TARGET_SIZE + 2 * TARGET_PADDING
    embeds = model.embed.weight.data  # (K, embed_dim), the learned per-emoji vectors
    alphas = np.linspace(0.0, 1.0, args.alphas)

    rows = []
    for i, j in itertools.combinations(range(K), 2):
        row, fracs = [], []
        for a in alphas:
            e = ((1 - a) * embeds[i] + a * embeds[j])[None].to(device)
            x = rollout(model, e, size, args.steps, device)
            row.append(frame(x))
            fracs.append(alive_fraction(x))
        rows.append(np.stack(row))
        print('%s -> %s  alive fraction @ alpha=%s:' % (TARGET_EMOJIS[i], TARGET_EMOJIS[j], np.round(alphas, 2)))
        print('  ', np.round(fracs, 3))

    grid = np.concatenate(rows, axis=0)   # (pairs * len(alphas), H, W, 3)
    tiled = tile2d(grid, w=len(alphas))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    imwrite(args.out, tiled)
    print('wrote %s  (rows = emoji pairs, columns = alpha 0..1)' % args.out)


if __name__ == '__main__':
    main()
