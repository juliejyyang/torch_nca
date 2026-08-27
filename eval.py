"""Render growth (and, for Regenerating models, damage-recovery) videos
from a trained NCA checkpoint -- one video per target emoji.

Usage:
    python eval.py [checkpoint] [--steps N] [--out-dir DIR]

If no checkpoint is given, the newest .pt file in train_log/ is used.
Requires ffmpeg on PATH (moviepy shells out to it).
"""
import argparse
import glob
import os

import torch

from config import *
from data import make_seed, to_rgb
from model import CAModel
from media_utils import VideoWriter, zoom

EVAL_STEPS = 300    # rollout length for visualization (unrelated to training's iter_n)
DAMAGE_STEP = 150   # when to punch a hole, for Regenerating models


def latest_checkpoint():
    paths = sorted(glob.glob('train_log/*.pt'))
    if not paths:
        raise FileNotFoundError('no checkpoints found in train_log/; run training.py first')
    return paths[-1]


def frame(x):
    # x: (1, C, H, W) -> (H, W, 3) numpy in [0, 1], upscaled for viewing
    rgb = to_rgb(x)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return zoom(rgb, 4)


def punch_hole(x):
    # zero out a chunk in the middle of the pattern to test regeneration
    h, w = x.shape[2], x.shape[3]
    x = x.clone()
    x[:, :, h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] = 0.0
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', default=None)
    parser.add_argument('--steps', type=int, default=EVAL_STEPS)
    parser.add_argument('--out-dir', default='train_log/videos')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = args.checkpoint or latest_checkpoint()
    print('loading', ckpt)

    model = CAModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    size = TARGET_SIZE + 2 * TARGET_PADDING

    with torch.no_grad():
        for label_i, emoji in enumerate(TARGET_EMOJIS):
            x = make_seed(size, device=device)
            label = torch.tensor([label_i], device=device)
            code = hex(ord(emoji))[2:].lower()
            fn = os.path.join(args.out_dir, 'emoji_u%s.mp4' % code)
            with VideoWriter(fn) as vw:
                for step in range(args.steps):
                    if EXPERIMENT_TYPE == 'Regenerating' and step == DAMAGE_STEP:
                        x = punch_hole(x)
                    x = model(x, label)
                    vw.add(frame(x))
            print('wrote', fn)


if __name__ == '__main__':
    main()
