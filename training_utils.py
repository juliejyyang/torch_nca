import os
import glob
import numpy as np
import matplotlib.pylab as pl
from data import *
from media_utils import *

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        # pick n random integers from _size, replace = False (no dups)
        idx = np.random.choice(self._size, n, False)

        # after choosing random rows, copy them out, wrap the copy with a return address
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx = idx)
        return batch

    # scatter back into pool.x at original indices
    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

# for regeneration, create n random filled circles as binary masks, one per batch element
# 1.0 is inside circle, 0.0 is outside, these are the "damage" masks
def make_circle_mask(n, h, w, device='cpu'):
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]

    center = torch.empty(2, n, 1, 1, device=device).uniform_(-0.5, 0.5)
    r      = torch.empty(n, 1, 1, device=device).uniform_(0.1, 0.4)

    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = (x * x + y * y < 1.0).float()
    return mask

    # JAX PRNG discipline, one key must never be used twice
    k1, k2 = torch.random.split(key)
    center = torch.empty(2, n, 1, 1, device=device).uniform_(-0.5, 0.5)
    r      = torch.empty(n, 1, 1, device=device).uniform_(0.1, 0.4)

    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = ( x * x + y * y < 1.0).astype(jnp.float32)
    return mask

import pickle

def save_params(model, fn):
    torch.save(model.state_dict(), fn)


os.makedirs('train_log', exist_ok=True)

def to_rgb_np(x):                    # x: NumPy, NCHW (N, C, H, W)
    x = np.moveaxis(x, 1, -1)        # -> (N, H, W, C), channels last for display
    rgb, a = x[..., :3], np.clip(x[..., 3:4], 0, 1)
    return 1.0 - a + rgb             # (N, H, W, 3)

def generate_pool_figures(pool, step_i):
  tiled_pool = tile2d(np.asarray(to_rgb_np(pool.x[:49])))
  fade = np.linspace(1.0, 0.0, 72)
  ones = np.ones(72)
  tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
  tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
  tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
  tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
  imwrite('train_log/%04d_pool.jpg'%step_i, tiled_pool)

def visualize_batch(x0, x, step_i):
  vis0 = np.hstack(to_rgb_np(x0))
  vis1 = np.hstack(to_rgb_np(x))
  vis = np.vstack([vis0, vis1])
  imwrite('train_log/batches_%04d.jpg'%step_i, vis)
  print('batch (before/after):')
  imshow(vis)

def plot_loss(loss_log):
  fig = pl.figure(figsize=(10, 4))
  pl.title('Loss history (log10)')
  pl.plot(np.log10(loss_log), '.', alpha=0.1)
  pl.savefig('train_log/loss.png')
  if IN_NOTEBOOK:
    pl.show()
  # 8000 steps / 100 = 80 figures otherwise, all held open by pyplot
  pl.close(fig)
