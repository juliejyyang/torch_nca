import os
import sys
import glob
import pickle
import numpy as np
import torch
import tqdm
from config import *
from data import *
from model import *
from media_utils import *
from training_utils import *

targets = []
p = TARGET_PADDING
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for e in TARGET_EMOJIS:                      # TARGET_EMOJIS comes in via `from config import *`
    img = load_emoji(e)
    padded = np.pad(img, ((p, p), (p, p), (0, 0)))
    targets.append(padded)
target_bank = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2).to(device)  # (K, 4, H, W)

h, w = target_bank.shape[2], target_bank.shape[3]

seed = np.zeros([CHANNEL_N, h, w], np.float32)
seed[3:, h//2, w//2] = 1.0

loss_log = []

def loss_f(x, label):                          # x: (B, C, H, W)
    tgt = target_bank[label]
    return ((to_rgba(x) - tgt) ** 2).mean(dim=(1, 2, 3))

model = CAModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.1)

# loss of the bare seed, good baseline
labels = np.arange(POOL_SIZE) % K
pool = SamplePool(x=np.repeat(seed[None], POOL_SIZE, 0), label=labels)


def train_step(x0, label, iter_n):
    x = x0

    for _ in range(iter_n):             # fori_loop -> plain loop, no key threading
        x = model(x, label)

    loss = loss_f(x, label).mean()

    optimizer.zero_grad()
    loss.backward()                     # replaces value_and_grad
    with torch.no_grad():               # per-tensor grad norm, same as original
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= (param.grad.norm() + 1e-8)
    optimizer.step()
    return x.detach(), loss.item()


for i in range(TRAIN_STEPS):
    if USE_PATTERN_POOL:
        batch = pool.sample(BATCH_SIZE)
        x0 = batch.x
        labels_b = batch.label

        with torch.no_grad():
            ranks = loss_f(torch.from_numpy(x0).to(device),
                                torch.from_numpy(labels_b).to(device).long()).cpu().numpy().argsort()[::-1]

        x0 = x0[ranks]
        labels_b = labels_b[ranks]
        x0[:1] = seed
        if DAMAGE_N:
            damage = 1.0 - make_circle_mask(DAMAGE_N, h, w).cpu().numpy()[:, None]
            x0[-DAMAGE_N:] *= damage
    else:
        x0 = np.repeat(seed[None], BATCH_SIZE, 0)

    iter_n = int(np.random.randint(64, 96))
    x0_t = torch.from_numpy(x0).to(device)
    label_t = torch.from_numpy(labels_b).to(device).long()
    x_out, loss = train_step(x0_t, label_t, iter_n)
    scheduler.step()

    if USE_PATTERN_POOL:
        batch.x[:] = x_out.cpu().numpy()
        batch.label[:] = labels_b            # write reordered labels back too
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(float(loss))
    if step_i % 10 == 0:
        generate_pool_figures(pool, step_i)
    if step_i % 100 == 0:
        clear_output()
        visualize_batch(x0, x_out.cpu().numpy(), step_i)
        plot_loss(loss_log)
        save_params(model, 'train_log/%04d.pt' % step_i)
    print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(float(loss))),
          end='', flush=True)

# final checkpoint, so eval.py always has the end state to load
save_params(model, 'train_log/%04d.pt' % len(loss_log))
print('\ndone. run eval.py to render videos.')
