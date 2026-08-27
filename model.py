import numpy as np
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_living_mask(x):
    alpha = x[:, 3:4]
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

# the three fixed 3x3 filters
identity = torch.tensor([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=torch.float32)
sobel_x  = torch.tensor([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=torch.float32) / 8.0
sobel_y  = sobel_x.t()

# stack into (3, 3, 3): three filters, each 3x3
filters = torch.stack([identity, sobel_x, sobel_y])   # shape (3, 3, 3)
weight = filters.repeat(CHANNEL_N, 1, 1).unsqueeze(1)

class CAModel(nn.Module):
    def __init__(self, k_targets=K, embed_dim=4) -> None:
        super().__init__()
        self.channel_n = CHANNEL_N
        self.fire_rate = CELL_FIRE_RATE

        self.embed = nn.Embedding(k_targets, embed_dim)
        self.conv1 = nn.Conv2d(48 + embed_dim, 128, 1)
        self.conv2 = nn.Conv2d(128, 16, 1)
        self.perceive = nn.Conv2d(16, 48, kernel_size=3, padding=1, groups=16, bias=False)

        with torch.no_grad():
            self.perceive.weight.copy_(weight)
            self.perceive.weight.requires_grad_(False)

            self.conv2.weight.zero_()
            self.conv2.bias.zero_()

    def forward(self, x, label, step_size=1.0, fire_rate=None):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x)

        e = self.embed(label)                             # (B, E)
        e = e[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])  # (B, E, H, W)
        y = torch.cat([y, e], dim=1)                      # (B, 48+E, H, W)

        dx = self.conv1(y)
        dx = F.relu(dx)
        dx = self.conv2(dx)
        dx = dx * step_size

        if fire_rate is None:
          fire_rate = self.fire_rate

        # one Bernoulli draw per cell (shape (B, H, W, 1) broadcasts over channels)
        update_mask = (torch.rand(x[:, :1].shape, device=x.device) <= fire_rate).float()
        x = x + dx * update_mask

        post_life_mask = get_living_mask(x) # cell survives only if alive both before and after the update
        life_mask = (pre_life_mask & post_life_mask).float() # cell survives only if alive both before and after the update
        return x * life_mask

def main():
    model = CAModel(k_targets=K, embed_dim=4)
    x = torch.zeros(2, 16, 72, 72)          # batch of 2
    label = torch.tensor([0, 1])            # one per sample
    out = model(x, label)
    print(out.shape)                        # expect (2, 16, 72, 72)

if __name__ == '__main__':
    main()
