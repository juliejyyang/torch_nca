import io
import PIL.Image
import numpy as np
import requests
from config import *
import torch

def load_image(url, max_size=TARGET_SIZE):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
  img = np.float32(img)/255.0
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
  return load_image(url)

def to_rgba(x):
    return x[:, :4]

def to_alpha(x):
    return torch.clip(x[:, 3:4], 0.0, 1.0)

def to_rgb(x):
    rgb, a = x[:, :3], to_alpha(x)
    return 1.0 - a + rgb

def make_seed(size, n=1):
  x = torch.zeros((n, size, size, CHANNEL_N), dtype = torch.float32, device = 'cuda')
  # single living cell in the center: alpha + all hidden channels = 1
  return x.at[:, size//2, size//2, 3:].set(1.0)
