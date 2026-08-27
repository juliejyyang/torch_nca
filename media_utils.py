import base64
import io
import os


def in_notebook():
  # True only under an IPython kernel, not `python foo.py` or plain ipython
  try:
    from IPython import get_ipython
    return type(get_ipython()).__module__.startswith('ipykernel')
  except Exception:
    return False


IN_NOTEBOOK = in_notebook()

import matplotlib
if not IN_NOTEBOOK:
  # Default backend here is TkAgg, whose show() blocks until the window is
  # closed -- that stalls the training loop at the first plot_loss() call.
  matplotlib.use('Agg')

import matplotlib.pylab as pl
import numpy as np
import PIL.Image
import PIL.ImageDraw
from IPython.display import HTML, Image, clear_output, display
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
if IN_NOTEBOOK:
  clear_output()   # no-op outside a kernel, but skip it to keep script logs

# numpy array --> PIL Image object
def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

# saves an array as an image file (accepts filename or file object)
def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

# compresses an array to image bytes in memory
def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'   # JPEG can't store alpha
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

# array -> base64 data URL for embedding in HTML
def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# display an array inline in the notebook
def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

# pack a batch of images (N, th, tw, ...) into one grid image
def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

# nearest-neighbor upscale (blocky, good for tiny CA grids)
def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
