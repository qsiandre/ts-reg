from ts import *
import os
from PIL import Image

def get_class(i):
  xc = int(i / 50) * 50
  pct = xc / 1000
  if pct == 0:
    return '0'
  if pct <= 0.25:
    return '0-25'
  if pct <= .50:
    return '25-50'
  if pct <= .75:
    return '50-75'
  return '75-100'

def gen_samples(n, base):
  for i in range(0, n):
    if i % 100 == 0:
      print(i)
    xc = int(i / 50) * 50
    diff = [15, 20, 30, 50, 100][int(i % 5)]
    ts = gen_ts(n, xc, diff, std=10)
    (w, s) = asap(normalize(ts))
    (_, y) = mov(ts, w, s)
    norm_y = max_min_norm(normalize(y))
    g = gramian(norm_y, lambda xi, v: np.cos(np.arccos(xi) + np.pi - np.arccos(v)))
    diff = 0 if xc == 0 else diff
    classname = get_class(i)
    rgb = mat_to_rgb(g)
    im = Image.fromarray(rgb)
    name = '_'.join([classname, str(i), str(n), str(xc), str(diff)])
    directory = '/'.join([base, classname])
    if not os.path.exists(directory):
      os.makedirs(directory)
    im.save('/'.join([directory, name + ".png"]))
    
if __name__ == '__main__':
  gen_samples(1000, 'data')