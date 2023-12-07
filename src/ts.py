import sys 
import numpy as np
from PIL import Image

def normalize(ts):
  return (ts - ts.mean()) / ts.std()

def gen_ts(n, xc, diff, mean=100, std=20):
  return np.append(np.random.normal(mean, std, xc), np.random.normal(mean + diff, std, n - xc))

def mov(ts, w, s):
  x = list(range(max(s, w), len(ts) + s - 1, s))
  return (x, np.array([ts[max(0, ix - w - 1): ix].mean() for ix in x]))

def roughness(ts):
  return ((ts[:-1] - ts[1:]) ** 2).mean() / len(ts)

def kurtosis(ts):
  return ((ts - ts.mean() / ts.std()) ** 4).mean()

def asap(ts, chart_w=350, pixel_p=3, max_drop=.05, k_sim=.3):
  maxp = chart_w / pixel_p
  slide = int(len(ts) / maxp)
  min_window = slide
  max_window = int(len(ts) * max_drop)
  k = kurtosis(ts)
  answer = (min_window, min(slide, min_window), roughness(mov(ts, min_window, min(slide, min_window))[1]))
  for w in range(max_window, min_window, -1):
    s = min(slide, w)
    (_, y) = mov(ts, w, s)
    mov_k = kurtosis(y)
    if abs(mov_k - k) / k < k_sim:
      continue
    mov_r = roughness(y)
    if answer[2] < mov_r:
      continue
    answer = (w, s, mov_r)
  return (answer[0], answer[1])

def mat_to_rgb(m):
    max = m.max()
    min = m.min()
    normalized = (m - min) / (max - min)
    source = np.array([23, 9, 135])
    target = np.array([240, 249, 32])
    dx = target - source
    answer = np.array([[normalized[i][j] * dx + source for j in range(len(m[i]))] for i in range(len(m))])
    return answer.astype('uint8')

def gramian(ts, fn):
  return np.array([fn(xi, ts) for xi in ts])

def max_min_norm(ts):
  return (ts - ts.max() + (ts - ts.min())) / (ts.max() - ts.min())

if __name__ == "__main__":
  n = int(sys.argv[1])
  xc = int(sys.argv[2])
  diff = int(sys.argv[3])
  for e in gen_ts(n, xc, diff):
    print(e)