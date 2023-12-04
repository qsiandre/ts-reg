from ts import *
from fastai import *
from fastai.data.all import *
from fastai.vision.all import *

def triangulate(ts, learn_inf, viz=False):
  (w, s) = asap(ts)
  (x, y) = mov(ts, w, s)
  if viz:
    plt.plot(ts)
    plt.plot(x, y)
  norm_y = max_min_norm(y)
  answer = [None, None, None, None]
  for ic, color in enumerate(["blue", "yellow", "green", "red"]):
    gaf = gramian(norm_y, lambda xi, v: np.cos(np.arccos(xi) + np.pi - np.arccos(v)))
    rgb = mat_to_rgb(gaf)
    im = Image.fromarray(rgb)
    im.save("sample.png")
    (label, idx, w) = learn_inf.predict("sample.png")
    # print(label, idx, w)
    label = label.split('-')
    if idx == 0:
      break

    ldx = -1 if idx == 1 else w[idx - 1]
    rdx = -1 if idx == 4 else w[idx + 1]
    r = (float(label[0]) / 100., float(label[1]) / 100.)
    d = x[-1] - x[0]
    if ldx > rdx:
      r = (r[0] - 0.25, r[1])
    else:
      r = (r[0], r[1] + 0.25)

    if viz:
      plt.axvspan(x[0] + r[0] * d, x[0] + r[1] * d, color=color, alpha=0.2 * (ic + 1))
    answer[ic] = ((x[0] + r[0] * d, x[0] + r[1] * d), r)
    norm_y = norm_y[int(len(norm_y) * r[0]): int(len(norm_y) * r[1])]
    x = x[int(len(x) * r[0]): int(len(x) * r[1])]
  return answer

def get_label(fname):
  return str(fname).split('/')[1]

if __name__ == '__main__':
  ts = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=get_label,
    item_tfms=Resize(128)
  )
  dls = ts.dataloaders(Path('data'))
  # dls.valid.show_batch(max_n=4, nrows=1)
  # matplotlib.pyplot.show()

  learn = vision_learner(dls, resnet18, metrics=error_rate)
  learn.fine_tune(7)

  interp = ClassificationInterpretation.from_learner(learn)
  # interp.plot_confusion_matrix()
  # matplotlib.pyplot.show()

  # interp.plot_top_losses(20, nrows=5)
  # matplotlib.pyplot.show()

  learn.export('models/reg-7epoch.pkl')

  learn_inf = load_learner('models/reg-7epoch.pkl')
  n = 1000
  cp = 0.65
  ts = gen_ts(n, int(n * cp), 50, std=10)
  triangulate(ts, learn_inf, viz=True)   
  matplotlib.pyplot.show()