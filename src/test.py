from train import *
from ts import *
from fastai import *
from fastai.data.all import *
from fastai.vision.all import *

if __name__ == "__main__":
  learn_inf = load_learner('models/reg-7epoch.pkl')
  n = 1000
  cp = 0.65
  ts = gen_ts(n, int(n * cp), 50, std=10)
  triangulate(ts, learn_inf, viz=True)   
  matplotlib.pyplot.show()

  ts = gen_ts(n, int(n * cp), 20, std=10)
  triangulate(ts, learn_inf, viz=True)   
  matplotlib.pyplot.show()

  ts = gen_ts(n, int(n * cp), 10, std=10)
  triangulate(ts, learn_inf, viz=True)   
  matplotlib.pyplot.show()

  ts = gen_ts(n, int(n * cp), 5, std=10)
  triangulate(ts, learn_inf, viz=True)   

  matplotlib.pyplot.show()