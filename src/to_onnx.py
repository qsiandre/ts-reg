from fastai.data.all import *
import onnxruntime as rt
from ts import *
from train import get_label
from fastai.vision.all import *
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


def fastai_to_pytorch(learn_inf):
  pytorch_model = learn_inf.model.eval() # gets the PyTorch model
  softmax_layer = torch.nn.Softmax(dim=1) # define softmax
  normalization_layer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization layer

  # assembling the final model
  return nn.Sequential(
    # normalization_layer,
    pytorch_model,
    softmax_layer
  )

def image_to_tensor(image, size: int) -> torch.Tensor:
  '''Helper function to transform image.'''
  # transformation pipeline
  transformation = transforms.Compose([
    transforms.Resize([size,size]), # resizes image
    transforms.ToTensor() # converts to image to tensor
  ])

  image_tensor = transformation(image).unsqueeze(0)
  print('Tensor shape: ', image_tensor.shape)

  return image_tensor

def image_transform_onnx(image, size):
  # now our image is represented by 3 layers - Red, Green, Blue
  # each layer has a 224 x 224 values representing
  image = np.array(image)
  target = [[(0, 0, 0) for i in range(size)] for j in range(size)]
  width = len(image) 
  height = len(image) 
  newWid = size
  newHt = size
  for x in range(0, newWid):  
    for y in range(0, newHt):
      srcX = int( round( float(x) / float(newWid) * float(width) ) )
      srcY = int( round( float(y) / float(newHt) * float(height) ) )
      srcX = min( srcX, width-1)
      srcY = min( srcY, height-1)
      target[x][y] = image[srcX][srcY]
  image = np.array(target)
  # print('Conversion to tensor: ',image.shape)

  # dummy input for the model at export - torch.randn(1, 3, 224, 224)
  image = image.transpose(2,0,1).astype(np.float32)
  # print('Transposing the tensor: ',image.shape)

  # our image is currently represented by values ranging between 0-255
  # we need to convert these values to 0.0-1.0 - those are the values that are expected by our model

  # print('Integer value: ', image[0][0][40])
  image /= 255
  # print('Float value: ', image[0][0][40])

  # expanding the alread existing tensor with the final dimension (similar to unsqueeze(0))
  # currently our tensor only has rank of 3 which needs to be expanded to 4 - torch.randn(1, 3, 224, 224)
  # 1 can be considered the batch size

  image = image[None, ...]
  # print('Final shape of our tensor', image.shape, '\n')
  return image

if __name__ == "__main__":
  n = 1000
  cp = .35
  ts = gen_ts(n, int(n * cp), 50, std=10)
  (w, s) = asap(ts)
  (x, y) = mov(ts, w, s)
  norm_y = max_min_norm(y)
  gaf = gramian(norm_y, lambda xi, v: np.cos(np.arccos(xi) + np.pi - np.arccos(v)))
  rgb = mat_to_rgb(gaf)
  im = Image.fromarray(rgb, mode="RGB")
  t = image_to_tensor(im, 128)
  # https://pytorch.org/vision/stable/models.html
  learn_inf = load_learner('models/reg-7epoch.pkl')
  labels = learn_inf.dls.vocab
  final_model = fastai_to_pytorch(learn_inf)
  torch.onnx.export(
    final_model, 
    torch.randn(1, 3, 128, 128),
    "models/asap.onnx",
    do_constant_folding=True,
    export_params=True, # if set to False exports untrained model
    input_names=["image_1_3_256_256"],
    output_names=["regression"],
    opset_version=16
  )
  final_model = fastai_to_pytorch(learn_inf)
  with torch.no_grad():
    results = final_model(t)
  result = labels[np.argmax(results.detach().numpy())], results.detach().numpy().astype(float)
  print("expected", cp, "got", result)
  sess = rt.InferenceSession('models/asap.onnx')
  input_name = sess.get_inputs()[0].name
  output_name = sess.get_outputs()[0].name
  input_dims = sess.get_inputs()[0].shape
  print(input_name, output_name, input_dims)
  onnx_t = image_transform_onnx(im, 128)
  results = sess.run([output_name], {input_name: onnx_t})[0]
  print(labels[np.argmax(results)], results, labels)

