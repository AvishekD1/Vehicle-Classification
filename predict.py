# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:12:23 2019

@author: admin
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model2.h5'
model_weights_path = './models/weights2.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: fourwheeler")
  elif answer == 1:
    print("Label: heavy")
  elif answer == 2:
    print("Label: motorcycle")

  return answer

four_t = 0
four_f = 0
heavy_t = 0
heavy_f = 0
moto_t = 0
moto_f = 0

for i, ret in enumerate(os.walk('data/predict/fourwheeler')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: fourwheeler")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      four_t += 1
    else:
      four_f += 1

for i, ret in enumerate(os.walk('data/predict/heavy')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: heavy")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      heavy_t += 1
    else:
      heavy_f += 1

for i, ret in enumerate(os.walk('data/predict/motorcycle')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: motorcycle")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      moto_t += 1
    else:
      moto_f += 1

"""
Check metrics
"""
print("\nTrue fourwheeler: ", four_t)
print("False fourwheeler: ", four_f)
print("True heavy: ", heavy_t)
print("False heavy: ", heavy_f)
print("True motorcycle: ", moto_t)
print("False motorcycle: ", moto_f)

print("\nAccuracy",((four_t+ heavy_t+ moto_t)/ 150)* 100)



