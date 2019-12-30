# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:11:54 2019

@author: DELL
"""
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

custom_set = test_datagen.flow_from_directory(
		'data/validation',
		shuffle=False,
		target_size=(150,150),
		class_mode='categorical',
		batch_size=1
		)

filenames = custom_set.filenames
nb_samples = len(filenames)
print(filenames)

predict = model.predict_generator(custom_set,steps = nb_samples)
print(predict)
y_pred = np.rint(predict)
y_true = custom_set.classes

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)