# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:50:56 2021
@author: Eriny
"""

from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from keras.models import load_model


path = './clothing-dataset/train/t-shirt/5f0a3fa0-6a3d-4b68-b213-72766a643de7.jpg'
img  = load_img(path, target_size=(150,150))
x = np.array(img)
X = np.array([x])

model = load_model('xception_v2_05_0.830.h5')

X     = preprocess_input(X)
preds = model.predict(X)

labels = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

results = dict(zip(labels, preds[0]))
print(results)