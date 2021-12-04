# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:01:49 2021
@author: Eriny
"""

'''
Convolutional Neural Network (CNN):
    1. Convolutional Layers: Consists of Filters (Kind of small images --> max: 5x5)
    2. Dense Layers
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import ConvNN as nn

### 
learning_rate = 0.001
droupout      = 0.2        ## Regularization Technique Rate
image_size    = (150, 150) ## (150, 150, 3)
batch_size    = 32         ## How many image at once -- (32, 150, 150, 3)


def plotAccuracy(epochs, val, train):
    plt.figure(figsize=(6,4))
    
    plt.plot(epochs, val, color='black', linestyle='solid', label='validation')
    plt.plot(epochs, train, color='black', linestyle='dashed', label='train')
    
    plt.title(f'Xception -- lr={learning_rate}, dropout={droupout}, augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(10))
    plt.legend()
    plt.savefig('xception_v2_acc.svg')
    plt.show()


### Pre-Trained Neural Network
# the model itself (Xception)
# the preprocess_input function that takes an image and prepares it
# the decode_predictions that converts the predictions of the model into human-readable classes


### Transfer learning
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, ## 0:255 --> -1:1
                               shear_range=10.0,                        ## Augmentation:
                               zoom_range=0.1,                          ## Data Regularization- 
                               horizontal_flip=True)                    ## Technique
train_ds = train_gen.flow_from_directory("clothing-dataset/train", seed=1,
                                         target_size=image_size, batch_size=batch_size)
print(train_ds.class_indices, '\n')
X, y = next(train_ds) ## Next Batch Images
#print(y) ## One-Hot Encoded

validation_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = validation_gen.flow_from_directory("clothing-dataset/validation", seed=1,
                                            target_size=image_size, batch_size=batch_size)

### Create & Train The Model
model = nn.make_CompileModel(droupout, learning_rate)
history = nn.fitModel(train_ds, val_ds, model)

epochs = history.epoch
val = history.history['val_accuracy']
train = history.history['accuracy']

plotAccuracy(epochs, val, train)

### Test The Saved Model
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_ds = test_gen.flow_from_directory("clothing-dataset/test", seed=1,
                                       target_size=image_size, batch_size=batch_size)

loadedModel = load_model('xception_v2_05_0.830.h5')
loadedModel.evaluate(test_ds)




