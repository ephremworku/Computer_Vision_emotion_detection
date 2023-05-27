import pickle
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

image_arr_test = pickle.load(open('img_arr_test.pkl', 'rb'))
image_arr_train = pickle.load(open('img_arr_train.pkl', 'rb'))
label_test = pickle.load(open('label_test.pkl', 'rb'))
label_train = pickle.load(open('label_train.pkl', 'rb'))
image_arr_test = image_arr_test.astype('float32') / 255
image_arr_train = image_arr_train.astype('float32') / 255

print(image_arr_test.shape)
print(image_arr_train.shape)
print(label_test)
print(label_train)
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(42, 42, 3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_labels = to_categorical(label_train)
test_label = to_categorical(label_test)

model.fit(x=np.array(image_arr_train, np.float32),  y=np.array(train_labels, np.float32),
          validation_data=(np.array(image_arr_test, np.float32),  np.array(test_label, np.float32)), epochs=10,
          batch_size=64)

model.save('myModelPreTrained.h5')


