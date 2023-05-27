import pickle
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# from keras.callbacks import TensorBoard
# NAME = f'cat-vs-dog-{int(time.time())}'
# tensorboard = TensorBoard(log_dir=f'logs\\{NAME}')
image_size = 42

image_arr_test = pickle.load(open('img_arr_test.pkl', 'rb'))
image_arr_train = pickle.load(open('img_arr_train.pkl', 'rb'))
label_test = pickle.load(open('label_test.pkl', 'rb'))
label_train = pickle.load(open('label_train.pkl', 'rb'))
image_arr_test = image_arr_test.astype('float32') / 255
image_arr_train = image_arr_train.astype('float32') / 255

print(image_arr_test.shape)
print(image_arr_train.shape)
print(label_test.shape)
print(label_train.shape)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(image_size, image_size, 3)))
model.add(Dropout(0.5))

model.add(MaxPooling2D((2, 2), strides=2, name='mp1'))
model.add(Conv2D(64, (3, 3),  activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=2, name='mp2'))

model.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
model.add(Dropout(0.5))

model.add(MaxPooling2D((2, 2), strides=2, name='mp3'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_labels = to_categorical(label_train)
test_label = to_categorical(label_test)
model.summary()
model.fit(x=np.array(image_arr_train, np.float32),  y=np.array(train_labels, np.float32), validation_split=0.2,
          epochs=30, batch_size=32)

model.save('myModel.h5')

test_loss, test_acc = model.evaluate(np.array(image_arr_test, np.float32), np.array(label_test, np.float32))
print('test_acc:', test_acc)

