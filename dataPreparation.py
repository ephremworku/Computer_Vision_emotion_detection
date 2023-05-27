import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY = r'D:\AI\python_code\EmotionDetaction\archive(1)'
TASK = ['test', 'train']
CATEGORY = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
draw = 0
train_data = []
test_data = []
image_size = 42
for task in TASK:
    folder = os.path.join(DIRECTORY, task)
    for category in CATEGORY:
        folder2 = os.path.join(folder, category)
        label = CATEGORY.index(category)
        for img in os.listdir(folder2):
            img_path = os.path.join(folder2, img)

            img_arr = cv2.imread(img_path)
            if draw == 0:
                plt.imshow(img_arr, cmap=plt.cm.binary)
                plt.show()
                draw = draw + 1
            try:
                img_arr = cv2.resize(img_arr, (image_size, image_size))
            except:
                continue

            if task == 'test':
                test_data.append([img_arr, label])
            elif task == 'train':
                train_data.append([img_arr, label])
        # break
print(len(test_data))
print(len(train_data))
random.shuffle(train_data)
random.shuffle(test_data)
img_arr_train = []
label_train = []
img_arr_test = []
label_test = []
for img_arr_train_feature, labelTrain in train_data:
    img_arr_train.append(img_arr_train_feature)
    label_train.append(labelTrain)
for img_arr_test_feature, labelTest in test_data:
    img_arr_test.append(img_arr_test_feature)
    label_test.append(labelTest)


plt.imshow(img_arr_train[5], cmap=plt.cm.binary)
plt.show()
img_arr_train = np.array(img_arr_train)
label_train = np.array(label_train)
img_arr_test = np.array(img_arr_test)
label_test = np.array(label_test)

print(len(img_arr_train))
print(img_arr_train.shape)

print(len(label_train))
print(img_arr_train.shape)

print(len(img_arr_test))
print(img_arr_train.shape)

print(len(label_test))
print(img_arr_train.shape)

# pickle.dump(img_arr_train, open('img_arr_train.pkl', 'wb'))
# pickle.dump(label_train, open('label_train.pkl', 'wb'))
# pickle.dump(img_arr_test, open('img_arr_test.pkl', 'wb'))
# pickle.dump(label_test, open('label_test.pkl', 'wb'))
