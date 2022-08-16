import tensorflow as tf
import os
import cv2 as cv
import numpy as np

# 创建CNNmodel 对图像进行处理  提取特征
# 返回处理后的图像特征
def creat_model(data):
    images = []
    labels = []
    label = 0
    for subDirname in os.listdir(data):
        subjectPath = os.path.join(data, subDirname)
        if os.path.isdir(subjectPath):
            # 每一个文件夹下存放着一个人的照片
            for fileName in os.listdir(subjectPath):
                imgPath = os.path.join(subjectPath, fileName)
                img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
                img = np.expand_dims(img,axis=2)
                images.append(img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    print(labels)
    labels = tf.one_hot(labels,31)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(112, 92, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 5, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(31, activation='softmax')
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(images,labels)

    return model

data = 'C:\Workspace\PycharmProjects\Face_recognition\Data'
model = creat_model(data)

img = cv.imread('C:\Workspace\PycharmProjects\Face_recognition\Data\ywx\\0.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img = np.expand_dims(img,axis=0)
img = np.expand_dims(img,axis=3)
print(img.shape)
input = model.predict(img)
print(input)

