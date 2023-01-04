import numpy as np
import pandas as pd
import tensorflow as tf


import os

from sklearn.metrics import classification_report

dataset_path = os.listdir('C:\\Users\\ASUS\\Desktop\\day\\dataset')

print (dataset_path)  #what kinds of classes are in this dataset

print("Types of classes labels found: ", len(dataset_path))



class_labels = []

for item in dataset_path:
 # Get all the file names
 all_classes = os.listdir('C:\\Users\\ASUS\\Desktop\\day\\dataset' + '/' +item)
 #print(all_classes)

 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))



df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
print(df.head())
print(df.tail())

print("Total number of images in the dataset: ", len(df))

label_count = df['Labels'].value_counts()
print(label_count)

import cv2

path = 'C:\\Users\\ASUS\\Desktop\\day\\dataset\\'
dataset_path = os.listdir('C:\\Users\\ASUS\\Desktop\\day\\dataset\\')

im_size = 224

images = []
labels = []

for i in dataset_path:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)


images = np.array(images)
print('burasi')
images = images.astype('float32') / 255.0
images.shape



from sklearn.preprocessing import LabelEncoder , OneHotEncoder
y=df['Labels'].values
print(y)

y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)

y=y.reshape(-1,1)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
print(Y[:5])
print(Y[35:])




from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

#inpect the shape of the training and testing.
print(train_x.shape)#(100, 224, 224, 3)
print(train_y.shape)#(100, 2)
print(test_x.shape)#(6, 224, 224, 3)
print(test_y.shape)#(6, 2)

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

NUM_CLASSES = 2
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)


inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))


# Using model without transfer learning

outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )

model.summary()

hist = model.fit(train_x, train_y, epochs=1, verbose=2)

import matplotlib.pyplot as plt

def plot_hist(hist):
        plt.plot(hist.history["accuracy"])
        # plt.plot(hist.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()


plot_hist(hist)

from sklearn.metrics import classification_report
print(classification_report(labels, model.predict,target_names=dataset_path))

# preds = model.evaluate(test_x, test_y)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))
#
#
#
# from sklearn.metrics import classification_report
# target_names = ['evening', 'morning']
# predIdxs = model.predict(train_x, verbose = 1)
# predictions= np.argmax(predIdxs, axis=1)
#
# print(classification_report(labels, predictions,target_names=test_x.class_names))