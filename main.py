#Main code goes here
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#All taken from Lab 5
#Download the dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

#Check Shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Visualize the Dataset
class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.gray)
    plt.xlabel(class_names[int(y_train[i])])
plt.show()

#Normalize Data
train_images = x_train / 255.0
test_images = x_test / 255.0