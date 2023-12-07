#Main code goes here
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions import *

#Download the dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

#Check Shape
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

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

#Build Discriminator
discriminator = define_discriminator((32,32,3),'discriminator')
discriminator.summary()

#Build Generator
generator = define_generator(100,'Generator')
generator.summary()

#Build GAN
gan = define_gan(generator, discriminator,'GAN')
gan.summary()

batch_size = 256
n_epochs = 30
bat_per_epo = int(train_images.shape[0] / batch_size)
half_batch = int(batch_size / 2)
latent_dim = 100

for i in range(n_epochs):
    for j in range(bat_per_epo):

      X_real, y_real = generate_real_samples(train_images, half_batch)
      X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
      X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
      d_loss, _ = discriminator.train_on_batch(X, y)

      X_gan = np.random.randn(latent_dim * batch_size)
      X_gan = X_gan.reshape(batch_size, latent_dim)
      y_gan = np.ones((batch_size, 1))
      g_loss = gan.train_on_batch(X_gan, y_gan)

      print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))

    if (i+1) % 10 == 0:
      evaluate(i, generator, discriminator, train_images, latent_dim)

#thomas github test