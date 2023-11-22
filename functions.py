from keras.datasets import fashion_mnist
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import InputLayer, Reshape, Conv2DTranspose, LeakyReLU, ReLU
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#Model Definitions

def define_discriminator(in_shape,modname):
	model = Sequential(name=modname)
	model.add(Conv2D(32, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_generator(latent_dim,modname):
	model = Sequential(name=modname)

	n_nodes = 128 * 8 * 8
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((8, 8, 128)))

	model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model

def define_gan(generator, discriminator,modname):

	discriminator.trainable = False

	model = Sequential(name=modname)
	model.add(generator)
	model.add(discriminator)

	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

#Extra Functions

def generate_real_samples(dataset, n_samples):
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = np.ones((n_samples, 1))
	return X, y

def generate_fake_samples(generator, latent_dim, n_samples):
  x_input = np.random.randn(latent_dim * n_samples)
  x_input = x_input.reshape(n_samples, latent_dim)
  X = generator.predict(x_input)
  y = np.zeros((n_samples, 1))
  return X, y

def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

def evaluate(epoch, generator, discriminator, dataset, latent_dim, n_samples=100):
	X_real, y_real = generate_real_samples(dataset, n_samples)
	_, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)

	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

	save_plot(x_fake, epoch)
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	generator.save(filename)