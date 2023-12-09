import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential,load_model

model = load_model('generator_model_1000.h5')

vector = np.random.randn(100 * 1)
vector = vector.reshape(1, 100)

X = model.predict(vector)

plt.imshow(X[0, :, :, :])
plt.show()