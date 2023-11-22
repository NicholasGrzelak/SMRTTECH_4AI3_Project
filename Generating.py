from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('generator_model_020.h5')

vector = np.random.randn(100 * 1)
vector = vector.reshape(1, 100)

X = model.predict(vector)
#print(X)

plt.imshow(X[0, :, :, :])
plt.show()