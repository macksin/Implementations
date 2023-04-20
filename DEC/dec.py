from time import time
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
import os

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Select 5000 random samples from the training set
indices = np.random.choice(x_train.shape[0], size=30_000, replace=False)
x_train, y_train = x_train[indices], y_train[indices]

# variables
input_size = x_train.shape[1]

# Autoencoder
autoencoder_layer_dins = [input_size, 300, 300, 1200, 10]
number_of_assymetric_layers = len(autoencoder_layer_dins) - 1

# Input layer
x = Input(shape=(input_size,), name='input')
encoder = x

# Intermediary layers
activation = 'relu'
init = 'glorot_uniform'
for i in range(number_of_assymetric_layers-1):
    encoder = Dense(
        autoencoder_layer_dins[i+1],
        activation=activation,
        kernel_initializer=init,
        name="encoder_%d" % i)(encoder)

# Hidden Layer
encoder = Dense(
    autoencoder_layer_dins[-1],
    kernel_initializer=init,
    name="encoder_%d" % (number_of_assymetric_layers - 1))(encoder)

decoder = encoder

# Final layers
activation = 'relu'
init = 'glorot_uniform'
for i in range(number_of_assymetric_layers-1, 0, -1):
    decoder = Dense(
        autoencoder_layer_dins[i],
        activation=activation,
        kernel_initializer=init,
        name="decoder_%d" % i)(decoder)

# Output
decoder = Dense(
    autoencoder_layer_dins[0],
    kernel_initializer=init,
    name="decoder_0")(decoder)


model_autoencoder = Model(inputs=x, outputs=decoder, name='autoencoder')
model_encoder = Model(inputs=x, outputs=encoder, name='encoder')

model_autoencoder.compile(optimizer='adam', loss='mse')

# Train
weights_file = "model_weights.h5"

if not os.path.exists(weights_file):
    model_autoencoder.fit(x_train, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
    model_autoencoder.save_weights(weights_file)
else:
    model_autoencoder.load_weights(weights_file)

# Predict
features = model_encoder.predict(x_test)

kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(features)
print("bincount(y_test): ", np.bincount(y_pred) / len(y_pred))

# Verify
from sklearn.metrics import adjusted_rand_score
print("adjusted_rand_score(test) = %.4f" % adjusted_rand_score(y_test, y_pred))