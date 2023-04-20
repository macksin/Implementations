from tensorflow.keras.datasets import mnist
import numpy as np

(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape((len(x_test), -1))

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def kl_divergence_loss(y_true, y_pred):
    return K.sum(y_true * K.log(y_true / (y_pred + 1e-10) + 1e-10), axis=-1)

loaded_dec_model = load_model('dec_model.h5', custom_objects={'kl_divergence_loss': kl_divergence_loss})

q_test = loaded_dec_model.predict(x_test)
y_pred_test = q_test.argmax(1)
print(np.bincount(y_pred_test))