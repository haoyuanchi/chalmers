__author__ = 'hychi'


from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as K


lam = 1e-4

inputs = Input(shape=(N,))
encoded = Dense(N_hidden, activation='sigmoid', name='encoded')(inputs)
outputs = Dense(N, activation='linear')(encoded)

model = Model(input=inputs, output=outputs)


def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = model.get_layer('encoded').output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

    return mse + contractive

model.compile(optimizer='adam', loss=contractive_loss)
model.fit(X, X, batch_size=N_batch, nb_epoch=5)



import numpy as np

# Let's say we have minibatch of 32, and 64 hidden units
# Our input is 786 elements vector
X = np.random.randn(32, 786)
W = np.random.randn(786, 64)

Z = np.dot(W, X)
h = sigmoid(Z)  # 32x64

Wj_sqr = np.sum(W.T**2, axis=1)  # Marginalize i (note the transpose), 64x1
dhj_sqr = (h * (1 - h))**2  # Derivative of h, 32x64
J_norm = np.sum(dhj_sqr * Wj_sqr, axis=1) # 32x1, i.e. 1 jacobian norm for each data point
