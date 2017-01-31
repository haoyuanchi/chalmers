
from keras.models import Sequential
import numpy as np
from kerosene.datasets import mnist
from kerosene.datasets import cifar100
from kerosene.datasets import svhn2
from kerosene.datasets import cifar10


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    ntrn = len(X_train)
    ntst = len(X_test)
    train_set = X_train[0:ntrn, :, :].reshape(-1, 1, 28, 28).astype('float32'), y_train[0:ntrn, ].flatten()
    test_set = X_test[:, :, :].reshape(-1, 1, 28, 28).astype('float32'), y_test.flatten()
    valid_set = train_set[0][50000:, :], train_set[1][50000:, ]
    train_set = train_set[0][0:50000, :], train_set[1][0:50000, ]

    rval = [train_set, valid_set, test_set]
    return rval


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    train_set = X_train[0:45000, :, :].reshape(45000, 3, 32, 32), y_train[0:45000, :].flatten()
    valid_set = X_train[45000:, :, :].reshape(5000, 3, 32, 32), y_train[45000:, :].flatten()
    test_set = X_test.reshape(-1, 3, 32, 32), y_test.flatten()
    rval = [train_set, valid_set, test_set]
    return rval


def load_cifar100():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    train_set = X_train[0:45000, :, :].reshape(45000, 3, 32, 32), y_train[0:45000, :].flatten()
    valid_set = X_train[45000:, :, :].reshape(5000, 3, 32, 32), y_train[45000:, :].flatten()
    test_set = X_test.reshape(len(X_test), 3, 32, 32), y_test.flatten()
    rval = [train_set, valid_set, test_set]
    return rval


def load_svhn():
    # street view house numbers defaults: 73,257 train / 26,032 test
    (X_train, y_train), (X_test, y_test) = svhn2.load_data()
    # have time to burn? use 'extra' and train on > 600,000 examples!
    (X_extra, y_extra), = svhn2.load_data(sets=['extra'])
    X_train = np.concatenate([X_train, X_extra])
    y_train = np.concatenate([y_train, y_extra])

    # Downsample the training dataset if specified
    train_set_len = len(X_train[1])

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in X_train]
    train_set = [x[:-(train_set_len//10)] for x in X_train]

    test_set = X_test[:, :, :].reshape(train_set_len, 3, 32, 32).astype('float32'), y_test
    rval = [train_set, valid_set, test_set]
    return rval




