__author__ = 'hychi'

from load_data import load_svhn, load_mnist, load_cifar10, load_cifar100
import theano
import theano.tensor as T
import lasagne
import nolearn

import numpy as np


import time
from scipy.misc import imresize


feature_width = 256
feature_height = 256


def cropImage(im):
    im2 = np.dstack(im).astype(np.float32)
    # return centered 128x128 from original 250x250 (40% of area)
    # newim = im2[61:189, 61:189]
    sized1 = imresize(im2[:, :, 0:3], (feature_width, feature_height), interp="bicubic", mode="RGB")
    im_resize = np.asarray([sized1[:, :, 0], sized1[:, :, 1], sized1[:, :, 2]])
    x_start = np.random.randint(1, 29)
    y_start = np.random.randint(1, 29)
    im_crop = im_resize[:, x_start : (x_start+227), y_start : (y_start+227)]
    del im2 , sized1, im_resize

    im_crop = im_crop / np.float32(255.0)
    im_crop = np.subtract(np.multiply(2., im_crop), 1.)
    return im_crop


def load_dataset(which_data):
    which_data = which_data.lower()

    if which_data not in ('mnist', 'cifar10', 'svhn', 'cifar100'):
        return 'Need to choose corrrect dataset either "mnist", "svhn", or "cifar10"'

    print ('Loading %s data...' % which_data.upper())
    if which_data=='mnist':
        datasets = load_mnist()

    elif which_data=='svhn':
        datasets = load_svhn()

    elif which_data=='cifar10':
        datasets = load_cifar10()

    elif which_data=='cifar100':
        datasets = load_cifar100()

    return datasets


# A function which shuffles a dataset
def shuffle(X, y):
    shuffle_parts = 1
    chunk_size = len(X) / shuffle_parts
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, which_data,
         batch_size, num_epochs, hash_bit,
         save_model_path, save_hash_path):
    network = [
        # layer dealing with the input data
        ('input', lasagne.layers.InputLayer, {'shape': (None, 3, 32, 32)}),

        # first stage of our convolutional layers
        ('conv1_1', lasagne.layers.Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
        ('conv1_2', lasagne.layers.Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
        ('pool1', lasagne.layers.MaxPool2DLayer, {'pool_size': 2}),
        ('bn1', lasagne.layers.BatchNormLayer),

        ('conv2_1', lasagne.layers.Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
        ('conv2_2', lasagne.layers.Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
        ('pool2', lasagne.layers.MaxPool2DLayer, {'pool_size': 2}),
        ('bn2', lasagne.layers.BatchNormLayer),

        ('conv3_1', lasagne.layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
        ('conv3_1', lasagne.layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
        ('pool3', lasagne.layers.MaxPool2DLayer, {'pool_size': 2}),
        ('bn3', lasagne.layers.BatchNormLayer),

        ('fc4', lasagne.layers.DenseLayer, {'num_units': 1024}),
        ('bn4', lasagne.layers.BatchNormLayer),

        ('fc5', lasagne.layers.DenseLayer, {'num_units': 1024}),
        ('bn5', lasagne.layers.BatchNormLayer),

        ('hash', lasagne.layers.DenseLayer, {'num_units': hash_bit}),

        ('fc_out', lasagne.layers.DenseLayer, {'num_units': 10, 'nonlinearity': lasagne.nonlinearities.softmax}),
    ]

    net0 = nolearn.lasagne.NeuralNet(
        layers=network,
        max_epochs=200,

        update=adam,
        update_learning_rate=0.0002,

        objective_l2=0.0025,

        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
    )

    net0.fit(X, y)


if __name__ == "__main__":
    which_data = 'cifar10'
    nb_classes = 10

    # Load the dataset
    print("Loading data...")
    datasets = load_dataset(which_data)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set_x = np.subtract(np.multiply(2., train_set_x), 1.)
    valid_set_x = np.subtract(np.multiply(2., valid_set_x), 1.)
    test_set_x = np.subtract(np.multiply(2., test_set_x), 1.)

    # flatten targets
    train_set_y = np.hstack(train_set_y)
    valid_set_y = np.hstack(valid_set_y)
    test_set_y = np.hstack(test_set_y)

    # Onehot the targets
    train_set_y = np.float32(np.eye(nb_classes)[train_set_y])
    valid_set_y = np.float32(np.eye(nb_classes)[valid_set_y])
    test_set_y = np.float32(np.eye(nb_classes)[test_set_y])

    # for hinge loss
    train_set_y = 2 * train_set_y - 1.
    valid_set_y = 2 * valid_set_y - 1.
    test_set_y = 2 * test_set_y - 1.

    del datasets

    batch_size = 100
    num_epochs = 200
    hash_bits = np.array([48, 32, 24, 12])

    for i in range(len(hash_bits)):
        hash_bit = hash_bits[i]
        #  'You have {} things.'.format(things)  # str.format()
        #  'You have %d things.' % things  # % interpolation
        #  f'You have {things} things.'  # f-string (since Python 3.6)
        save_model_path = 'bdhn_' + which_data + '_model_' + str(hash_bit) + '.npz'
        save_hash_path = 'bdhn_' + which_data + '_hash_' + str(hash_bit) + '.npz'
        network = main(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, which_data,
                       batch_size, num_epochs, hash_bit,
                       save_model_path, save_hash_path)




