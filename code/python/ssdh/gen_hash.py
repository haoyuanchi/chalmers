
from load_data import load_svhn, load_mnist, load_cifar10, load_cifar100
import theano
import theano.tensor as T
import lasagne

import numpy as np

import scipy.io as sio

import h5py
from pandas import HDFStore
import pandas
import os


# os.environ['FUEL_DATA_PATH'] = 'E:/datasets';

def load_dataset(which_data='cifar10'):
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


def build_model(input_var, which_data):
    network = {}
    if which_data=='mnist':
        network['input'] = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                                     input_var=input_var)

    elif which_data=='svhn':
        network['input'] = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                                     input_var=input_var)

    elif which_data=='cifar10':
        network['input'] = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                                     input_var=input_var)

    elif which_data=='cifar100':
        network['input'] = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                                     input_var=input_var)

    network['conv1'] = lasagne.layers.Conv2DLayer(network['input'], num_filters=32, filter_size=(5, 5),
                                                  nonlinearity=lasagne.nonlinearities.rectify,
                                                  W=lasagne.init.GlorotUniform())
    network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))
    network['conv2'] = lasagne.layers.Conv2DLayer(network['pool1'], num_filters=32, filter_size=(5, 5),
                                                  nonlinearity=lasagne.nonlinearities.rectify,
                                                  W=lasagne.init.GlorotUniform())
    network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network['fc3'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['pool2'], p=0.5),
                                               num_units=256,
                                               nonlinearity=lasagne.nonlinearities.rectify,
                                               W=lasagne.init.GlorotUniform())

    # binary hash code
    network['fc4'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['fc3'], p=0.5),
                                               num_units=48,
                                               nonlinearity=lasagne.nonlinearities.sigmoid,
                                               W=lasagne.init.GlorotUniform())

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network['fc5'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['fc4'], p=0.5),
                                               num_units=10,
                                               nonlinearity=lasagne.nonlinearities.softmax)
    return network


if __name__ == "__main__":
    which_data = 'mnist'
    input_var = T.tensor4('inputs')
    network = build_model(input_var, which_data)

    with np.load(which_data + '.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network['fc5'], param_values[0])

    datasets = load_dataset(which_data)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set = np.vstack((train_set_x, valid_set_x))

    batch_size = 500

    # train dataset
    batches = len(train_set)/batch_size
    for i in range(batches):
        if i == 0:
            feature_out = np.array(lasagne.layers.get_output(network['fc4'],
                                                             train_set[i*batch_size: (i+1)*batch_size],
                                                             deterministic=True).eval())
        else:
            temp = np.array(lasagne.layers.get_output(network['fc4'],
                                                      train_set[i*batch_size: (i+1)*batch_size],
                                                      deterministic=True).eval())
            feature_out = np.vstack((feature_out, temp))

    with h5py.File('feature_train.h5', 'w') as hf:
        hf.create_dataset('feature_train', data=feature_out, compression="gzip", compression_opts=9)
        hf.close()

    # test dataset
    batches = len(test_set_x)/batch_size
    for i in range(batches):
        if i == 0:
            feature_out = np.array(lasagne.layers.get_output(network['fc4'],
                                                             test_set_x[i*batch_size: (i+1)*batch_size],
                                                             deterministic=True).eval())
        else:
            temp = np.array(lasagne.layers.get_output(network['fc4'],
                                                      test_set_x[i*batch_size: (i+1)*batch_size],
                                                      deterministic=True).eval())
            feature_out = np.vstack((feature_out, temp))

    with h5py.File('feature_test.h5', 'w') as hf:
        hf.create_dataset('feature_test', data=feature_out)
        hf.close()

    # feature_out = np.array(lasagne.layers.get_output(network['fc4'],
    #                                                  train_set,
    #                                                  deterministic=True).eval())
    # with h5py.File('feature_train.h5', 'w') as hf:
    #     hf.create_dataset('feature_train', data=feature_out, compression="gzip", compression_opts=9)
    #     hf.close()
    #
    # feature_out = np.array(lasagne.layers.get_output(network['fc4'],
    #                                                  test_set_x,
    #                                                  deterministic=True).eval())
    # with h5py.File('feature_test.h5', 'w') as hf:
    #     hf.create_dataset('feature_test', data=feature_out)
    #     hf.close()

        # temp = sio.loadmat('feature_train.mat')
        # feature_out = np.vstack(temp['feature_train'], feature_out)
        # sio.savemat('feature_train.mat', {'feature_train': feature_out})

    # hdf = pandas.read_hdf('storage.h5')

        # temp = sio.loadmat('feature_train.mat')
        # feature_out = np.vstack(temp['feature_train'], feature_out)
        # sio.savemat('feature_train.mat', {'feature_train': feature_out})

    # for i in range(train_set.shape[0]):
    #     feature_out = np.array(lasagne.layers.get_output(network['fc4'], train_set[i], deterministic=True).eval())
    #     sio.savemat('feature_train.mat', feature_out)

