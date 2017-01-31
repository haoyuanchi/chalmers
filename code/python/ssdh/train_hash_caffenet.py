__author__ = 'hychi'


from load_data import load_svhn, load_mnist, load_cifar10, load_cifar100
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

import numpy as np

import time

from scipy.misc import imresize
import matplotlib.pyplot as plt


feature_width = 227
feature_height = 227
batch_size = 128
num_epochs = 200
hash_bits = 48

which_data = 'cifar10'

def cropImage(im):
    im2 = np.dstack(im).astype(np.float32)
    # return centered 128x128 from original 250x250 (40% of area)
    # newim = im2[61:189, 61:189]
    newim = im2
    if which_data=='mnist':
        img_return = imresize(newim, (feature_width, feature_height), interp="bicubic")
    else:
        sized1 = imresize(newim[:, :, 0:3], (feature_width, feature_height), interp="bicubic", mode="RGB")
        img_return = np.asarray([sized1[:, :, 0], sized1[:, :, 1], sized1[:, :, 2]])

    return img_return


def plot_img(X_set, y_set, axis):
    # Plot an example digit with its label
    img = X_set[0].transpose(1, 2, 0)  # image shape 32 * 32 * 3
    axis.imshow(img)
    # axis.title("Label: {}".format(y_set[0]))
    # axis.gca().set_axis_off()


def normalization(X):
    X -= X.mean()
    X /= X.std()
    return X


def load_dataset():
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


def build_model(input_var):
    net = {}

    if which_data == 'mnist':
        net['input'] = lasagne.layers.InputLayer(shape=(None, 1, feature_width, feature_height),
                                                 input_var=input_var)
    else:
        net['input'] = lasagne.layers.InputLayer(shape=(None, 3, feature_width, feature_height),
                                                 input_var=input_var)

    # conv1
    net['conv1'] = lasagne.layers.Conv2DLayer(
        net['input'],
        num_filters=96,
        filter_size=(11, 11),
        stride=4,
        nonlinearity=lasagne.nonlinearities.rectify)

    # pool1
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(
        net['pool1'],
        n=5,
        alpha=0.0001 / 5.0,
        beta=0.75,
        k=1)

    # conv2
    # before conv2 split the data
    net['conv2_data1'] = lasagne.layers.SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = lasagne.layers.SliceLayer(net['norm1'], indices=slice(48, 96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = lasagne.layers.Conv2DLayer(
        net['conv2_data1'],
        num_filters=128,
        filter_size=(5, 5),
        pad=2)
    net['conv2_part2'] = lasagne.layers.Conv2DLayer(
        net['conv2_data2'],
        num_filters=128,
        filter_size=(5, 5),
        pad=2)

    # now combine
    net['conv2'] = lasagne.layers.concat((net['conv2_part1'], net['conv2_part2']), axis=1)

    # pool2
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2)

    # norm2
    net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(
        net['pool2'],
        n=5,
        alpha=0.0001 / 5.0,
        beta=0.75,
        k=1)

    # conv3
    # no group
    net['conv3'] = lasagne.layers.Conv2DLayer(
        net['norm2'],
        num_filters=384,
        filter_size=(3, 3),
        pad=1)

    # conv4
    # group = 2
    net['conv4_data1'] = lasagne.layers.SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = lasagne.layers.SliceLayer(net['conv3'], indices=slice(192, 384), axis=1)
    net['conv4_part1'] = lasagne.layers.Conv2DLayer(
        net['conv4_data1'],
        num_filters=192,
        filter_size=(3, 3),
        pad=1)
    net['conv4_part2'] = lasagne.layers.Conv2DLayer(
        net['conv4_data2'],
        num_filters=192,
        filter_size=(3, 3),
        pad=1)
    net['conv4'] = lasagne.layers.concat((net['conv4_part1'], net['conv4_part2']), axis=1)

    # conv5
    # group 2
    net['conv5_data1'] = lasagne.layers.SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = lasagne.layers.SliceLayer(net['conv4'], indices=slice(192, 384), axis=1)
    net['conv5_part1'] = lasagne.layers.Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5_part2'] = lasagne.layers.Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5'] = lasagne.layers.concat((net['conv5_part1'], net['conv5_part2']), axis=1)

    # pool 5
    net['pool5'] = lasagne.layers.MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride=2)

    # fc6
    net['fc6'] = lasagne.layers.DenseLayer(
        net['pool5'],
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fc7
    net['fc7'] = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(net['fc6'], p=0.5),
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # binary hash code
    net['hash'] = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(net['fc7'], p=0.5),
        num_units=hash_bits,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    # fc8
    net['fc8'] = lasagne.layers.DenseLayer(
        net['hash'],
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net


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


def main(X_train, train_set_y, X_valid, valid_set_y, X_test, test_set_y):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    seedval=12345
    rng = np.random.RandomState(seedval)

    network = build_model(input_var)

    output_layer = network['fc8']
    hash_layer = network['hash']

    import pickle
    with open('C:\Users\chiha\github\Lasagne\Recipes\pretrained\imagenet\caffe_reference.pkl', 'rb') as f:
        params = pickle.load(f)
    lasagne.layers.set_all_param_values(network['fc7'], params[0: -2])

    prediction = lasagne.layers.get_output(output_layer)

    hash_out = lasagne.layers.get_output(hash_layer)
    hash_out_shape = lasagne.layers.get_output_shape(hash_layer)
    # assert(hash_out_shape == (batch_size, hash_bits))

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # lasagne.regularization.l2(network['fc4'])

    l2_penalty = regularize_layer_params(output_layer, l2)
    l1_penalty = regularize_layer_params(output_layer, l1) * 1e-4

    # making each bit has 50% probability of being one or zero
    # sum the batch size
    avg = theano.tensor.sum(hash_out, 0) / batch_size
    even_dist_loss = l2(avg - 0.5)
    # x = theano.tensor.ivector('x')
    # y = theano.tensor.iscalar('y')
    # y = l2(x - 0.5)
    # f = theano.function([x], y)
    # even_dist_loss = f(avg)
    even_dist_loss_weight = 0.1

    # encourage the activations of the units in H to be close to either 0 or 1
    # x = theano.tensor.imatrix('x')
    # y = theano.tensor.iscalar('y')
    # y = T.mean(T.sum((x - 0.5) * (x - 0.5), 1))     # every hash bit - 0.5
    # f = theano.function([x], y)
    # force_binary_loss = f(hash_out)
    force_binary_loss = T.mean(T.sum((hash_out - 0.5) * (hash_out - 0.5), 0))
    force_binary_loss_weight = 0.1

    # l2_penalty + l1_penalty + \
    loss = loss + \
           even_dist_loss * even_dist_loss_weight + \
           force_binary_loss * force_binary_loss_weight

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, train_set_y, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_valid, valid_set_y, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, test_set_y, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    return network

if __name__ == "__main__":
    # Load the dataset
    print("Loading data...")
    datasets = load_dataset()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # crop features and resize the image
    X_train = np.asarray(map(cropImage, train_set_x))
    X_valid = np.asarray(map(cropImage, valid_set_x))
    X_test = np.asarray(map(cropImage, test_set_x))

    # apply some very simple normalization to the data
    X_train = normalization(X_train)
    X_valid = normalization(X_valid)
    X_test = normalization(X_test)

    # plt.ion()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # plot_img(valid_set_x, valid_set_x, ax1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # plot_img(X_valid, valid_set_x, ax2)
    # plt.show()
    # plt.ioff()

    network = main(X_train, train_set_y, X_valid, valid_set_y, X_test, test_set_y)

    output_layer = network['fc8']

    # dump the network weights to a file
    np.savez(which_data + '.npz', lasagne.layers.get_all_param_values(output_layer))




