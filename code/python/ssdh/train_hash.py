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


feature_width = 256
feature_height = 256
batch_size = 128
nb_classes = 2
nb_epoch = 12
hash_bits = 48

def cropImage(im):
    im2 = np.dstack(im).astype(np.uint8)
    # return centered 128x128 from original 250x250 (40% of area)
    # newim = im2[61:189, 61:189]
    newim = im2
    sized1 = imresize(newim[:, :, 0:3], (feature_width, feature_height), interp="bicubic", mode="RGB")
    # sized2 = imresize(newim[:, :, 3:6], (feature_width, feature_height), interp="bicubic", mode="RGB")
    return np.asarray([sized1[:, :, 0], sized1[:, :, 1], sized1[:, :, 2]]) #, sized2[:, :, 0], sized2[:, :, 1], sized2[:, :, 2]])


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
        network['input'] = lasagne.layers.InputLayer(shape=(batch_size, 1, 28, 28),
                                                     input_var=input_var)

    elif which_data=='svhn':
        network['input'] = lasagne.layers.InputLayer(shape=(batch_size, 3, 32, 32),
                                                     input_var=input_var)

    elif which_data=='cifar10':
        network['input'] = lasagne.layers.InputLayer(shape=(batch_size, 3, 32, 32),
                                                     input_var=input_var)

    elif which_data=='cifar100':
        network['input'] = lasagne.layers.InputLayer(shape=(batch_size, 3, 32, 32),
                                                     input_var=input_var)
    network['conv1'] = lasagne.layers.Conv2DLayer(
        network['input'], num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))

    network['conv2'] = lasagne.layers.Conv2DLayer(
        network['pool1'], num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network['fc3'] = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network['pool2'], p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # binary hash code
    network['hash'] = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network['fc3'], p=0.5),
        num_units=48,
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform())

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network['fc4'] = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network['hash'], p=0.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


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


def main(which_data='cifar10', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    datasets = load_dataset(which_data)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # crop features and resize the image
    # X_train = np.asarray(map(cropImage, train_set_x))
    # X_valid = np.asarray(map(cropImage, valid_set_x))
    # X_test = np.asarray(map(cropImage, test_set_x))

    # show images
    # import matplotlib.pyplot as plt
    # # Plot an example digit with its label
    # img = train_set_x[0].transpose(1, 2, 0)  # image shape 32 * 32 * 3
    # pixels = img.reshape((28, 28))
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.imshow(pixels, cmap=plt.cm.Greys)
    # # ax1.imshow(img)
    # # ax1.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # # ax2 = fig.add_subplot(1, 2, 2)
    # # ax2.imshow(img[0], cmap=plt.cm.gray_r)
    # plt.title("Label: {}".format(train_set_y[0]))
    # plt.gca().set_axis_off()
    # plt.show()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    seedval=12345
    rng = np.random.RandomState(seedval)

    network = build_model(input_var, which_data)

    prediction = lasagne.layers.get_output(network['fc4'])

    hash_out = lasagne.layers.get_output(network['hash'])
    hash_out_shape = lasagne.layers.get_output_shape(network['hash'])
    assert(hash_out_shape == (batch_size, hash_bits))

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # lasagne.regularization.l2(network['fc4'])

    l2_penalty = regularize_layer_params(network['fc4'], l2)
    l1_penalty = regularize_layer_params(network['fc4'], l1) * 1e-4

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

    params = lasagne.layers.get_all_params(network['fc4'], trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network['fc4'], deterministic=True)
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
        for batch in iterate_minibatches(train_set_x, train_set_y, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(valid_set_x, valid_set_y, batch_size, shuffle=False):
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
    for batch in iterate_minibatches(test_set_x, test_set_y, batch_size, shuffle=False):
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
    which_data = 'cifar10'
    network = main(which_data)

    # dump the network weights to a file
    np.savez(which_data + '.npz', lasagne.layers.get_all_param_values(network['fc4']))




