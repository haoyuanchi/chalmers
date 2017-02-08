__author__ = 'hychi'


from load_data import load_svhn, load_mnist, load_cifar10, load_cifar100
import theano
import theano.tensor as T
import lasagne
from keras.utils import np_utils

import binary_net
import model

import image_preprocess

import numpy as np
from collections import OrderedDict

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

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    LR_start = 0.01
    LR_fin = 0.0000003
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)

    network = model.bulid_model_vgg_like(input_var, hash_bit=hash_bit, which_data=which_data)

    #############  train
    output_layer = network['prob']
    hash_layer = network['hash_out']
    hash_out = lasagne.layers.get_output(hash_layer)

    prediction = lasagne.layers.get_output(output_layer, deterministic=False)

    ##########    lose function cross entropy
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = loss.mean()
    ##########    lose function hinge
    loss = T.mean(T.sqr(T.maximum(0., 1. - prediction * target_var)))

    # # making each bit has 50% probability of being one or zero
    # # sum the batch size
    avg = theano.tensor.sum(hash_out, 0) / batch_size
    even_dist_loss = lasagne.regularization.l2(avg)
    # even_dist_loss = lasagne.regularization.l1(avg)
    even_dist_loss_weight = 0.2

    loss = loss + even_dist_loss * even_dist_loss_weight

    ############ update
    # W updates
    W = lasagne.layers.get_all_params(output_layer, binary=True)
    W_grads = binary_net.compute_grads(loss, output_layer)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates, output_layer)

    # other parameters updates
    params = lasagne.layers.get_all_params(output_layer, trainable=True, binary=False)
    updates = OrderedDict(
        updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    ############# test
    ##########    lose function cross entropy
    # test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    # test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    #                                                         target_var)
    # test_loss = test_loss.mean()
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    ##########    lose function hinge
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - test_prediction * target_var)))
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var, LR], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")

    best_val_acc = 0
    best_epoch = 1
    LR = LR_start

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X, y, LR):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        batches = len(X) / batch_size
        for i in range(batches):
            train_err += train_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size], LR)

        train_err /= batches
        return train_err

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):
        # And a full pass over the validation data:
        val_loss = 0
        val_acc = 0
        batches = len(X) / batch_size
        for i in range(batches):
            loss, acc = val_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
            val_loss += loss
            val_acc += acc

        val_acc = val_acc / batches * 100
        val_loss /= batches

        return val_loss, val_acc

    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()

        train_set_x, train_set_y = shuffle(train_set_x, train_set_y)
        train_err = train_epoch(train_set_x, train_set_y, LR)
        val_loss, val_acc = val_epoch(valid_set_x, valid_set_y)

        # test if validation error went down
        if val_acc >= best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch + 1

            test_loss, test_acc = val_epoch(test_set_x, test_set_y)

            if save_model_path is not None:
                np.savez(save_model_path, *lasagne.layers.get_all_param_values(output_layer))
            if save_hash_path is not None:
                np.savez(save_hash_path, *lasagne.layers.get_all_param_values(hash_layer))

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch " + str(epoch + 1) + " of " + str(num_epochs) + " took " + str(epoch_duration) + "s")
        print("  LR:                            " + str(LR))
        print("  training loss:                 " + str(train_err))
        print("  validation loss:               " + str(val_loss))
        print("  validation accuracy rate:      " + str(val_acc) + "%")
        print("  best epoch:                    " + str(best_epoch))
        print("  best validation accuracy rate: " + str(best_val_acc) + "%")
        print("  test loss:                     " + str(test_loss))
        print("  test accuracy rate:            " + str(test_acc) + "%")

        # decay the LR
        LR *= LR_decay

    return network


if __name__ == "__main__":
    which_data = 'svhn'
    nb_classes = 11

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
        save_model_path = 'bdhn_' + which_data + '_cifar10_model_' + str(hash_bit) + '.npz'
        save_hash_path = 'bdhn_' + which_data + '_hash_' + str(hash_bit) + '.npz'
        network = main(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, which_data,
                       batch_size, num_epochs, hash_bit,
                       save_model_path, save_hash_path)




