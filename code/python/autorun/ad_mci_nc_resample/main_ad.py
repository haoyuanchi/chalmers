from __future__ import print_function

import numpy as np
import get_data_ad

import theano.sandbox.cuda.nvcc_compiler

print(theano.sandbox.cuda.nvcc_compiler.is_nvcc_available())

import time
import os
import random
import argparse
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, Input
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
import keras

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import h5py
import scipy.io as sio


########################## setting param
np.random.seed(1337)  # for reproducibility

result_fold = '/home/hychi/data/medical/results/ADNI1_resample/'
save_fold = '/home/hychi/data/medical/results/ADNI1_resample/test_data/'

def load_dataset(datafold, test_part, random_state):
    # get x_set , y_set file
    X_files, y = get_data_ad.load_dataset_file(datafold)

    # random_state = random.randint(1, 1000)
    # random_state = 0
    X_train_files, X_test_files, y_train, y_test = train_test_split(X_files, y, test_size=test_part,
                                                                    random_state=random_state)

    X_train_files = np.asarray(X_train_files)
    X_test_files = np.asarray(X_test_files)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train_files, y_train, X_test_files, y_test


def build_3DCNN(size_i, size_j, size_k, channels, nb_filters_1, nb_filters_2, nb_filters_3, nb_filters_4,
                dropout_rate, pool_size, nb_classes, model_file):
    ########## build CNN

    input_shape = (channels, size_i, size_j, size_k)

    input_img = Input(shape=input_shape)

    conv1 = Convolution3D(nb_filters_1, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(input_img)
    pool1 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv1)
    conv2 = Convolution3D(nb_filters_2, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool1)
    pool2 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv2)
    conv3_1 = Convolution3D(nb_filters_3, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool2)
    pool3 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv3_1)
    conv3_2 = Convolution3D(nb_filters_4, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool3)
    encoded = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv3_2)

    dropout_1 = Dropout(p=dropout_rate)(encoded)
    flatten = Flatten()(dropout_1)
    dense_1 = Dense(output_dim=512, activation='relu')(flatten)
    dropout_2 = Dropout(p=dropout_rate)(dense_1)
    dense_2 = Dense(output_dim=512, activation='relu')(dropout_2)
    dropout_3 = Dropout(p=dropout_rate)(dense_2)
    output = Dense(output_dim=nb_classes, activation='softmax')(dropout_3)

    model = Model(input_img, output)

    # print the structure
    # plot(model, to_file=model_file, show_shapes=True, show_layer_names=True)

    # print model detals
    print(model.summary())

    return model


def fine_tune(model, model_weight_file):
    import h5py
    f = h5py.File(model_weight_file)
    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    # just get the 0~7 layers
    layer_names = layer_names[0:8]
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        layer = model.layers[k]
        symbolic_weights = layer.weights
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)


def myGenerator(inputs, targets, nb_classes, batch_size, MAX_VALUE, MIN_VALUE, MEAN_VALUE, shuffle,
                portion_i, portion_j, portion_k, is_rescale, is_mean):
    assert len(inputs) == len(targets)

    while True:
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            x_bitch, y_bitch = get_data_ad.preprocess(inputs[excerpt], targets[excerpt], nb_classes, 
                                                      portion_i=portion_i, portion_j=portion_j, portion_k=portion_k,
                                                      MIN_VALUE=MIN_VALUE, MAX_VALUE=MAX_VALUE, MEAN_VALUE=MEAN_VALUE,
                                                      is_scale=is_rescale, is_mean=is_mean)

            yield x_bitch, y_bitch


def ParserCommandLine():
    parser = argparse.ArgumentParser(description='train 3d cnn on alzheimer')
    default_image_dir = '/home/hychi/data/MedicalDataset/AD_MCI_NC_resample'
    parser.add_argument('-I', '--data_dir', default=default_image_dir,
                        help='location of image files; default=%s' % default_image_dir)
    parser.add_argument('-nc', '--nb_classes', type=int, default=2, help='number of class')
    parser.add_argument('-fs', '--filter_size', type=int, default=3, help='filter size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('-fn1', '--nb_filters_1', type=int, default=32, help='3D cnn filter number layer 1')
    parser.add_argument('-fn2', '--nb_filters_2', type=int, default=64, help='3D cnn filter number layer 2')
    parser.add_argument('-fn3', '--nb_filters_3', type=int, default=128, help='3D cnn filter number layer 3')
    parser.add_argument('-fn4', '--nb_filters_4', type=int, default=128, help='3D cnn filter number layer 4')
    parser.add_argument('-dp', '--dropout_rate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-ep', '--nb_epoch', type=int, default=400, help='number of epochs')
    parser.add_argument('-tp', '--test_part', type=float, default=0.3, help='split testing part')
    parser.add_argument('-rm', '--random_state', type=int, default=0, help='random state of split testing part')
    parser.add_argument('-id', '--interesting_class_id', type=int, default=-1, help='interesting class id')

    args = parser.parse_args()
    return args.data_dir, args.nb_classes, args.filter_size, args.learning_rate, \
           args.nb_filters_1, args.nb_filters_2, args.nb_filters_3, args.nb_filters_4, \
           args.dropout_rate, args.batch_size, args.nb_epoch, args.test_part, args.random_state, args.interesting_class_id


def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_true with class_id_preds for recall here
        accuracy_mask = K.equal(class_id_true, interesting_class_id)
        class_acc_tensor = K.equal(class_id_true, class_id_preds) * accuracy_mask
        class_acc = K.cast(K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1), dtype='float32')
        return class_acc
    return fn


def main():
    data_dir, nb_classes, filter_size, learning_rate, nb_filters_1, nb_filters_2, nb_filters_3, nb_filters_4, \
    dropout_rate, batch_size, nb_epoch, test_part, random_state, interesting_class_id = ParserCommandLine()

    # change the param to get different performance
    param_name = 'multi{0}_lr{1}_fn1_{2}_fn2_{3}_fn3_{4}_fn4_{5}_dp_{6}_bs_{7}_ep_{8}_tp_{9}_'.format(nb_classes,
                                                                                                      learning_rate,
                                                                                                      nb_filters_1,
                                                                                                      nb_filters_2,
                                                                                                      nb_filters_3,
                                                                                                      nb_filters_4,
                                                                                                      dropout_rate,
                                                                                                      batch_size,
                                                                                                      nb_epoch,
                                                                                                      test_part,
                                                                                                      random_state,
                                                                                                      interesting_class_id)
    model_file = result_fold + param_name + 'model.png'
    accuracy_figure = result_fold + param_name + 'accuracy.png'
    loss_figure = result_fold + param_name + 'loss.png'
    accuracy_file = result_fold + param_name + 'accuracy.txt'

    is_fine_tune = 0

    # cube patch
    portion_i = 1
    portion_j = 1
    portion_k = 1
    portion_num = portion_i * portion_j * portion_k

    channels = 1
    pool_size = (2, 2, 2)

    is_rescale = True
    is_mean = False

    X_train_files, y_train, X_test_files, y_test = load_dataset(data_dir, test_part, random_state)
    MAX_VALUE, MIN_VALUE, MEAN_VALUE = get_data_ad.cal_statistics(X_train_files)

    size_i, size_j, size_k = get_data_ad.cal_cube_size(X_test_files[0], portion_i, portion_j, portion_k)

    model = build_3DCNN(size_i, size_j, size_k, channels,
                        nb_filters_1, nb_filters_2, nb_filters_3, nb_filters_4,
                        dropout_rate, pool_size, nb_classes, model_file)

    sgd = keras.optimizers.SGD(lr=learning_rate, decay=5e-4, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adagrad = keras.optimizers.Adagrad(lr=learning_rate, epsilon=1e-8, decay=0.00001)

    if interesting_class_id == -1:
        model.compile(loss='categorical_crossentropy',
                      optimizer=adagrad,
                      metrics=['accuracy', 'recall', 'precision', 'fmeasure'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=adagrad,
                      metrics=['accuracy', single_class_accuracy(interesting_class_id)])

    if is_fine_tune:
        #########  load param for fine tuned
        load_param_name = 'cae_batch3_epoch100_layer4_filter32_adagrad'
        model_json_file = load_param_name + "model.json"
        model_weight_file = load_param_name + "model.h5"

    start_time = time.time()
    history = model.fit_generator(myGenerator(X_train_files, y_train, nb_classes, batch_size=batch_size, shuffle=True,
                                              portion_i=1, portion_j=1, portion_k=1,
                                              MAX_VALUE=MAX_VALUE, MIN_VALUE=MIN_VALUE, MEAN_VALUE=MEAN_VALUE,
                                              is_rescale=is_rescale, is_mean=is_mean),
                                  samples_per_epoch=len(X_train_files),
                                  nb_epoch=nb_epoch,
                                  validation_data=myGenerator(X_test_files, y_test, nb_classes, batch_size=batch_size, shuffle=True,
                                                              portion_i=1, portion_j=1, portion_k=1,
                                                              MAX_VALUE=MAX_VALUE, MIN_VALUE=MIN_VALUE, MEAN_VALUE=MEAN_VALUE,
                                                              is_rescale=is_rescale, is_mean=is_mean),
                                  nb_val_samples=len(X_test_files))
    train_time = time.time() - start_time
    print("Train time cost " + str(train_time / 60) + "min")

    X_test, Y_test = get_data_ad.preprocess(X_test_files, y_test, nb_classes,
                                            portion_i=portion_i, portion_j=portion_j, portion_k=portion_k,
                                            MIN_VALUE=MIN_VALUE, MAX_VALUE=MAX_VALUE, MEAN_VALUE=MEAN_VALUE,
                                            is_scale=is_rescale, is_mean=is_mean, is_shuffle=False)
    start_time = time.time()
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    test_time = time.time() - start_time
    print("Test time cost " + str(test_time) + "s")

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    ############################################### save the figure and results
    ###############################################
    # list all data in history
    print(history.history.keys())

    param_file = result_fold + param_name + '.txt'

    # save the param
    with file(param_file, 'a') as outfile:
        # outfile.writelines('model detail \n'.format(model.summary()))
        outfile.writelines('epoch number {0} \n'.format(nb_epoch))
        outfile.writelines('batch size {0} \n'.format(batch_size))
        outfile.writelines('optimizer method {0} \n'.format(str(model.optimizer)))
        outfile.writelines('model detail \n')
        summary = str(model.to_json())
        outfile.writelines('{0} \n'.format(summary))

    # print model details
    print(model.summary())

    # show the slices of correct image and incorrect image
    # save the test data and prediction results
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred_label = np.argmax(y_pred, axis=-1)
    equal = np.equal(y_pred_label, y_test)
    correct_index = np.nonzero(equal)

    get_data_ad.save_image(save_fold, X_test_files)

    prediction_results = result_fold + param_name + 'pred.mat'
    sio.savemat(prediction_results, {'pred': y_pred_label, 'true': y_test})

    # summarize history for accuracy
    acc_result_file = result_fold + param_name + 'result.mat'
    sio.savemat(acc_result_file, {'acc': history.history['acc'], 'val_acc': history.history['val_acc'],
                                  'loss': history.history['loss'], 'val_loss': history.history['val_loss']})


    # summarize history for accuracy
    fig_accuracy = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig_accuracy.savefig(accuracy_figure)

    # summarize history for loss
    fig_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('loss_origin.png')
    # plt.show()
    fig_loss.savefig(loss_figure)

    # the close() can be called for you automatically using the with statement
    with file(accuracy_file, 'a') as outfile:
        outfile.writelines('# train accuary {0} \n'.format(history.history['acc'][-1]))
        for item in history.history['acc']:
            outfile.write("{0}  ".format(item))
        outfile.writelines('\n')

        outfile.writelines('# val accuary {0} \n'.format(history.history['val_acc'][-1]))
        for item in history.history['val_acc']:
            outfile.write("{0}  ".format(item))
        outfile.writelines('\n')

        outfile.writelines('# test accuary {0} \n'.format(score[1]))

        outfile.writelines('# train loss {0} \n'.format(history.history['loss'][-1]))
        for item in history.history['loss']:
            outfile.write("{0}  ".format(item))
        outfile.writelines('\n')

        outfile.writelines('# val loss {0} \n'.format(history.history['val_loss'][-1]))
        for item in history.history['val_loss']:
            outfile.write("{0}  ".format(item))
        outfile.writelines('\n')

        outfile.writelines('# train time cost {0} \n'.format(train_time))
        outfile.writelines('# test time cost {0} \n'.format(test_time))

        # raw_input("Press Enter to continue")


if __name__ == '__main__':
    sys.exit(main())






