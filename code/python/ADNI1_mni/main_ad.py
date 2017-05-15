from __future__ import print_function

import numpy as np
import get_data_ad

import theano.sandbox.cuda.nvcc_compiler

print(theano.sandbox.cuda.nvcc_compiler.is_nvcc_available())

import time
import os
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, Input

from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
import keras

# from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# save results to mat format
import scipy.io as sio


########################## setting param
np.random.seed(1337)  # for reproducibility

batch_size = 4
nb_classes = 3
nb_epoch = 1

# cube patch
portion_i = 1
portion_j = 1
portion_k = 1
portion_num = portion_i * portion_j * portion_k

channels = 1
nb_filters = 32
pool_size = (2, 2, 2)

lr_start = 0.002
dropout_rate = 0.5

# Zoom images uniformly from each side
zoom = False
trim = 200

is_rescale = True
is_mean = False

# change the param to get different performance
# param_name = 'layer4_kernel_5_fc1_512_epoch_100_'
result_fold = '/home/hychi/data/medical/results/ADNI1_resample/'
param_name = 'class_ad_nc_batch{0}_epoch{1}_layer{2}_filter{3}_{4}_lr_{5}_'.format(batch_size, nb_epoch, 4, nb_filters, 'adagrad', lr_start)
model_file = result_fold + param_name + 'model.png'
accuracy_figure = result_fold + param_name + 'accuracy.png'
loss_figure = result_fold + param_name + 'loss.png'
accuracy_file = result_fold + param_name + 'accuracy.txt'

# get x_set , y_set file
X_files, y = get_data_ad.load_dataset_file("/home/hychi/data/MedicalDataset/AD_MCI_NC_resample")

# random_state = random.randint(1, 1000)
random_state = 0
X_train_files, X_test_files, y_train, y_test = train_test_split(X_files, y, test_size=0.2, random_state=random_state)

X_train_files = np.asarray(X_train_files)
X_test_files = np.asarray(X_test_files)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

get_data_ad.show_center(X_train_files[0])

MAX_VALUE = 32767
MIN_VALUE = 0
MEAN_VALUE = 314.854

# MAX_VALUE, MIN_VALUE = get_data_ad.cal_statistics(X_train_files, image_size=cube_size)
# MAX_VALUE, MIN_VALUE, MEAN_VALUE = get_data_ad.cal_statistics(X_train_files)


def myGenerator(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    # targets = targets.astype('float32')

    while True:
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            x_bitch, y_bitch = get_data_ad.preprocess(nb_classes, inputs[excerpt], targets[excerpt],
                                                      portion_i=portion_i, portion_j=portion_j, portion_k=portion_k,
                                                      MIN_VALUE=MIN_VALUE, MAX_VALUE=MAX_VALUE, MEAN_VALUE=MEAN_VALUE,
                                                      is_scale=is_rescale, is_mean=is_mean)

            yield x_bitch, y_bitch


########## build CNN
size_i, size_j, size_k = get_data_ad.cal_cube_size(X_test_files[0], portion_i, portion_j, portion_k)

input_shape = (channels, size_i, size_j, size_k)

input_img = Input(shape=input_shape)

conv1 = Convolution3D(nb_filters, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(input_img)
pool1 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv1)
conv2 = Convolution3D(nb_filters * 2, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool1)
pool2 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv2)
conv3_1 = Convolution3D(nb_filters * 2 * 2, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool2)
pool3 = MaxPooling3D(pool_size=pool_size, border_mode='same')(conv3_1)
conv3_2 = Convolution3D(nb_filters * 2 * 2, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, activation='relu', border_mode='same')(pool3)
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

#########  load param
load_param_name = 'cae_batch3_epoch100_layer4_filter32_adagrad'
model_json_file = load_param_name + "model.json"
model_weight_file = load_param_name + "model.h5"

# json_file = open(model_json_file, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# from keras.models import model_from_json
# loaded_model = model_from_json(loaded_model_json)
#
# loaded_model.load_weights(model_weight_file)
#
# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
#
# # load weights into new model
# model.load_weights(model_weight_file)

# import h5py
# f = h5py.File(model_weight_file)
# # New file format.
# layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
# # just get the 0~7 layers
# layer_names = layer_names[0:8]
# weight_value_tuples = []
# for k, name in enumerate(layer_names):
#     g = f[name]
#     weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
#     weight_values = [g[weight_name] for weight_name in weight_names]
#     layer = model.layers[k]
#     symbolic_weights = layer.weights
#     weight_value_tuples += zip(symbolic_weights, weight_values)
# K.batch_set_value(weight_value_tuples)

# import h5py
# f = h5py.File(model_weight_file)
# for k in range(f.attrs['nb_layers']):
#     if k >= (len(model.layers) - 1):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     model.layers[k].set_weights(weights)
# f.close()
# print('Model loaded.')


sgd = keras.optimizers.SGD(lr=lr_start, decay=5e-4, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=lr_start, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adagrad = keras.optimizers.Adagrad(lr=lr_start, epsilon=1e-8, decay=0.00001)

optimizers = {}
optimizers['sgd'] = sgd
optimizers['adam'] = adam
optimizers['adagrad'] = adagrad


model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])

start_time = time.time()
history = model.fit_generator(myGenerator(X_train_files, y_train, batch_size=batch_size),
                              samples_per_epoch=len(X_train_files),
                              nb_epoch=nb_epoch,
                              validation_data=myGenerator(X_test_files, y_test, batch_size=batch_size),
                              nb_val_samples=len(X_test_files))
train_time = time.time() - start_time
print("Train time cost " + str(train_time / 60) + "min")
import scipy.io as sio
X_test, Y_test = get_data_ad.preprocess(nb_classes, X_test_files, y_test,
                                        portion_i=portion_i, portion_j=portion_j, portion_k=portion_k,
                                        MIN_VALUE=MIN_VALUE, MAX_VALUE=MAX_VALUE, MEAN_VALUE=MEAN_VALUE,
                                        is_scale=is_rescale, is_mean=is_mean, is_shuffle=False)
start_time = time.time()
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
test_time = time.time() - start_time
print("Test time cost " + str(test_time) + "s")

print('Test score:', score[0])
print('Test accuracy:', score[1])

# show the slices of correct image and incorrect image
# save the test data and prediction results
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred_label = np.argmax(y_pred, axis=-1)
equal = np.equal(y_pred_label, y_test)
correct_index = np.nonzero(equal)

save_fold = '/home/hychi/data/medical/results/ADNI1_resample/'
get_data_ad.save_image(save_fold, X_test_files)

prediction_results = result_fold + 'pred.mat'
sio.savemat(prediction_results, {'pred': y_pred_label, 'true': y_test})


############################################### save the figure and results
###############################################
# list all data in history
print(history.history.keys())

param_file = save_fold + param_name + '.txt'

# save the param
with file(param_file, 'a') as outfile:
    outfile.writelines('input shape {0}\n'.format(input_shape))
    # outfile.writelines('model detail \n'.format(model.summary()))
    outfile.writelines('epoch number {0} \n'.format(nb_epoch))
    outfile.writelines('batch size {0} \n'.format(batch_size))
    outfile.writelines('optimizer method {0} \n'.format(str(model.optimizer)))
    outfile.writelines('model detail \n')
    summary = str(model.to_json())
    outfile.writelines('{0} \n'.format(summary))


# print model detals
print(model.summary())

# summarize history for accuracy

acc_result_file = result_fold + 'result.mat'
sio.savemat(acc_result_file, {'acc': history.history['acc'], 'val_acc': history.history['val_acc'],
                      'loss': history.history['loss'], 'val_loss': history.history['val_loss']})

# fig_accuracy = plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# fig_accuracy.savefig(accuracy_figure)
#
# # summarize history for loss
# fig_loss = plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.savefig('loss_origin.png')
# # plt.show()
# fig_loss.savefig(loss_figure)

# the close() can be called for you automatically using the with statement
with file(accuracy_file, 'a') as outfile:
    outfile.writelines('# train accuary {0} \n'.format(history.history['acc'][-1]))
    for item in history.history['acc']:
        outfile.write("{0}  ".format(item))
    # np.savetxt(outfile, history.history['acc'], fmt='%-7.2f')
    outfile.writelines('\n')

    outfile.writelines('# val accuary {0} \n'.format(history.history['val_acc'][-1]))
    for item in history.history['val_acc']:
        outfile.write("{0}  ".format(item))
    outfile.writelines('\n')

    outfile.writelines('# test accuary {0} \n'.format(score[1]))

    outfile.writelines('# train time cost {0} \n'.format(train_time))
    outfile.writelines('# test time cost {0} \n'.format(test_time))

# raw_input("Press Enter to continue")