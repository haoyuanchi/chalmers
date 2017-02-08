import binary_net
from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import lasagne


activation = binary_net.binary_tanh_unit


def bulid_model_vgg_like(input_var, hash_bit, which_data):
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

    # 128C3-128C3-P2
    network['conv1_1'] = binary_net.Conv2DLayer(network['input'],
                                                num_filters=128,
                                                filter_size=(3, 3),
                                                pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    # network['bn1_1'] = lasagne.layers.BatchNormLayer(network['conv1_1'])
    network['ac1_1'] = lasagne.layers.NonlinearityLayer(network['conv1_1'], nonlinearity=activation)

    network['conv1_2'] = binary_net.Conv2DLayer(network['ac1_1'],
                                              num_filters=128,
                                              filter_size=(3, 3),
                                              pad=1,
                                              nonlinearity=lasagne.nonlinearities.identity)
    network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1_2'], pool_size=(2, 2))
    network['bn1_2'] = lasagne.layers.BatchNormLayer(network['pool1'])
    network['ac1_2'] = lasagne.layers.NonlinearityLayer(network['bn1_2'], nonlinearity=activation)

    # 256C3-256C3-P2
    network['conv2_1'] = binary_net.Conv2DLayer(network['ac1_2'],
                                                num_filters=256,
                                                filter_size=(3, 3),
                                                pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    # network['bn2_1'] = lasagne.layers.BatchNormLayer(network['conv2_1'])
    network['ac2_1'] = lasagne.layers.NonlinearityLayer(network['conv2_1'], nonlinearity=activation)

    network['conv2_2'] = binary_net.Conv2DLayer(network['ac2_1'],
                                                num_filters=256,
                                                filter_size=(3, 3),
                                                pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2_2'], pool_size=(2, 2))
    network['bn2_2'] = lasagne.layers.BatchNormLayer(network['pool2'])
    network['ac2_2'] = lasagne.layers.NonlinearityLayer(network['bn2_2'], nonlinearity=activation)

    # 512C3-512C3-P2
    network['conv3_1'] = binary_net.Conv2DLayer(network['ac2_2'],
                                                num_filters=512,
                                                filter_size=(3, 3),
                                                pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    # network['bn3_1'] = lasagne.layers.BatchNormLayer(network['conv3_1'])
    network['ac3_1'] = lasagne.layers.NonlinearityLayer(network['conv3_1'], nonlinearity=activation)

    network['conv3_2'] = binary_net.Conv2DLayer(network['ac3_1'],
                                                num_filters=512,
                                                filter_size=(3, 3),
                                                pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3_2'], pool_size=(2, 2))
    network['bn3_2'] = lasagne.layers.BatchNormLayer(network['pool3'])
    network['ac3_2'] = lasagne.layers.NonlinearityLayer(network['bn3_2'], nonlinearity=activation)

    # 1024FP-1024FP-10FP
    network['fc4'] = binary_net.DenseLayer(network['ac3_2'], num_units=1024, nonlinearity=lasagne.nonlinearities.identity)
    network['bn4'] = lasagne.layers.BatchNormLayer(network['fc4'])
    network['ac4'] = lasagne.layers.NonlinearityLayer(network['bn4'], nonlinearity=activation)

    network['fc5'] = binary_net.DenseLayer(network['ac4'], num_units=1024, nonlinearity=lasagne.nonlinearities.identity)
    network['bn5'] = lasagne.layers.BatchNormLayer(network['fc5'])
    network['ac5'] = lasagne.layers.NonlinearityLayer(network['bn5'], nonlinearity=activation)

    network['fc_hash'] = binary_net.DenseLayer(network['ac5'], num_units=hash_bit, nonlinearity=lasagne.nonlinearities.identity)
    network['bn_hash'] = lasagne.layers.BatchNormLayer(network['fc_hash'])
    network['hash_out'] = lasagne.layers.NonlinearityLayer(network['bn_hash'], nonlinearity=activation)

    network['fc_out'] = binary_net.DenseLayer(network['hash_out'], num_units=11, nonlinearity=lasagne.nonlinearities.identity)
    network['bn_out'] = lasagne.layers.BatchNormLayer(network['fc_out'])
    network['prob'] = lasagne.layers.NonlinearityLayer(network['bn_out'], nonlinearity=lasagne.nonlinearities.softmax)

    return network




def build_model_alexnet(input_var, hash_bit, which_data):
    net = {}

    net['input'] = InputLayer((None, 3, 227, 227))

    # conv1
    net['conv1'] = binary_net.Conv2DLayer(
        net['data'],
        num_filters=96,
        filter_size=(11, 11),
        stride=4,
        nonlinearity=lasagne.nonlinearities.identity)

    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    net['bn1'] = lasagne.layers.BatchNormLayer(net['conv1'])
    net['ac1'] = lasagne.layers.NonlinearityLayer(net['bn1'], nonlinearity=activation)

    # conv2
    net['conv2_data1'] = SliceLayer(net['ac1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['ac1'], indices=slice(48, 96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = binary_net.Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2,
                                                nonlinearity=lasagne.nonlinearities.identity)
    net['conv2_part2'] = binary_net.Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2,
                                                nonlinearity=lasagne.nonlinearities.identity)

    # now combine
    net['conv2'] = concat((net['conv2_part1'], net['conv2_part2']), axis=1)

    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2)

    # norm2
    net['bn2'] = lasagne.layers.BatchNormLayer(net['pool2'])
    net['ac2'] = lasagne.layers.NonlinearityLayer(net['bn2'], nonlinearity=activation)

    # conv3
    # no group
    net['conv3'] = binary_net.Conv2DLayer(net['ac2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad=1,
                                          nonlinearity=lasagne.nonlinearities.identity)
    net['ac3'] = lasagne.layers.NonlinearityLayer(net['conv3'], nonlinearity=activation)


    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['ac3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['ac3'], indices=slice(192, 384), axis=1)
    net['conv4_part1'] = binary_net.Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    net['conv4_part2'] = binary_net.Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    net['conv4'] = concat((net['conv4_part1'], net['conv4_part2']), axis=1)
    net['ac4'] = lasagne.layers.NonlinearityLayer(net['conv4'], nonlinearity=activation)


    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['ac4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['ac4'], indices=slice(192, 384), axis=1)
    net['conv5_part1'] = binary_net.Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    net['conv5_part2'] = binary_net.Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1,
                                                nonlinearity=lasagne.nonlinearities.identity)
    net['conv5'] = concat((net['conv5_part1'], net['conv5_part2']), axis=1)

    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride=2)

    net['ac5'] = lasagne.layers.NonlinearityLayer(net['pool5'], nonlinearity=activation)

    # fc6
    net['fc6'] = binary_net.DenseLayer(
        net['ac5'], num_units=4096,
        nonlinearity=lasagne.nonlinearities.identity)
    net['bn6'] = lasagne.layers.BatchNormLayer(net['fc6'])

    net['ac6'] = lasagne.layers.NonlinearityLayer(net['bn6'], nonlinearity=activation)

    # fc7
    net['fc7'] = binary_net.DenseLayer(
        net['ac6'],
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.identity)
    net['bn7'] = lasagne.layers.BatchNormLayer(net['fc7'])
    net['ac7'] = lasagne.layers.NonlinearityLayer(net['bn7'], nonlinearity=activation)

    net['hash'] = binary_net.DenseLayer(
        net['ac7'],
        num_units=hash_bit,
        nonlinearity=lasagne.nonlinearities.identity)
    net['bn8'] = lasagne.layers.BatchNormLayer(net['hash'])
    net['ac8'] = lasagne.layers.NonlinearityLayer(net['bn8'], nonlinearity=activation)

    # fc8
    net['fc8'] = binary_net.DenseLayer(
        net['ac8'],
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net