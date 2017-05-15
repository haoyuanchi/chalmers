from __future__ import print_function

import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img
import nilearn
from nilearn import image
from nilearn import plotting
# from nibabel import show_slices

from keras.utils import np_utils

import matplotlib.pyplot as plt


AFFINE_CONST = 4

import shutil

def load_dataset_file(image_folder):
    folderlist = os.listdir(image_folder)
    # Sort List
    folderlist = sorted(folderlist)

    # Count number of files in folder
    nbrOfImages = 0
    for i in xrange(0, len(folderlist)):
        nbrOfImages += len(os.listdir(image_folder + "/" + folderlist[i]))
    print("Nbr of images: ", nbrOfImages)

    x_set = []
    y_set = []

    for i in xrange(0, len(folderlist)):
        # List of strings with names of images in subfolder
        image_list = os.listdir(image_folder + "/" + folderlist[i])
        for j in xrange(0, len(image_list)):
            x_set.append(image_folder + '/' + folderlist[i] + '/' + image_list[j])
            y_set.append(i)

    return x_set, y_set


def save_image(img_fold, img_files):
    for i in xrange(0, len(img_files)):
        # get file name
        img_name = img_files[i].split("/")[-1]
        img_file_new = img_fold + str(i) + '_' + img_name
        shutil.copy2(img_files[i], img_file_new)


def cal_statistics(img_files):
    MAX_VALUE = 0
    MIN_VALUE = 100000
    MEAN_VALUE = 0

    # load the data
    for i in xrange(0, len(img_files)):
        # Load image from subfolder
        epi_img = nib.load(img_files[i])
        epi_img_data = epi_img.get_data()

        tmp = np.max(epi_img_data)
        if tmp > MAX_VALUE:
            MAX_VALUE = tmp

        tmp_min = np.min(epi_img_data)
        if MIN_VALUE > tmp_min:
            MIN_VALUE = tmp_min

        tmp = np.mean(epi_img_data)
        MEAN_VALUE += tmp

        # print(epi_img_data.shape)

    MEAN_VALUE = MEAN_VALUE / len(img_files)
    return MAX_VALUE, MIN_VALUE, MEAN_VALUE


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def show_center(img_data):
    if isinstance(img_data, str):
        img = nib.load(img_data)
        img_data = img.get_data()

    # show the center slices for MRI image
    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) / 2
    center_j = (n_j - 1) / 2
    center_k = (n_k - 1) / 2

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for ADNI image")



def test(img_files, image_size, depth):
    nbrOfImages = len(img_files)
    for i in xrange(0, len(img_files)):
        # Load image from subfolder
        epi_img = nib.load(img_files[i])

        epi_img_data = epi_img.get_data()
        epi_img_data = epi_img_data[:, :, :, 0]

        # affine
        print(epi_img.affine)
        epi_vox_center = (np.array(epi_img_data.shape) - 1) / 2.
        print(epi_img.affine.dot(list(epi_vox_center) + [1]))


def cal_cube_size(img_file, portion_i, portion_j, portion_k):
    epi_img = nib.load(img_file)

    epi_img_data = epi_img.get_data()
    # epi_img_data = epi_img_data[:, :, :, 0]
    num_i, num_j, num_k = epi_img_data.shape

    size_i = num_i / portion_i
    size_j = num_j / portion_j
    size_k = num_k / portion_k

    return size_i, size_j, size_k


def preprocess(img_files, target, portion_i, portion_j, portion_k, MIN_VALUE, MAX_VALUE, MEAN_VALUE,
               is_scale=True, is_mean=True, is_shuffle=True):
    nbrOfImages = len(img_files)

    portion_num = portion_i * portion_j * portion_k

    size_i, size_j, size_k = cal_cube_size(img_files[0], portion_i, portion_j, portion_k)

    X_set = np.zeros(shape=(nbrOfImages * portion_num, 1, size_i, size_j, size_k), dtype=np.float32)
    y_set = np.zeros(shape=(nbrOfImages * portion_num))

    for i in xrange(0, len(img_files)):
        # Load image from subfolder
        epi_img = nib.load(img_files[i])

        epi_img_data = epi_img.get_data()
        # epi_img_data = epi_img_data[:, :, :, 0]

        tmp = np.max(epi_img_data)
        if tmp > MAX_VALUE:
            MAX_VALUE = tmp

        tmp_min = np.min(epi_img_data)
        if MIN_VALUE > tmp_min:
            MIN_VALUE = tmp_min

        # split the data from the slice
        index_temp = 0
        for index_i in xrange(0, portion_i):
            for index_j in xrange(0, portion_j):
                for index_k in xrange(0, portion_k):
                    # Store image in array
                    data_index = i * portion_num + index_temp
                    data = epi_img_data[(index_i * size_i):((index_i+1) * size_i),
                           (index_j * size_j):((index_j + 1) * size_j),
                           (index_k * size_k):((index_k + 1) * size_k)]
                    if is_scale:
                        data = (data - data.min()) / data.max()
                    X_set[data_index, 0, :, :, :] = data
                    y_set[data_index] = target[i]
                    # show_center(data)
                    index_temp += 1
    # if is_mean:
    #     X_set = X_set - MEAN_VALUE
    # if is_scale:
    #     X_set = (X_set - X_set.min()) / X_set.max()
    #     X_set = (X_set - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

    Y_set = np_utils.to_categorical(y_set, 2)

    if is_shuffle:
        # Shuffle images and labels
        indices = np.arange(len(y_set))
        np.random.shuffle(indices)
        excerpt = indices[0:len(y_set)]
    else:
        excerpt = np.arange(len(y_set))

    return X_set[excerpt], Y_set[excerpt]