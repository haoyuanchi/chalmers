__author__ = 'hychi'

import os
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import nibabel as nib

# param set
result_fold = '/home/hychi/data/Chalmsers/20170413/experiments/results/'
save_fold = '/home/hychi/data/Chalmsers/20170413/experiments/test_data/'

#  load the test file
img_file_list = os.listdir(save_fold)

x_set = []
for i in xrange(0, len(img_file_list)):
    x_set.append(save_fold + '/' + img_file_list[i])

#  load the prediction and true label
mat_dict = sio.loadmat(result_fold + 'ad_nc_lr0.002_fn1_32_fn2_64_fn3_128_fn4_128_dp_0.5_bs_4_ep_200_tp_0.3_rm_10_pred.mat')

y_pred = mat_dict['pred'][0]
y_true = mat_dict['true'][0]

equal = np.equal(y_pred, y_true)
correct_index = np.nonzero(equal)

show_number = 10
# AD correct
fig_ac, axes_ac = plt.subplots(1, show_number)
# NC correct
fig_nc, axes_nc = plt.subplots(1, show_number)
# AD incorrect
fig_ai, axes_ai = plt.subplots(1, show_number)
# NC incorrect
fig_ni, axes_ni = plt.subplots(1, show_number)


#  draw the slices of AD & NC
def show_slices(img_file, axes, index):
    if isinstance(img_file, str):
        img = nib.load(img_file)
        img_data = img.get_data()

    # show the center slices for MRI image
    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) / 2
    center_j = (n_j - 1) / 2
    center_k = (n_k - 1) / 2

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    axes[index].imshow(slice_2.T, cmap="gray", origin="lower")
    axes[index].axis('off')


AD_correct_number = 0
AD_incorrect_number = 0
NC_correct_number = 0
NC_incorrect_number = 0

for i in xrange(0, len(y_pred)):
    if y_true[i] == 0:
        if y_true[i] == y_pred[i]:
            if AD_correct_number >= show_number:
                continue
            else:
                show_slices(x_set[i], axes_ac, AD_correct_number)
                AD_correct_number += 1
        else:
            if AD_incorrect_number >= show_number:
                continue
            else:
                show_slices(x_set[i], axes_ai, AD_incorrect_number)
                AD_incorrect_number += 1
    else:
        if y_true[i] == y_pred[i]:
            if NC_correct_number >= show_number:
                continue
            else:
                show_slices(x_set[i], axes_nc, NC_correct_number)
                NC_correct_number += 1
        else:
            if NC_incorrect_number >= show_number:
                continue
            else:
                show_slices(x_set[i], axes_ni, NC_incorrect_number)
                NC_incorrect_number += 1

fig_ac.savefig('AD_correct.png')
fig_nc.savefig('NC_correct.png')
fig_ai.savefig('AD_incorrect.png')
fig_ni.savefig('NC_incorrect.png')





