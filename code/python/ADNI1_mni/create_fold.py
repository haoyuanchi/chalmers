__author__ = 'hychi'
import os
import shutil

base_img_data_path = '/home/hychi/data/MedicalDataset/AD_MCI_NC_mni/'

count_num = 0
for type_fold in sorted(os.listdir(base_img_data_path)):
    type_fold_path = os.path.join(base_img_data_path, type_fold)
    for subject_fold in sorted(os.listdir(type_fold_path)):
        subject_fold_path = os.path.join(type_fold_path, subject_fold)
        anat_fold_path = os.path.join(subject_fold_path, 'anat')
        if not os.path.exists(anat_fold_path):
            os.mkdir(anat_fold_path)
        for file in os.listdir(subject_fold_path):
            if file.endswith('.nii'):
                file_path = os.path.join(subject_fold_path, file)
                destination_path = os.path.join(anat_fold_path, 'mprage.nii')
                shutil.copy2(file_path, destination_path)
                count_num += 1
                print('copy the ' + str(count_num) + ': ' + file)


