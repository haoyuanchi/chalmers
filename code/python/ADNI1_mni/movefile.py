import shutil
import os

import xml.etree.ElementTree as ET


base_img_data_path = '/home/hychi/data/ADNI/ADNI1_Screening/data/'
base_img_config_path = '/home/hychi/data/ADNI/ADNI1_Screening/config/'

dest_img_path = '/home/hychi/data/MedicalDataset/ADNI1_new/'

lmci_list = []
normal_list = []
ad_list = []
emci_list = []
smc_list = []
mci_list = []


# analysis the xml
for file in os.listdir(base_img_config_path):
    file_path = os.path.join(base_img_config_path, file)
    tree = ET.parse(file_path)
    # for elem in tree.iterfind('branch/sub-branch'):
    group_type_node = tree.find('.//subjectInfo[@item="DX Group"]')
    subject_id_node = tree.find('.//subjectIdentifier')
    # print group_type_node.text
    if group_type_node.text == 'LMCI':
        lmci_list.append(subject_id_node.text)
    elif group_type_node.text == 'EMCI':
        emci_list.append(subject_id_node.text)
    elif group_type_node.text == 'Normal':
        normal_list.append(subject_id_node.text)
    elif group_type_node.text == 'AD':
        ad_list.append(subject_id_node.text)
    elif group_type_node.text == 'SMC':
        smc_list.append(subject_id_node.text)
    elif group_type_node.text == 'MCI':
        mci_list.append(subject_id_node.text)
    else:
        print group_type_node.text
        raise NotImplementedError("group type " + group_type_node.text + " not understood.")

subject_fold_list = os.listdir(base_img_data_path)

count = 0
for i in xrange(len(subject_fold_list)):
    subject_fold_path = os.path.join(base_img_data_path, subject_fold_list[i])
    for imgtype_fold in os.listdir(subject_fold_path):
        imgtype_fold_path = os.path.join(subject_fold_path, imgtype_fold)
        # if 'MPR__GradWarp__B1_Correction__N3__Scaled' in imgtype_fold:
        if len(os.listdir(subject_fold_path)) == 1 or imgtype_fold.endswith('Scaled'):
            for time_fold in os.listdir(imgtype_fold_path):
                time_fold_path = os.path.join(imgtype_fold_path, time_fold)
                for series_fold in os.listdir(time_fold_path):
                    series_fold_path = os.path.join(time_fold_path, series_fold)
                    if len(os.listdir(time_fold_path)) != 1:
                        print series_fold + subject_fold_list[i]
                    else:
                        for file in os.listdir(series_fold_path):
                            if file.endswith('.nii'):
                                file_path = os.path.join(series_fold_path, file)
                                # dest path
                                if subject_fold_list[i] in lmci_list:
                                    subject_group = 'LMCI'
                                elif subject_fold_list[i] in emci_list:
                                    subject_group = 'EMCI'
                                elif subject_fold_list[i] in normal_list:
                                    subject_group = 'NC'
                                elif subject_fold_list[i] in ad_list:
                                    subject_group = 'AD'
                                elif subject_fold_list[i] in smc_list:
                                    subject_group = 'SMC'
                                elif subject_fold_list[i] in mci_list:
                                    subject_group = 'MCI'
                                else:
                                    raise NotImplementedError("group type " + subject_fold_list[i] + " not understood.")

                                destination_path = os.path.join(dest_img_path, subject_group, subject_fold_list[i])

                                if not os.path.exists(destination_path):
                                    os.mkdir(destination_path, 0755)

                                # os.rename(file_path, destination_path)
                                shutil.copy2(file_path, destination_path)

                                count += 1
                                print('copy the ' + str(count) + ': ' + file)
        else:
            print subject_fold_list[i] + imgtype_fold


    # imgtype_fold_list = os.listdir(subject_fold_list[i])
    # for j in xrange(len(imgtype_fold_list)):
    #     time_fold_path = os.path.join(imgtype_fold_path, imgtype_fold_list[j])
    #     name = imgtype_fold_list[j]
    #     if 'Spatially_' in name:
    #         time_fold_list = os.listdir(time_fold_path)
    #         for k in xrange(len(time_fold_list)):
    #             series_fold_path = os.path.join(time_fold_path, time_fold_list[k])
    #             series_fold_list = os.listdir(series_fold_path)
    #             if len(series_fold_list) != 1:
    #                 print subject_fold_list[i]
    #             else:
    #                 for file in os.listdir(series_fold_list[0]):
    #                     if file.endswith('.nii'):
    #                         file_path = os.path.join(series_fold_path, file)
    #                         # dest path
    #                         if subject_fold_list[i] in lmci_list:
    #                             subject_group = 'LMCI'
    #                         elif subject_fold_list[i] in emci_list:
    #                             subject_group = 'EMCI'
    #                         elif subject_fold_list[i] in normal_list:
    #                             subject_group = 'NC'
    #                         elif subject_fold_list[i] in ad_list:
    #                             subject_group = 'AD'
    #                         elif subject_fold_list[i] in smc_list:
    #                             subject_group = 'SMC'
    #                         else:
    #                             raise NotImplementedError("group type " + subject_fold_list[i] + " not understood.")
    #
    #                         destination_path = os.path.join(dest_img_path, subject_group)
    #
    #                         # os.rename(file_path, destination_path)
    #                         shutil.move(file_path, destination_path)
    #         # for root, dirs, files in os.walk(name):
    #         #     for file in files:
    #         #         if file.endswith('.txt'):
    #         #             print file

raw_input("Press Enter to continue")
