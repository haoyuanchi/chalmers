close all;
clear;

model_path = '/home/hychi/Research/caffe/models/';

% addpath(genpath('/home/hychi/Research/caffe/mywork/cvprw15'));
addpath('/home/hychi/Research/caffe');
addpath('/home/hychi/Research/caffe/matlab');
addpath('/home/hychi/Research/caffe/matlab/+caffe');

% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;

% top K returned images
top_k = 1000;
feat_len = 32;
binary_len = 64;

% set result folder
result_folder = '../analysis';

% models
model_file = '../models/32/SSDH32_iter_50000.caffemodel';
% model definition
model_def_file = '../models/32/deploy.prototxt';

% train-test
test_file_list = '～/Research/caffe/data/cifar10-dataset/test-file-list.txt';
test_label_file = '～/Research/caffe/data/cifar10-dataset/test-label.txt';
train_file_list = '～/Research/caffe/data/cifar10-dataset/train-file-list.txt';
train_label_file = '～/Research/caffe/data/cifar10-dataset/train-label.txt';

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)


% --- settings end here ---

% outputs
feat_test_file = sprintf('%s/feat-%s-test.mat', result_folder, int2str(feat_len));
feat_train_file = sprintf('%s/feat-%s-train.mat', result_folder, int2str(feat_len));
binary_test_file = sprintf('%s/binary-%s-test.mat', result_folder, int2str(feat_len));
binary_train_file = sprintf('%s/binary-%s-train.mat', result_folder, int2str(feat_len));

% map and precision outputs
map_file = sprintf('%s/map.txt', result_folder);
precision_file = sprintf('%s/precision-at-k.txt', result_folder);

% feature extraction- training set
if exist(binary_train_file, 'file') ~= 0
    load(binary_train_file);
else
    feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
    save(feat_train_file, 'feat_train', '-v7.3');    
    binary_train = gen_binary(feat_train);
    save(binary_train_file,'binary_train','-v7.3');
end

% feature extraction- test set
if exist(binary_test_file, 'file') ~= 0
    load(binary_test_file);
else
    feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
    save(feat_test_file, 'feat_test', '-v7.3');
    binary_test = gen_binary(feat_test);
    save(binary_test_file,'binary_test','-v7.3');
end
    


trn_label = load(train_label_file);
tst_label = load(test_label_file);

[map, precision_at_k] = precision( trn_label, binary_train, tst_label, binary_test, top_k, 1);
fprintf('MAP = %f\n',map);
save(map_file, 'map', '-ascii');
P = [[1:1:top_k]' precision_at_k'];
save(precision_file, 'P', '-ascii');