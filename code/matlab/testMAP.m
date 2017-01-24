clear all;
close all;

%% deep hashing
deephashmethods = {'ssdh'};
nhmethods = length(hashmethods);
loopnbits = [48];
db_name = 'CIFAR10';

% train-test
test_file_list = '/home/hychi/Research/caffe/data/cifar10/imgs/test-file-list.txt';
test_label_file = '/home/hychi/Research/caffe/data/cifar10/imgs/test-label.txt';
train_file_list = '/home/hychi/Research/caffe/data/cifar10/imgs/train-file-list.txt';
train_label_file = '/home/hychi/Research/caffe/data/cifar10/imgs/train-label.txt';

trn_label = load(train_label_file);
tst_label = load(test_label_file);

top_k = 1000;

for i =1:length(loopnbits)
    load('./ssdh_binary48-test.mat');
    load('./ssdh_binary48-train.mat');
    fprintf('======start %d bits encoding======\n\n', loopnbits(i));
    for j = 1:nhmethods
        [precision{i, j}, recall{i, j}, mAP{i, j}] = averagePrecision( trn_label, binary_train, tst_label, binary_test, top_k, 1);
    end
end

% save result
result_name = ['./ResultSaveToMat/final_', db_name, '_result.mat'];
save(result_name, 'precision', 'recall', 'mAP', 'hashmethods', 'nhmethods', 'loopnbits');

%% ################ plot 
% plot attribution
line_width=2;
marker_size=8;
xy_font_size=14;
legend_font_size=12;
linewidth = 1.6;
title_font_size=xy_font_size;

pos = [1 : 1 : top_k];
choose_bits = 1; % i: choose the bits to show

%% show recall vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
for j = 1: nhmethods
    pos = param.pos;
    recc = recall{choose_bits, j};    
    p = plot(pos, recc);
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('The number of retrieved samples');
h2 = ylabel(['Recall @ ', str_nbits, ' bits']);
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
for j = 1: nhmethods
    prec = precision{choose_bits, j};    
    p = plot(pos, prec);
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('The number of retrieved samples');
h2 = ylabel(['Precision @ ', str_nbits, ' bits']);
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. recall , i is the selection of which bits.
figure('Color', [1 1 1]); hold on;

for j = 1: nhmethods
    p = plot(recall{choose_bits, j}, precision{choose_bits, j});
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
h1 = xlabel(['Recall @ ', str_nbits, ' bits']);
h2 = ylabel('Precision');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
set(gca, 'linewidth', linewidth);
box on;
grid on;
hold off;

%% show mAP. 
figure('Color', [1 1 1]); hold on;
for j = 1: nhmethods
    map = [];
    for i = 1: length(loopnbits)
        map = [map, MAP{i, j}];
    end
    p = plot(log2(loopnbits), map);
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP)');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
set(gca, 'xtick', log2(loopnbits));
set(gca, 'XtickLabel', {'8', '16', '32', '64', '128'});
set(gca, 'linewidth', linewidth);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on;
grid on;
hold off;



