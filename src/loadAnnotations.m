function loadAnnotations()
% Load annotations

dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache';
dataset        = 'train';
feature        = 'hog2x2'; % select only from hog2x2, hog3x3, sift(slow), ssim(too slow)
desc_per_img   = 100; % shouldn't change
dic_size       = 1000; % should change
flag_white     = true;
debug          = true; % if enabled, we will use small subset of data

%% Load dataset
load(fullfile(devkit_folder, 'filelists.mat'));
load(fullfile(devkit_folder, 'classes.mat'));

fprintf('loading annotations...');

filename = fullfile(cache_folder, 'annotations_train.mat');
if ~exist(filename, 'file')
    annotations_train = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_train');
end

filename = fullfile(cache_folder, 'annotations_val.mat');
if ~exist(filename, 'file')
    annotations_val = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), val_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_val');
end

filename = fullfile(cache_folder, 'annotations_test.mat');
if ~exist(filename, 'file')
    annotations_test = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), test_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_test');
end

fprintf('done\n');

end
%End of function