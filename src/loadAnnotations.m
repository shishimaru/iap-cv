function [annotations_train, annotations_val, annotations_test] = loadAnnotations()
globals();
% Load annotations

%% Load dataset
load(fullfile(devkit_folder, 'filelists.mat'));
load(fullfile(devkit_folder, 'classes.mat'));

fprintf('loading annotations...');

filename = fullfile(cache_folder, 'annotations_train.mat');
if ~exist(filename, 'file')
    annotations_train = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_train');
else
    load(filename);
end

filename = fullfile(cache_folder, 'annotations_val.mat');
if ~exist(filename, 'file')
    annotations_val = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), val_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_val');
else
    load(filename);
end

filename = fullfile(cache_folder, 'annotations_test.mat');
if ~exist(filename, 'file')
    annotations_test = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), test_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_test');
else
    load(filename);
end

fprintf('done\n');

end
%End of function