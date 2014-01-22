% demo to show how to compute average precision (AP) on validation data
%
% the predicted confidence is going to be random values for each class
% i.e. the result will change each time you run the code

% IMPORTANT: UPDATE DATASET FOLDER
dataset_folder = '../viscomp/';

% load list of validation files and list of classes
load('filelists.mat', 'val_data');
load('classes.mat');

fprintf('loading annotations...');
annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), val_data.annotations', 'UniformOutput', false);
fprintf('done\n');

average_precision = zeros(length(classes), 1);

% compute average precision, using rand to predict the confidence for each image
for i=1:length(classes)
    ground_truth_labels = cellfun(@(x) str2double(x.annotation.classes.(classes{i})), annotations);
    predicted_labels = rand(size(annotations));
    average_precision(i) = computeAP(predicted_labels, ground_truth_labels, 1)*100;
    fprintf('class: %s, average precision: %.02f%%\n', classes{i}, average_precision(i));
end
fprintf('mean average precision: %.02f%%\n', mean(average_precision));