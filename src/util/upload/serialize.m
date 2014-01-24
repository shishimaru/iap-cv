function [] = serialize(confidence, dataName)
% confidence : [(# of image), (# of classes)]
% dataName   : 'test'(default) or 'val'

if(nargin == 1), dataName = 'test'; end

% IMPORTANT: UPDATE DATASET FOLDER
dataset_folder = '../viscomp/';
devkit_folder = '../devkit/';
cache_folder = '../cache/';

% load information about test data
fprintf('loading classes...');
load('classes.mat');

% check error of class size
if(size(confidence, 2) ~= length(classes)), 
    error('Dimention of confidence must be same to classes %d\n', length(classes));
end

% load annotations
fprintf('loading annotations_%s.mat...', dataName);
load(fullfile(cache_folder, sprintf('annotations_%s.mat', dataName)));
if(strcmp(dataName, 'val') == 1)
    annotations = annotations_val;
elseif(strcmp(dataName, 'test') == 1)
    annotations = annotations_test;
end
fprintf('done\n');

% check error of vector size
if(size(confidence, 1) > length(annotations)),
    error('Vector size of confidence is larger than one of annotations\n');
end

% get the list of image names for test data as shown below
annotations = annotations(1:size(confidence, 1), :);
image_names = cellfun(@(x) x.annotation.filename, annotations, 'UniformOutput', false);

% start to serialize the confidence
upload_file_name = sprintf('util/upload/upload_%s_%d.txt', dataName, size(confidence,1));
fprintf('serializing %s\n', upload_file_name);
createUploadFile(image_names, classes, confidence, upload_file_name);
fprintf('done\n');
end