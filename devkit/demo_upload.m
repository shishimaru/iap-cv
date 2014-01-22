% this demo shows how to create file for upload to submit your final results

% IMPORTANT: UPDATE DATASET FOLDER
dataset_folder = '../viscomp/';

% load information about test data
load('classes.mat');
load('filelists.mat', 'test_data');

fprintf('loading annotations...');
annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), test_data.annotations', 'UniformOutput', false);
fprintf('done\n');

% get the list of image names for test data as shown below
image_names = cellfun(@(x) x.annotation.filename, annotations, 'UniformOutput', false);

% build a matrix of confidence of each image containing each object class
% where 'classes' variable contains the list and order of classes
fprintf('creating confidence matrix...');
confidence = rand([length(image_names) length(classes)]);
fprintf('done\n');

upload_file_name = 'sample_upload.txt';

fprintf('outputting upload file: %s\n', upload_file_name);
createUploadFile(image_names, classes, confidence, upload_file_name);