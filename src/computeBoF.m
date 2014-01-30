% Script to compute HOG2x2 or SIFT features
%
% 
% NOTE: This is adapted from Aditya's feature-extraction codes.
%
% Copyright Aditya Khosla http://mit.edu/khosla
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, J. Xiao, A. Torralba, A. Oliva
%   Memorability of Image Regions
%   Advances in Neural Information Processing Systems (NIPS) 2012
%

addpath(genpath(pwd));

% Initialize variables for calling datasets_feature function
info = load('../devkit/filelists.mat');
train_lists = {info.train_data.images};
test_lists = {info.test_data.images};
val_lists = {info.val_data.images};
datasets = {'iap-cv'};
feature = 'hog2x2';
%feature = 'sift';
load('../devkit/classes.mat');
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
ytrain = []; yval = [];


fprintf('loading ytrain...');
for i=1:size(annotations_train,1)
     clss = annotations_train{i}.annotation.classes;
     ytmp = [];
     for j=1:size(classes,2)
        ytmp(1,end+1) = str2double(clss.(classes{j}));
     end
     ytrain(end+1,:) = ytmp;
end
fprintf('done\n');

fprintf('loading yval...');
for i=1:size(annotations_val,1)
     clss = annotations_val{i}.annotation.classes;
     ytmp = [];
     for j=1:size(classes,2)
        ytmp(1,end+1) = str2double(clss.(classes{j}));
     end
     yval(end+1,:) = ytmp;
end
fprintf('done\n');

for i = 1:length(val_lists{1})
    train_lists{1}{end+1} = val_lists{1}{i};
end

train_labels = [ytrain; yval];
test_labels = zeros(5000,1);

% Load the configuration and set dictionary size to 20 (for fast demo)
c = conf();
c.feature_config.(feature).dictionary_size=20;



% Compute train and test features
datasets_feature(datasets, train_lists, test_lists, feature, c);

% Load train and test features
train_features = load_feature(datasets{1}, feature, 'train', c);
test_features = load_feature(datasets{1}, feature, 'test', c);

HOG2x2_train = train_features(1:8000,:);
HOG2x2_val = train_features(8001:end,:);
HOG2x2_test = test_features;
save('../feature-BoF/HOG2x2_train.mat', 'HOG2x2_train');
save('../feature-BoF/HOG2x2_val.mat', 'HOG2x2_val');
save('../feature-BoF/HOG2x2_test.mat', 'HOG2x2_test');
