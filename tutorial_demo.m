addpath(genpath('feature-extraction'));
addpath(genpath('liblinear-1.94'));

%% STEP 1
% Compile the feature-extraction toolbox
%  - cd to feature-extraction folder
%  - type compile (there will be warnings printed - ignore them)

%% STEP 2
% Compile liblinear (only for non-Windows machines)
%  - cd to liblinear-1.94/matlab
%  - edit line: MATLABDIR ?= /usr/local/MATLAB/R2012b
%  - MATLABDIR can be obtained by typing "matlabroot" in Matlab
%  - Go to the same folder in terminal and type make (again, ignore warnings)

%% STEP 3
% Read instructions of feature-extraction package available at 
% feature-extraction/README.md or https://github.com/adikhosla/feature-extraction
% 
% Implement the code shown there to extract features

% Initialize variables for calling datasets_feature function
info = load('filelists.mat');
datasets = {'tutorial'};
train_lists = {info.trainfiles};
test_lists = {info.testfiles};

c = conf();
c.verbosity = 1;

% Use the following settings for your first try:
%  - feature is color
%  - dictionary size is 20
feature = 'ssim';
c.feature_config.(feature).dictionary_size=20;

% Some default parameters for faster learning (do not modify)
c.feature_config.(feature).num_desc = 2e4;
c.feature_config.(feature).descPerImage = 200;

% Compute train and test features
datasets_feature(datasets, train_lists, test_lists, feature, c);

% Load train and test features to variables:
% 'train_features' and 'test_features'
train_features = load_feature(datasets{1}, feature, 'train', c);
test_features = load_feature(datasets{1}, feature, 'test', c);

%% STEP 4
% Try playing with the parameters and names of features above to see how it
% affects the performance of your object recognition system.
%
% Note: delete 'cache' folder when you modify parameters
%
% Some possible things to play with:
%  - following types of features are available:
%    'color', 'gist', 'hog2x2', 'hog3x3', 'lbp', 'sift', 'ssim'
%
%  - c.feature_config.(feature).dictionary_size:
%    try increasing/decreasing it to see what happens
%  
%  - c.feature_config.(feature).pyramid_levels:
%    try different values from 0 to 4


%% YOU SHOULD NOT NEED TO MODIFY BELOW THIS LINE
% Sample code for usage of features with Liblinear SVM classifier:

svm_options = '-s 2 -B 1 -c 1 -q';
model = train(info.trainlabels', sparse(double(train_features)), svm_options);
% predicted_labels = predict(info.testlabels', sparse(double(test_features)), model);

[predicted_labels, ~, predicted_conf] = predict(info.testlabels', sparse(double(test_features)), model);


% subplot images in the order of confidence
[~,b] = sort(predicted_conf, 'descend');
ha=tight_subplot(5,6,[0.01 0.01], [0.05 0.05], [0.05 0.05]);
for i = 1:30
    axes(ha(i));
    im=imread(test_lists{1}{b(i)});
    image(im);
    axis off;
end




