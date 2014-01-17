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

%% GO TO LINE 43. NO NEED TO TOUCH HERE.

% Initialize variables for calling datasets_feature function
info = load('filelists.mat');


datasets = {'tutorial'};
train_lists = {info.trainfiles};
test_lists = {info.testfiles};
feature = 'hog2x2';

% Load the configuration and set dictionary size to 20 (for fast demo)
c = conf();
c.verbosity = 1;
c.feature_config.(feature).dictionary_size = 20;

% Some default parameters for faster learning (do not modify)
c.feature_config.(feature).num_desc = 2e4;
c.feature_config.(feature).descPerImage = 200;

% Compute train and test features
datasets_feature(datasets, train_lists, test_lists, feature, c);

% Load train and test features
train_features = load_feature(datasets{1}, feature, 'train', c);
test_features = load_feature(datasets{1}, feature, 'test', c);

%% STEP 3: Play with different C parameters and see how the results change
% In the real scenario, you need to find the best performing C paramter
% based on the validation set, and you can test on the test set for the
% final performance.

% svm parameter setup
C_param = 0.1;
svm_options = ['-s 2 -B 1 -c ' num2str(C_param) ' -q'];

% SVM training with extracted training features
model = train(info.trainlabels', sparse(double(train_features)), svm_options);

% SVM testing with the trained model on the test set
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