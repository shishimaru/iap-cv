%% init
clear all; close all; clc
globals();
featureDL_folder = '../feature-deeplearning';
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
load(fullfile(devkit_folder, 'classes.mat')); % load classes

%% Load Data
Xtrain = []; Xval = []; Xtest = [];
ytrain = []; yval = [];

%  load train
fprintf('loading Xtrain...');
for i=1:40,
    filename = sprintf('%s/train_%d.mat', featureDL_folder, i);
    load(filename);
    Xtrain = [Xtrain; scores];
end
fprintf('done\n');
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

%  load validation
fprintf('loading Xval...');
for i=1:10,
    filename = sprintf('%s/val_%d.mat', featureDL_folder, i);
    load(filename);
    Xval = [Xval; scores];
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

%  load test
fprintf('loading Xtest...');
for i=1:25,
    filename = sprintf('%s/test_%d.mat', featureDL_folder, i);
    load(filename);
    Xtest = [Xtest; scores];
end
fprintf('done\n');

%% DEBUG : shrink the image size
if(flag_debug)
    Xtrain = Xtrain(1:debug_num_imgs, :);
    ytrain = ytrain(1:debug_num_imgs, :);
    Xval   = Xval(1:debug_num_imgs, :);
    yval   = yval(1:debug_num_imgs, :);
    Xtest  = Xtest(1:debug_num_imgs, :);
end

%% Train SVM
% normalize
min_val = min(Xtrain);
max_val = max(Xtrain);
Xtrain = Xtrain - repmat(min_val, [size(Xtrain,1) 1]);
Xtrain = Xtrain ./ repmat((max_val - min_val), [size(Xtrain,1) 1]);

%%%$$$$$

% train
for i = 1:length(classes)
    svm_c = 1;
    opt = sprintf('-s 2 -B 1 -c %f -q', svm_c);
    model{i} = train(ytrain(:,i), sparse(double(Xtrain)), opt);
    %model{i} = svmtrain(ytrain(:,i), sparse(double(Xtrain)), '-s 0 -t 2 -q');
    
    % Reverse weights
    if model{i}.Label(1) == 0
        model{i}.w = -model{i}.w;
        model{i}.Label = [1;0];
    end
end

% compute AP
accuracies = zeros(size(classes,2),1);
for i = 1:length(classes)
    [~, ~, prob] = predict(ytrain(:,i), sparse(double(Xtrain)), model{i});
    %[~, ~, prob] = svmpredict(ytrain(:,i), sparse(double(Xtrain)), model{i});
    AP = computeAP(prob, ytrain(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Train Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Train Accuracy Average : %0.3f%%\n', mean(accuracies)); 


%% Evaluate SVM
% normalize
Xval = Xval - repmat(min_val, [size(Xval,1) 1]);
Xval = Xval ./ repmat((max_val - min_val), [size(Xval,1) 1]);

% compute AP
accuracies = zeros(size(classes,2),1);
for i = 1:length(classes)
    [~, ~, prob] = predict(yval(:,i), sparse(double(Xval)), model{i});
    %[~, ~, prob] = svmpredict(yval(:,i), sparse(double(Xval)), model{i});
    AP = computeAP(prob, yval(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Val Accuracy Average : %0.3f%%\n', mean(accuracies)); 


%% Predict Test-set
% normalize
Xtest = Xtest - repmat(min_val, [size(Xtest,1) 1]);
Xtest = Xtest ./ repmat((max_val - min_val), [size(Xtest,1) 1]);

% predict
probs = [];
for i =1:length(classes)
    [~, ~, prob] = predict(zeros(size(Xtest,1),1), sparse(double(Xtest)), model{i});
    %[~, ~, prob] = svmpredict(zeros(size(Xtest,1),1), sparse(double(Xtest)), model{i});
    probs(end+1,:) = prob;
end
serialize(probs', 'test');
