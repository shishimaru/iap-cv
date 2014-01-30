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


%% Load SVM models for 1st stage
load('../svm-models/models_SVM.mat');
first_svm_models = models;
clear models;


%% Train SVM
%svm_c_cand = [0.001 0.01 0.1 1 10];
%svm_c_cand = [0.5:0.1:5];
%svm_c_cand = 0.05:0.01:0.2;
%svm_c_cand = 0.2:0.001:1.0;% continued...
svm_c_cand = 1;% for submission
svm_type = 0; % 0: linear-kernel, 1:RBF-kernel

result_AP = zeros(length(classes), length(svm_c_cand));

scale = 1;

for iii = 1:length(svm_c_cand)
    svm_c = svm_c_cand(iii);
    fprintf('----- svm_c = %d -----\n', svm_c);
    % do 1st stage
    new_feat = zeros(0,0);
    for i = 1:length(classes)
        % do 1st stage
        [predicted_labels, ~, prob] = predict(ytrain(:,i), sparse(double(Xtrain)), first_svm_models{i}, '-q');
        %new_feat(:,end+1) = predicted_labels;
        new_feat(:,end+1) = prob;
    end

    % do 2nd stage (train)
    for i = 1:length(classes)
        % extend features
        if 0
            tmp = ytrain .* scale;
            tmp(:,i) = 0; % suppress self-fire
            XX = [Xtrain, tmp];
        else
            tmp = new_feat .* scale;
            tmp(:,i) = 0; % suppress self-fire
            XX = [Xtrain, tmp];
        end
        XX = new_feat; % use only outputs of 1st stage
        
        % modify training data so that liblinear and libSVM can correctly work
        idx = find(ytrain(:,i)==1);
        X = XX;
        X(1,:) = XX(idx(1),:);
        X(idx(1),:) = XX(1,:);
        Y = ytrain(:,i);
        Y(1) = ytrain(idx(1),i);
        Y(idx(1)) = ytrain(1,i);

        if svm_type == 0
            %opt = sprintf('-s 2 -B 1 -c %f -q', svm_c);
            opt = sprintf('-s 2 -c %f -q', svm_c); %HACK: no bias!!
            %opt = svm_options{i};
            models{i} = train(Y, sparse(double(X)), opt);
            models{i}.type = 'linear-SVM';
        elseif svm_type == 1
            opt = sprintf('-s 0 -t 2 -c %f -q', svm_c);
            models{i} = svmtrain(Y, sparse(double(X)), opt);
            models{i}.type = 'RBF-SVM';
        end
    end
    
    %% Validate
    % compute AP
    accuracies = zeros(size(classes,2),1);
    
    % do 1st stage
    new_feat = zeros(0,0);
    for i = 1:length(classes)
        % do 1st stage
        [predicted_labels, ~, prob] = predict(yval(:,i), sparse(double(Xval)), first_svm_models{i}, '-q');
        %new_feat(:,end+1) = predicted_labels;
        new_feat(:,end+1) = prob;
    end
    
    % do 2nd stage
    for i = 1:length(classes)
        
        % extend features
        if 0
            % for preliminary experiment
            tmp = yval .* scale;
            tmp(:,i) = 0; % suppress self-fire
            X = [Xval, tmp];
        else
            % for honban
            tmp = new_feat .* scale;
            tmp(:,i) = 0; % suppress self-fire
            X = [Xval, tmp];
        end
        X = new_feat; % use only outputs of 1st stage
        
        % predict
        if svm_type == 0
            [~, ~, prob] = predict(yval(:,i), sparse(double(X)), models{i}, '-q');
        elseif svm_type == 1
            [~, ~, prob] = svmpredict(yval(:,i), sparse(double(X)), models{i}, '-q');
        end
        AP = computeAP(prob, yval(:,i), 1)*100;
        accuracies(i, 1) = AP;
        fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
        result_AP(i,iii) = AP;
    end;
    fprintf('Val Accuracy Average : %0.3f%%\n', mean(accuracies));
    
end

%% Show the best svm_c
[v,I] = max(result_AP,[],2);
for i = 1:length(classes)
    fprintf('Best SVM_C (%13s) : %.3f (%.3f%%)\n', classes{i}, svm_c_cand(I(i)), v(i));
end
fprintf('Val Accuracy Average : %0.3f%%\n', mean(v));

%% ReTrain SVM with the best parameters
% do 1st stage
new_feat = zeros(0,0);
for i = 1:length(classes)
    % do 1st stage
    [predicted_labels, ~, prob] = predict(ytrain(:,i), sparse(double(Xtrain)), first_svm_models{i}, '-q');
    %new_feat(:,end+1) = predicted_labels;
    new_feat(:,end+1) = prob;
end

% do 2nd stage (train)
for i = 1:length(classes)
    % extend features
    if 0
        tmp = ytrain .* scale;
        tmp(:,i) = 0; % suppress self-fire
        XX = [Xtrain, tmp];
    else
        tmp = new_feat .* scale;
        tmp(:,i) = 0; % suppress self-fire
        XX = [Xtrain, tmp];
    end
    XX = new_feat; % use only outputs of 1st stage
    
    % modify training data so that liblinear and libSVM can correctly work
    idx = find(ytrain(:,i)==1);
    X = XX;
    X(1,:) = XX(idx(1),:);
    X(idx(1),:) = XX(1,:);
    Y = ytrain(:,i);
    Y(1) = ytrain(idx(1),i);
    Y(idx(1)) = ytrain(1,i);
    
    if svm_type == 0
        opt = sprintf('-s 2 -B 1 -c %f -q', svm_c_cand(I(i)));
        models{i} = train(Y, sparse(double(X)), opt);
        models{i}.type = 'linear-SVM';
    elseif svm_type == 1
        opt = sprintf('-s 0 -t 2 -c %f -q', svm_c_cand(I(i)));
        models{i} = svmtrain(Y, sparse(double(X)), opt);
        models{i}.type = 'RBF-SVM';
    end
    models{i}.AP = v(i);
    save(fullfile(cache_folder, 'models_SVM.mat'), 'models');
end


%% Predict Test-set
% do 1st stage
new_feat = zeros(0,0);
for i = 1:length(classes)
    % do 1st stage
    [predicted_labels, ~, prob] = predict(zeros(size(Xtest,1),1), sparse(double(Xtest)), first_svm_models{i}, '-q');
    %new_feat(:,end+1) = predicted_labels;
    new_feat(:,end+1) = prob;
end

% do 2nd stage
probs = [];
for i =1:length(classes)
    tmp = new_feat;
    tmp(:,i) = 0; % suppress self-fire
    X = [Xtest, tmp];
    X = new_feat; % use only outputs of 1st stage

    if svm_type == 0
        [~, ~, prob] = predict(zeros(size(Xtest,1),1), sparse(double(Xtest)), models{i}, '-q');
    elseif svm_type == 1
        [~, ~, prob] = svmpredict(zeros(size(Xtest,1),1), sparse(double(Xtest)), models{i}, '-q');
    end
    probs(end+1,:) = prob;
end
serialize(probs', 'test');

