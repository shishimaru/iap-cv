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

centroids = cell(length(classes));

%% Train SVM
svm_type = 0; % 0: linear-kernel, 1:RBF-kernel
num_components = 3;
models = cell(length(classes), num_components);

for i = 1:length(classes)
    % Split feature space
    idx = find(ytrain(:,i)==1);
    Xpos = Xtrain(idx,:);
    centroids{i} = kmeansFast(Xpos, num_components);
    dist = pdist2(Xtrain, centroids{i});
    [~,idx_comp] = min(dist, [], 2);
    for j = 1:num_components
        % Split training data 
        idx = find(idx_comp == j);
        X_sub = Xtrain(idx,:);
        y_sub = ytrain(idx,:);
        
        % Exception
        if length(ytrain) == length(find(ytrain==1))
            model{i,j}.type = 'fixed';
            model{i,j}.output = ytrain(1); % always anser ytrain(1)
            continue;
        end

        % modify training data so that liblinear and libSVM can correctly work
        idx = find(y_sub(:,i)==1);
        X = X_sub;
        X(1,:) = X_sub(idx(1),:);
        X(idx(1),:) = X_sub(1,:);
        Y = y_sub(:,i);
        Y(1) = y_sub(idx(1),i);
        Y(idx(1)) = y_sub(1,i);
        
        % Train SVM
        fprintf('training SVM of class(%13s), component %d (%d/%d)...', classes{i}, j, size(X_sub,1), size(Xtrain,1));
        if svm_type == 0
            %opt = sprintf('-s 2 -B 1 -c %f -q', svm_c);
            opt = svm_options{i};
            models{i,j} = train(Y, sparse(double(X)), opt);
            models{i,j}.type = 'linear-SVM';
        elseif svm_type == 1
            opt = sprintf('-s 0 -t 2 -c %f -q', svm_c);
            models{i,j} = svmtrain(Y, sparse(double(X)), opt);
            models{i,j}.type = 'RBF-SVM';
        end
        fprintf('done\n');
    end
end


%% Validate
% compute AP
accuracies = zeros(size(classes,2),1);
for i = 1:length(classes)
    % select a model
    dist = pdist2(Xval, centroids{i});
    [~,idx_comp] = min(dist, [], 2);
    prob = zeros(size(Xval,1),1);
    for j = 1:num_components
        % Split data 
        idx = find(idx_comp == j);
        X_sub = Xval(idx,:);
        y_sub = yval(idx,:);

        % predict
        if strcmp(models{i,j}.type, 'fixed')
            prob(idx) = models{i,j}.output * ones(size(X_sub,1),1);
        elseif strcmp(models{i,j}.type, 'linear-SVM')
%            [~, ~, prob(idx)] = predict(y_sub(:,i), sparse(double(X_sub)), models{i,j}, '-q');
            [~, ~, prob(idx)] = predict(y_sub(:,i), sparse(double(X_sub)), models{i,1}, '-q');
        elseif strcmp(models{i,j}.type, 'RBF-SVM')
            [~, ~, prob(idx)] = svmpredict(y_sub(:,i), sparse(double(X_sub)), models{i,j}, '-q');
        end
    end

    AP = computeAP(prob, yval(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Val Accuracy Average : %0.3f%%\n', mean(accuracies));

pause;

%% Train SVM
%svm_c_cand = [0.001 0.01 0.1 1 10];
%svm_c_cand = 0.05:0.001:0.2;
%svm_c_cand = 0.2:0.001:1.0;% continued...
svm_c_cand = 1;% for submission
svm_type = 0; % 0: linear-kernel, 1:RBF-kernel

result_AP = zeros(length(classes), length(svm_c_cand));

for iii = 1:length(svm_c_cand)
    svm_c = svm_c_cand(iii);
    fprintf('----- svm_c = %d -----\n', svm_c);
    % train
    for i = 1:length(classes)
        % modify training data so that liblinear and libSVM can correctly work
        idx = find(ytrain(:,i)==1);
        X = Xtrain;
        X(1,:) = Xtrain(idx(1),:);
        X(idx(1),:) = Xtrain(1,:);
        Y = ytrain(:,i);
        Y(1) = ytrain(idx(1),i);
        Y(idx(1)) = ytrain(1,i);

        %svm_c = 1;
        if svm_type == 0
            opt = sprintf('-s 2 -B 1 -c %f -q', svm_c);
            %opt = svm_options{i};
            models{i} = train(Y, sparse(double(X)), opt);
            models{i}.type = 'linear-SVM';
        elseif svm_type == 1
            opt = sprintf('-s 0 -t 2 -c %f -q', svm_c);
            models{i} = svmtrain(Y, sparse(double(X)), opt);
            models{i}.type = 'RBF-SVM';
        end
    end
    
    if 0
        % compute AP
        accuracies = zeros(size(classes,2),1);
        for i = 1:length(classes)
            if svm_type == 0
               [~, ~, prob] = predict(ytrain(:,i), sparse(double(Xtrain)), model{i}, '-q');
            elseif svm_type == 1
                [~, ~, prob] = svmpredict(ytrain(:,i), sparse(double(Xtrain)), model{i}, '-q');
            end
            AP = computeAP(prob, ytrain(:,i), 1)*100;
            accuracies(i, 1) = AP;
            fprintf('Train Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
        end;
        fprintf('Train Accuracy Average : %0.3f%%\n', mean(accuracies));
    end
    
    
    %% Validate
    % compute AP
    accuracies = zeros(size(classes,2),1);
    for i = 1:length(classes)
        if svm_type == 0
            [~, ~, prob] = predict(yval(:,i), sparse(double(Xval)), models{i}, '-q');
        elseif svm_type == 1
            [~, ~, prob] = svmpredict(yval(:,i), sparse(double(Xval)), models{i}, '-q');
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
for i = 1:length(classes)
    % modify training data so that liblinear and libSVM can correctly work
    idx = find(ytrain(:,i)==1);
    X = Xtrain;
    X(1,:) = Xtrain(idx(1),:);
    X(idx(1),:) = Xtrain(1,:);
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
% predict
probs = [];
for i =1:length(classes)
    if svm_type == 0
        [~, ~, prob] = predict(zeros(size(Xtest,1),1), sparse(double(Xtest)), models{i}, '-q');
    elseif svm_type == 1
        [~, ~, prob] = svmpredict(zeros(size(Xtest,1),1), sparse(double(Xtest)), models{i}, '-q');
    end
    probs(end+1,:) = prob;
end
serialize(probs', 'test');
