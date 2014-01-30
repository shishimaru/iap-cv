%% init
clear ; close all; clc
globals();
featureDL_folder = '../feature-deeplearning';
featureBoF_folder = '../feature-BoF';
featureSIFT_folder = '../feature-sift';
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
load(fullfile(devkit_folder, 'classes.mat')); % load classes
classNames = {'bicycle', 'bird', 'bottle', 'car', 'chair',...
    'diningtable', 'dog', 'person', 'pottedplant', 'sofa'};

FLAG_NORMALIZE = false;
FLAG_POLYNOMIAL = false;
FLAG_FEATURE_BOF = true;
FLAG_FEATURE_BNDBOX = true;
FLAG_FEATURE_SIFT = false;
FLAG_SHUFFLE = false;

%% Load deep-learning feature
Xtrain = []; Xval = []; Xtest = [];
ytrain = []; yval = [];
%%  load train
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

%%  load validation
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

%%  load test
fprintf('loading Xtest...');
for i=1:25,
    filename = sprintf('%s/test_%d.mat', featureDL_folder, i);
    load(filename);
    Xtest = [Xtest; scores];
end
fprintf('done\n');

%% load BoF feature
if(FLAG_FEATURE_BOF)
    featureBoF = load(sprintf('%s/BoF_train.mat', featureBoF_folder));
    Xtrain = [Xtrain, featureBoF.Xtrain];
    featureBoF = load(sprintf('%s/BoF_val.mat', featureBoF_folder));
    Xval = [Xval, featureBoF.Xval];
    featureBoF = load(sprintf('%s/BoF_test.mat', featureBoF_folder));
    Xtest = [Xtest, featureBoF.Xtest];
end

%% load Bndbox feature
if(FLAG_FEATURE_BNDBOX)
    filename = fullfile(cache_folder, '/', sprintf('../feature-bndbox.mat'));
    if exist(filename, 'file')
        load(filename);
    else
        box_feature_train = lr_extractFeature('train');
        box_feature_val = lr_extractFeature('val');
        box_feature_test = lr_extractFeature('test');
        [box_feature_train, mu, sigma] = lr_featureNormalize(box_feature_train);
        box_feature_val = (box_feature_val - repmat(mu, size(box_feature_val,1), 1))...
            ./ repmat(sigma, size(box_feature_val,1), 1);
        box_feature_test = (box_feature_test - repmat(mu, size(box_feature_test,1), 1))...
            ./ repmat(sigma, size(box_feature_test,1), 1);
        Xtrain = [Xtrain, box_feature_train/100];
        Xval = [Xval, box_feature_val/100];
        Xtest = [Xtest, box_feature_test/100];
        save(filename, 'box_feature_train', 'box_feature_val', 'box_feature_test');
    end
end

%% load SIFT feature
if(FLAG_FEATURE_SIFT)
    featureSIFT = load(sprintf('%s/SIFT_train.mat', featureSIFT_folder));
    Xtrain = [Xtrain, featureSIFT.SIFT_train/10];
    featureSIFT = load(sprintf('%s/SIFT_val.mat', featureSIFT_folder));
    Xval = [Xval, featureSIFT.SIFT_val/10];
    featureSIFT = load(sprintf('%s/SIFT_test.mat', featureSIFT_folder));
    Xtest = [Xtest, featureSIFT.SIFT_test/10];
end

%% Polynomial
if FLAG_POLYNOMIAL
    Xtrain = [Xtrain Xtrain.^2];
    Xval = [Xval Xval.^2];
    Xtest = [Xtest Xtest.^2];
end

%% cleaning data
if 0
    idx = cleaningByBndbox('train');
    Xtrain(idx, :) = [];
    ytrain(idx, :) = [];
    if(FLAG_SHUFFLE)
        idx = cleaningByBndbox('val');
        Xval(idx, :) = [];
        yval(idx, :) = [];
    end
end

%% Normalize
if FLAG_NORMALIZE
    [Xtrain, mu, sigma] = lr_featureNormalize(Xtrain);
    Xval = (Xval - repmat(mu, size(Xval,1), 1)) ./ repmat(sigma, size(Xval,1), 1);
    Xtest = (Xtest - repmat(mu, size(Xtest,1), 1)) ./ repmat(sigma, size(Xtest,1), 1);
end
%{
save(fullfile(featureDL_folder, '/', 'train_all_x.mat'), 'Xtrain');
save(fullfile(featureDL_folder, '/', 'train_all_y.mat'), 'ytrain');
save(fullfile(featureDL_folder, '/', 'val_all_x.mat'), 'Xval');
save(fullfile(featureDL_folder, '/', 'val_all_y.mat'), 'yval');
save(fullfile(featureDL_folder, '/', 'test_all_x.mat'), 'Xtest');
pause;
%}

%% Shuffle train, val
if FLAG_SHUFFLE
    Xtv = [Xtrain; Xval];
    ytv = [ytrain; yval];
    idx = randperm(size(Xtv, 1));
    Xtv = Xtv(idx,:);
    ytv = ytv(idx,:);
    Xtrain = Xtv(1:size(Xtrain,1),:);
    ytrain = ytv(1:size(ytrain,1),:);
    Xval = Xtv(size(Xtrain)+1:end,:);
    yval = ytv(size(ytrain)+1:end,:);
end

%% Add bias
Xtrain = [ones(size(Xtrain,1),1) Xtrain];
Xval = [ones(size(Xval,1),1) Xval];
Xtest = [ones(size(Xtest,1),1) Xtest];

%% ============= Part 1: Select best parameters        =============
if 0
    [AP, LAMBDA, ALPHA] = lr_selectParam(classes, Xtrain, ytrain, Xval, yval);
    fprintf('BEST AP: %0.5f%%, LAMBDA: %0.5f, ALPHA: %0.5f\n', AP, LAMBDA, ALPHA);
    pause;
else
    if FLAG_NORMALIZE % normalize
        LAMBDA = 30;
        ALPHA = 5;
    elseif FLAG_POLYNOMIAL % polynomial AP=71.534%
        LAMBDA = 0.02;
        ALPHA = 150;
    elseif FLAG_FEATURE_SIFT % BoF+Bndbox+SIFT feature AP=%
        LAMBDA = 0.001;
        ALPHA = 80;
    elseif FLAG_FEATURE_BNDBOX % BoF+Bndbox feature AP=72.97053%
        LAMBDA = 0.001;
        ALPHA = 80;
    elseif FLAG_SHUFFLE % with BoF+Shuffle AP=75.65463%
        LAMBDA = 0.006;
        ALPHA = 75;
    elseif FLAG_FEATURE_BOF % with BoF feature AP=71.710%
        LAMBDA = 0.01;
        ALPHA = 100;    
    else % no nomalize/polynomial AP=71.678%
        LAMBDA = 0.02;
        ALPHA = 175;
    end
end;

%Xtrain = [Xtrain; Xval];
%ytrain = [ytrain; yval];

%% ============= Part 2: Regularization and Accuracies =============
% Optimize
fprintf('computing thetas...\n');
ITER = 2000;  % 2000
filename = fullfile(cache_folder, '/',...
    sprintf('lr_thetas_L%0.5f_A%0.5f_I%d.mat', LAMBDA, ALPHA, ITER))
if exist(filename, 'file')
    load(filename);
else
    thetas = [];
    for i=1:size(classes,2)
        fprintf('start %s\n', classes{i});
        theta = zeros(size(Xtrain, 2), 1);
        for j=1:ITER
            [J, grad] = costFunctionReg(theta, Xtrain,...
                ytrain(:,i), LAMBDA);
            theta = theta - ALPHA * grad;
            
            if(mod(j, 10) == 0)
                fprintf('cost : %0.4f\n', J);
            end
        end
        thetas(:,end+1) = theta;
        fprintf('done %s\n', classes{i});
    end
    save(filename, 'thetas');
end    
fprintf('done\n');

%{
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
%}

%% Compute accuracy on our training set
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    prob = Xtrain * thetas(:,i);
    AP = computeAP(prob, ytrain(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Train Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Train Accuracy Average : %0.3f%%\n', mean(accuracies)); 

%% Compute accuracy on our validation set
accuracies = zeros(size(classes,2),1);
probs = [];
for i=1:size(classes,2)
    prob = Xval * thetas(:,i);
    probs(:, end + 1) = prob;
    AP = computeAP(prob, yval(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
save('../probs_uchida_best.mat', 'probs');
fprintf('Val Accuracy Average : %0.3f%%\n', mean(accuracies)); 

%% predict test set
probs = [];
for i=1:size(Xtest,1),
   prob = Xtest(i,:) * thetas;
   probs(end+1,:) = prob;
end
save('../probs_uchida_best.mat', 'probs');
serialize(probs, 'test');
fprintf('done');

if 1 %% DEBUG
    for i=105:size(Xval,1)
        filename = [dataset_folder, 'val/images/',...
            annotations_val{i}.annotation.filename, '.jpg'];
        img = imread(filename);
        figure(111);
        imshow(img);
        
        [prob, classIdx] = sort(sigmoid(Xval(i,:) * thetas), 'descend');
        k = 1;
        for j=1:length(classes)
            fprintf('%11s [%0.2f]\n', classes{classIdx(k)}, prob(j));
            k = k + 1;
        end
        fprintf('\n');
        pause;
        close(111);
    end
end