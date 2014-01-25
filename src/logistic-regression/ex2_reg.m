%% init
clear ; close all; clc
globals();
featureDL_folder = '../feature-deeplearning';
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
load(fullfile(devkit_folder, 'classes.mat')); % load classes

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
%{
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
%}

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

%% DEBUG : shrink the image size
if(flag_debug)
    Xtrain = Xtrain(1:debug_num_imgs, :);
    ytrain = ytrain(1:debug_num_imgs, :);
    Xval   = Xval(1:debug_num_imgs, :);
    yval   = yval(1:debug_num_imgs, :);
    Xtest  = Xtest(1:debug_num_imgs, :);
end

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?

% Set regularization parameter lambda to 1 (you should vary this)

% Optimize
fprintf('computing thetas...\n');
LAMBDA = 0.1; % 0.1
ALPHA = 1;    % 1
ITER = 1000;  % 1000
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
            %{
            if(mod(j, 10) == 0)
                fprintf('cost : %0.03f\n', J);
            end
            %}
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
%{
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    p = mypredict(thetas(:,i), Xtrain);
    accuracy = mean(double(p == ytrain(:,i))) * 100;
    accuracies(i, 1) = accuracy;
    fprintf('Train Accuracy (%10s) : %f\n', classes{i}, accuracy);
end;
fprintf('Train Accuracy Average : %f\n', mean(accuracies)); 
%}
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    prob = Xtrain * thetas(:,i);
    AP = computeAP(prob, ytrain(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Train Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Train Accuracy Average : %0.3f%%\n', mean(accuracies)); 

%% Compute accuracy on our validation set
%{
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    p = mypredict(thetas(:,i), Xval);
    accuracy = mean(double(p == yval(:,i))) * 100;
    accuracies(i, 1) = accuracy;
    fprintf('Val Accuracy (%10s) : %f\n', classes{i}, accuracy);
end;
fprintf('Val Accuracy Average : %f\n', mean(accuracies)); 
%}
accuracies = zeros(size(classes,2),1);
for i=1:size(classes,2)
    prob = Xval * thetas(:,i);
    AP = computeAP(prob, yval(:,i), 1)*100;
    accuracies(i, 1) = AP;
    fprintf('Val Accuracy (%13s) : %0.3f%%\n', classes{i}, AP);
end;
fprintf('Val Accuracy Average : %0.3f%%\n', mean(accuracies)); 

if 0 %% DEBUG
    for i=1:size(Xval,1)
        filename = [dataset_folder, 'val/images/',...
            annotations_val{i}.annotation.filename, '.jpg'];
        img = imread(filename);
        figure(111);
        imshow(img);
        
        p_max = -inf;
        p_cls = '';
        for j=1:size(classes,2)
            p_val = Xval(i,:) * thetas(:,j);
            if(p_val > p_max)
                p_max = p_val;
                p_cls = classes{j};
            end
        end
        fprintf('we predicted %s\npause\n', p_cls);
        pause;
        close(111);
    end
end

%% predict test set
probs = [];
for i=1:size(Xtest,1),
   prob = Xtest(i,:) * thetas;
   probs(end+1,:) = prob;
end
serialize(probs, 'test');
fprintf('done');