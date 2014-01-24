function [model] = trainSVMwithDLFeatures(cls, svm_c)

globals();

%% Load annotations
filename = fullfile('../cache/annotations_train.mat');
if exist(filename, 'file')
    load(filename);
    annotations = annotations_train;
else
    error('cannot load %s\n', filename)
end
num_imgs = length(annotations);

% setup classes
devkit_folder  = '../devkit/';
load(fullfile(devkit_folder, 'classes.mat')); % load classes

%% Create X
fprintf('Creating X...');
Xall = zeros(0,0);
for i = 1:40
    load(sprintf('../features/train_%d.mat', i));
    Xall(end+1:end+size(scores,1),:) = scores;
end
fprintf('done\n');


%% Create Y
fprintf('Creating Y...');
Yall = zeros(0,0);
for i = 1:num_imgs
    % Get a label based on the annotation
    Yall(end+1,1) = str2double(annotations{i}.annotation.classes.(cls));
end
fprintf('done\n');

%% Train a model
fprintf('Train %s model...', cls);
filename = fullfile(cache_folder, sprintf('model_%s.mat', cls));
if exist(filename, 'file')
    load(filename);
else
    cls_idx = 0;
    for i = 1:length(classes)
        if strcmp(classes{i}, cls) == 1
            break;
        end
    end
    opt = sprintf('-s 2 -B 1 -c %f -q', svm_c);
    model = train(Yall, sparse(Xall), opt);
    save(filename, 'model');
end

% Calculate confidences for the training data
[predicted_label, accuracy, prob] = predict(Yall, sparse(Xall), model);

%% Show classification scores
if 1
    clf;
    prob_pos = prob(find(Yall==1));
    prob_neg = prob(find(Yall==0));
    [~,I] = sort(prob_pos, 'descend');
    prob_pos = prob_pos(I);
    [~,I] = sort(prob_neg, 'descend');
    prob_neg = prob_neg(I);
    prob_all = [prob_pos;prob_neg];
    plot(prob_all, 'b.');
    hold on;
    plot(prob_pos, 'r.');
    ylabel('scores');
    xlabel('samples');
    drawnow;
end

end

