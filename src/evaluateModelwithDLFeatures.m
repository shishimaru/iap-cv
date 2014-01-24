function [prob, AP] = evaluateModelwithDLFeatures(dataset, model, cls)
globals();
% Evaluate the model with validation set

%% Load annotations
filename = sprintf('../cache/annotations_%s.mat', dataset);
if exist(filename, 'file')
    load(filename);
    if(strcmp(dataset, 'val') == 1)
        annotations = annotations_val;
    elseif(strcmp(dataset, 'test') == 1)
        annotations = annotations_test;
    end
else
    error('cannot load %s\n', filename)
end


num_imgs = length(annotations);


%% Create X
fprintf('Creating X...');
Xall = zeros(0,0);
for i = 1:10
    load(sprintf('../features/%s_%d.mat', dataset, i));
    Xall(end+1:end+size(scores,1),:) = scores;
end
fprintf('done\n');


%% Create Y
fprintf('Creating Y...');
Yall = zeros(0,0);
for i = 1:num_imgs
    % Get a label based on the annotation
    if(strcmp(dataset, 'val') == 1)
        Yall(end+1,1) = str2double(annotations{i}.annotation.classes.(cls));
    else
        Yall(end+1,1) = 0;
    end
end
fprintf('done\n');


%% Calculate confidences for the validation data
[~, ~, prob] = predict(Yall, sparse(Xall), model);


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


%% computeAP
%predicted_labels = [prob';prob';prob';prob';prob';prob';prob';prob';prob';prob'];
load(fullfile(devkit_folder, 'classes.mat')); % load classes

%average_precision = zeros(length(classes), 1);
  
% compute average precision, using rand to predict the confidence for each image
%{
for i=1:length(classes)
    ground_truth_labels = cellfun(@(x) str2double(x.annotation.classes.(classes{i})),annotations_val);
    ground_truth_labels = ground_truth_labels(1:length(prob));
    
    % predicted_labels = rand(size(annotations_val));
    average_precision(i) = computeAP(predicted_labels(i,:)', ground_truth_labels, 1)*100;
    fprintf('class: %s, average precision: %.02f%%\n', classes{i}, average_precision(i));
end
fprintf('mean average precision: %.02f%%\n', mean(average_precision));
%}

AP = 0;
if(strcmp(dataset, 'val') == 1)
    ground_truth_labels = cellfun(@(x) str2double(x.annotation.classes.(cls)),annotations_val);
    ground_truth_labels = ground_truth_labels(1:length(prob));
    % predicted_labels = rand(size(annotations_val));
    AP = computeAP(prob, ground_truth_labels, 1)*100;
end

end
%End of function