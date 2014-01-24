globals();
[annotations_train, annotations_val, annotations_test] = loadAnnotations();

[dictionary, sigma_inv_half, mu] = buildDictionary();

% setup classes
devkit_folder  = '../devkit/';
load(fullfile(devkit_folder, 'classes.mat')); % load classes

% DEBUG: reduce the number of classes
% if(flag_debug)
if 0
    classes = {'car', 'airplane'}; % target only one class 'car'
end

probs_val = [];
for i=1:length(classes)
    fprintf('CLASS: %s\n', classes{i});
    
    % create a model
    [model] = trainSVM(classes{i}, dictionary, sigma_inv_half, mu);
    models{i} = model;    

    % compute prob, AP for val
    [prob, AP] = evaluateModel('val', model, classes{i}, dictionary, sigma_inv_half, mu); % Evaluate model with val
    fprintf('AP: %f%%\n', AP);
    probs_val(:, end+1) = prob;
    
    % TODO: compute prob for test
end


%TODO: We need to predict the test set here
serialize(probs_val, 'val');