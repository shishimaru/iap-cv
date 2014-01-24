DO_VALIDATION_TEST = 1;

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

%% Create Models
fprintf('Create models\n');
for i=1:length(classes)
    fprintf('CLASS: %s\n', classes{i});
    
    % create a model
    [model] = trainSVM(classes{i}, dictionary, sigma_inv_half, mu);
    models{i} = model;    
end
fprintf('Done models creation\n');

%% Validation test
if DO_VALIDATION_TEST
    fprintf('Start validation test\n');
    probs_val = [];
    for i=1:length(classes)
        fprintf('CLASS: %s\n', classes{i});
        % compute prob, AP for val
        [prob, AP] = evaluateModel('val', models{i}, classes{i}, dictionary, sigma_inv_half, mu); % Evaluate model with val
        fprintf('AP: %f%%\n', AP);
        probs_val(:, end+1) = prob;
    end
    serialize(probs_val, 'val');
    fprintf('Done validation test\n');
    pause;
end

%% Test
fprintf('Start test\n');
probs_test = [];
for i=1:length(classes)
    fprintf('CLASS: %s\n', classes{i});
    % compute prob, AP for val
    [prob, AP] = evaluateModel('test', models{i}, classes{i}, dictionary, sigma_inv_half, mu); % Evaluate model with val
    fprintf('AP: %f%%\n', AP);
    probs_test(:, end+1) = prob;
end
serialize(probs_test, 'test');

