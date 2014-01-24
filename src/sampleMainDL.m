globals();

% setup classes
devkit_folder  = '../devkit/';
load(fullfile(devkit_folder, 'classes.mat')); % load classes

svm_c = [1];

for i = 1:length(svm_c)
    [model] = trainSVMwithDLFeatures('car', svm_c(i));
    [prob, AP] = evaluateModelwithDLFeatures('val', model, 'airplane'); % Evaluate model with val
    fprintf('========================= c = %f, AP = %f =====\n',svm_c(i), AP);
end


if 0
probs_val = [];
for i=1:length(classes)
    fprintf('CLASS: %s\n', classes{i});
    
    % create a model
    [model] = trainSVMwithDLFeatures(classes{i}, svm_c);
    models{i} = model;

    % compute prob, AP for val
    [prob, AP] = evaluateModelwithDLFeatures('val', model, classes{i}); % Evaluate model with val
    fprintf('AP: %f%%\n', AP);
    probs_val(:, end+1) = prob;
    
    % TODO: compute prob for test
end

%TODO: We need to predict the test set here
serialize(probs_val, 'val');

end