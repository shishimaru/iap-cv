function evaluateResult(predicted_labels)
  dataset_folder = '../viscomp/';
  devkit_folder  = '../devkit/';
  cache_folder   = '../cache/';

  load(fullfile(devkit_folder, 'classes.mat'));
  load(fullfile(devkit_folder, 'filelists.mat'));
  
  fprintf('loading annotations...');
  filename = '../cache/annotations_val.mat';
  if exist(filename, 'file')
    load(filename);
  else
    annotations_val = cellfun(@(x) VOCreadxml([dataset_folder x]), val_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations_val');
  end
  fprintf('done\n');

  average_precision = zeros(length(classes), 1);

  %% DEBUG
  annotations_val = annotations_val(1:1000,:);
  
  % compute average precision, using rand to predict the confidence for each image
  for i=1:length(classes)
    ground_truth_labels = cellfun(@(x) str2double(x.annotation.classes.(classes{i})),annotations_val);
    
    % predicted_labels = rand(size(annotations_val));
    average_precision(i) = computeAP(predicted_labels(i,:)', ground_truth_labels, 1)*100;
    fprintf('class: %s, average precision: %.02f%%\n', classes{i}, average_precision(i));
  end
  fprintf('mean average precision: %.02f%%\n', mean(average_precision));
end
