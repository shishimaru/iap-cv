function [X] = computeGistFeatures(dataset)
globals();

%% Load annotations
filename = sprintf('../cache/annotations_%s.mat', dataset);
if exist(filename, 'file')
    load(filename);
    if(strcmp(dataset, 'train') == 1)
        annotations = annotations_train;
    elseif(strcmp(dataset, 'val') == 1)
        annotations = annotations_val;
    elseif(strcmp(dataset, 'test') == 1)
        annotations = annotations_test;
    end
else
    error('cannot load %s\n', filename)
end

c = conf();

%% Extract features
fprintf('extracting features...');
X = zeros(0,0);
for i = 1:length(annotations)
    % Read an image
    img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));
    
    % Make sure the image is color
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    % Extract descriptors
    feat = extract_feature('gist', img, c);
    X(end+1,:) = feat;
    fprintf('.');
end

fprintf('done\n');

end
