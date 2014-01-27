function [X] = computeFeatures(dictionary, dataset)
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
    
    if 1
    % crop a image with one compounded bbox
    c_xmin = inf;
    c_xmax = -inf;
    c_ymin = inf;
    c_ymax = -inf;
    for j = 1:length(annotations{i}.annotation.object)
        xmin = round(str2double(annotations{i}.annotation.object(j).bndbox.xmin));
        xmax = round(str2double(annotations{i}.annotation.object(j).bndbox.xmax));
        ymin = round(str2double(annotations{i}.annotation.object(j).bndbox.ymin));
        ymax = round(str2double(annotations{i}.annotation.object(j).bndbox.ymax));
        if(xmin < c_xmin), c_xmin = xmin; end;
        if(xmax > c_xmax), c_xmax = xmax; end;
        if(ymin < c_ymin), c_ymin = ymin; end;
        if(ymax > c_ymax), c_ymax = ymax; end;
    end;
    if 0
        figure(111);
        imshow(img);
        pause;
        close(111);
        
        figure(111);
        img = img(c_ymin:c_ymax,c_xmin:c_xmax,:);
        imshow(img);
        pause;
        close(111);
    else
        img = img(c_ymin:c_ymax,c_xmin:c_xmax,:);
    end
    end
    
    % Extract descriptors
    feat = extract_feature(featureName, img, c);
    
    % Compute a histogram
    dist = pdist2(feat, dictionary);
    [~,I] = min(dist, [], 2); % in this case, min returns the smallest element of each row
    x = hist(I, [1:size(dictionary, 2)]);
    X(end+1,:) = x;
end

% Normalize the histogram
X = X ./ repmat(sum(X, 2)+eps, [1 size(X, 2)]);

fprintf('done\n');

end
