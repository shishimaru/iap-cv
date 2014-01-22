function sanitize(cls)
dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache/';
dataset        = 'train';
feature        = 'hog2x2';
desc_per_img   = 100;
dic_size       = 20;

%% Load dataset
load(fullfile(devkit_folder, 'filelists.mat'));
load(fullfile(devkit_folder, 'classes.mat'));

fprintf('loading annotations...');
filename = fullfile(cache_folder, 'annotations_train.mat');
if exist(filename, 'file')
    load(filename);
else
    annotations = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
    save(filename, 'annotations');
end
fprintf('done\n');

%% Extract features
num_train = length(annotations);
num_train = 10;
c = conf();

fprintf('extracting features...');
filename = fullfile(cache_folder, 'descriptors_train.mat');
if exist(filename, 'file')
    load(filename);
else
    descriptors = cell(num_train, 1);
    for i = 1:num_train
        % Read an image
        img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        % Crop images
        for j = 1:3
            try
            xmin = round(str2double(annotations{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations{i}.annotation.object(j).bndbox.ymax)+1);
            img_crop{j} = img(ymin:ymax,xmin:xmax,:);
            catch
                keyboard;
            end
        end

        % Extract features
        feat = zeros(0,0);
        for j = 1:3
            tmp = extract_feature(feature, img_crop{j}, c);
            feat = [feat; tmp];
        end

        % Shrink features
        r = randperm(size(feat, 1));
        descriptors{i} = feat(r(1:min(length(r), desc_per_img)), :);
        fprintf('.');
    end
    save(filename, 'descriptors');
end
all_descriptors = cell2mat(descriptors);
num_data = size(all_descriptors, 1);
fprintf('done\n');

%% Whiten data
%TODO: not implemented yet!

%% Build a dictionary
fprintf('building a dictionary...');
filename = fullfile(cache_folder, 'dictionary_train.mat');
if exist(filename, 'file')
    load(filename);
else
    dictionary = kmeansFast(descriptors, dic_size);
    save(filename, 'dictionary');
end
fprintf('done\n');

% Calculate features for SVM training
Xall = zeros(0,0);
Yall = zeros(0,0);
for i = 1:num_train
    Yall(end+1) = str2double(annotations{i}.annotation.classes.(cls));
    
    desc = descriptors{i};
    num_desc = size(desc, 1);
    x = zeros(1, dic_size);
    for j = 1:num_desc
        % Find the closest centroid
        closest_k = 0;
        min_dist = inf;
        for k = 1:dic_size
            dist = dictionary(k,:) - desc(j,:);
            dist = sum(dist .^ 2);
            if dist < min_dist
                min_dist = dist;
                closest_k = k;
            end
        end
        x(closest_k) = x(closest_k) + 1;
    end
    x = x ./ (sum(x) + eps); % normalize
    Xall(end+1,:) = x;
end

% Train a model
svm_options = '-s 2 -B 1 -c 1 -q';
model = train(Yall', sparse(Xall), svm_options);

% Calculate confidences
for i = 1:size(Xall,1);
    confidence = [Xall(i,:), 1] * model.w';
    fprintf('conf = %f, label = %d\n', confidence, Yall(i));
end


end
