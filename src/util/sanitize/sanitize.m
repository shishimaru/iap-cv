function [model] = sanitize(cls)

dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache/bbox/';
dataset_train  = 'train';
dataset_val    = 'val';
dataset_test   = 'test';
feature        = 'gist';%'hog2x2';
desc_per_img   = 100;
dic_size       = 1000;

hogesize = 300;

%% Load dataset
load(fullfile(devkit_folder, 'filelists.mat'));
load(fullfile(devkit_folder, 'classes.mat'));

fprintf('loading annotations...');
filename = fullfile('../cache/', 'annotations_train.mat');
if exist(filename, 'file')
   load(filename);
else
   annotations_train = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
   save(filename, 'annotations_train');
end

filename = fullfile('../cache/', 'annotations_val.mat');
if exist(filename, 'file')
   load(filename);
else
   annotations_val = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), val_data.annotations', 'UniformOutput', false);
   save(filename, 'annotations_val');
end

filename = fullfile('../cache/', 'annotations_test.mat');
if exist(filename, 'file')
   load(filename);
else
   annotations_test = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), test_data.annotations', 'UniformOutput', false);
   save(filename, 'annotations_test');
end
fprintf('done\n');


%% Find class index
for cls_idx = 1:10
    if strcmp(classes(cls_idx), cls) == 1
        break;
    end
end


%% Extract descriptors
c = conf();

fprintf('extracting descriptors...');
filename_desc = fullfile(cache_folder, 'descriptors.mat');
if exist(filename_desc, 'file')
    load(filename_desc);
else
    descriptors = zeros(0, 0);
    for i = 1:min(hogesize, length(annotations_train))
        % Read an image
        filename = fullfile(dataset_folder, dataset_train, 'images', [annotations_train{i}.annotation.filename '.jpg']);
        img = imread(filename);

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        for j = 1:length(annotations_train{i}.annotation.object)
            % Crop images
            xmin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymax)+1);
            img_crop = img(ymin:ymax,xmin:xmax,:);

            % Extract descriptors
            feat = extract_feature(feature, img_crop, c);
            r = randperm(size(feat, 1));
            feat_shrinked = feat(r(1:min(length(r), desc_per_img)), :);
            descriptors(end+1:end+size(feat_shrinked,1),:) = feat_shrinked;
        end

        fprintf('.');
    end
    save(filename_desc, 'descriptors');
end
fprintf('done\n');

%% Whiten data
%TODO: not implemented yet!

%% Build a dictionary
fprintf('building a dictionary...');
filename = fullfile(cache_folder, 'dictionary.mat');
if exist(filename, 'file')
    load(filename);
else
    dictionary = kmeansFast(descriptors, dic_size);
    save(filename, 'dictionary');
end
fprintf('done\n');

%% Extract features of train and val
fprintf('extracting features of all datasets...');
filename = fullfile(cache_folder, 'mydata.mat');
if exist(filename, 'file')
    load(filename);
else
    % for train set
    for i = 1:min(hogesize, length(annotations_train))
        mydata_train{i}.filename = fullfile(dataset_folder, dataset_train, 'images', [annotations_train{i}.annotation.filename '.jpg']);

        % Read an image
        img = imread(mydata_train{i}.filename);

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        
        mydata_train{i}.num_box = length(annotations_train{i}.annotation.object);
        for j = 1:mydata_train{i}.num_box
            % Crop images
            xmin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymax)+1);
            img_crop = img(ymin:ymax,xmin:xmax,:);

            % Extract descriptors
            feat = extract_feature(feature, img_crop, c);
            x = zeros(1, dic_size);
            if 0
            for k = 1:size(feat, 1)
                % Find the closest centroid
                closest_l = 0;
                min_dist = inf;
                for l = 1:dic_size
                    dist = dictionary(l,:) - feat(k,:);
                    dist = sum(dist .^ 2);
                    if dist < min_dist
                        min_dist = dist;
                        closest_l = l;
                    end
                end
                x(closest_l) = x(closest_l) + 1;
            end
            x = x ./ (sum(x) + eps); % normalize
            end
            
            dist = dictionary - repmat(feat, [size(dictionary, 1), 1]);
            dist = dist .^ 2;
            dist = sum(dist, 2);
            [~, idx] = min(dist);
            x(idx) = x(idx) + 1;
 
            mydata_train{i}.bbox{j} = [xmin xmax ymin ymax];
            mydata_train{i}.feat{j} = x;
            labels = zeros(10,1);
            labels(1) = str2double(annotations_train{i}.annotation.classes.airplane);
            labels(2) = str2double(annotations_train{i}.annotation.classes.bicycle);
            labels(3) = str2double(annotations_train{i}.annotation.classes.car);
            labels(4) = str2double(annotations_train{i}.annotation.classes.cup_or_mug);
            labels(5) = str2double(annotations_train{i}.annotation.classes.dog);
            labels(6) = str2double(annotations_train{i}.annotation.classes.guitar);
            labels(7) = str2double(annotations_train{i}.annotation.classes.hamburger);
            labels(8) = str2double(annotations_train{i}.annotation.classes.sofa);
            labels(9) = str2double(annotations_train{i}.annotation.classes.traffic_light);
            labels(10) = str2double(annotations_train{i}.annotation.classes.person);
            mydata_train{i}.labels{j} = labels;
        end
    end

    % for val set
    for i = 1:min(hogesize, length(annotations_val))
        mydata_val{i}.filename = fullfile(dataset_folder, dataset_val, 'images', [annotations_val{i}.annotation.filename '.jpg']);

        % Read an image
        img = imread(mydata_val{i}.filename);

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        
        mydata_val{i}.num_box = length(annotations_val{i}.annotation.object);
        for j = 1:mydata_val{i}.num_box
            % Crop images
            xmin = round(str2double(annotations_val{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations_val{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations_val{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations_val{i}.annotation.object(j).bndbox.ymax)+1);
            img_crop = img(ymin:ymax,xmin:xmax,:);

            % Extract descriptors
            feat = extract_feature(feature, img_crop, c);
            x = zeros(1, dic_size);
            for k = 1:size(feat, 1)
                % Find the closest centroid
                closest_l = 0;
                min_dist = inf;
                for l = 1:dic_size
                    dist = dictionary(l,:) - feat(k,:);
                    dist = sum(dist .^ 2);
                    if dist < min_dist
                        min_dist = dist;
                        closest_l = l;
                    end
                end
                x(closest_l) = x(closest_l) + 1;
            end
            x = x ./ (sum(x) + eps); % normalize
 
            mydata_val{i}.bbox{j} = [xmin xmax ymin ymax];
            mydata_val{i}.feat{j} = x;
            labels = zeros(10,1);
            labels(1) = str2double(annotations_train{i}.annotation.classes.airplane);
            labels(2) = str2double(annotations_train{i}.annotation.classes.bicycle);
            labels(3) = str2double(annotations_train{i}.annotation.classes.car);
            labels(4) = str2double(annotations_train{i}.annotation.classes.cup_or_mug);
            labels(5) = str2double(annotations_train{i}.annotation.classes.dog);
            labels(6) = str2double(annotations_train{i}.annotation.classes.guitar);
            labels(7) = str2double(annotations_train{i}.annotation.classes.hamburger);
            labels(8) = str2double(annotations_train{i}.annotation.classes.sofa);
            labels(9) = str2double(annotations_train{i}.annotation.classes.traffic_light);
            labels(10) = str2double(annotations_train{i}.annotation.classes.person);
            mydata_val{i}.labels{j} = labels;
        end
    end
    
    save(filename, 'mydata_train', 'mydata_val');
end
fprintf('done\n');



for iii = 1:10
    
fprintf('===== Iter: %d =====\n', iii);

%% Build training data
fprintf('building training data...');
Xall = zeros(0,0);
Yall = zeros(0,0);
for i = 1:length(mydata_train)
    for j = 1:mydata_train{i}.num_box
        Xall(end+1,:) = mydata_train{i}.feat{j};
        Yall(end+1,1) = mydata_train{i}.labels{j}(cls_idx);
    end
end
fprintf('done\n');




if 0
%% Extract features for SVM training
fprintf('extracting features for SVM training...');
filename = fullfile(cache_folder, 'training_data.mat');
if exist(filename, 'file')
    load(filename);
else
    Xall = zeros(0,0);
    Yall = zeros(0,0);
    for i = 1:num_imgs
        % Read an image
        img = imread(fullfile(dataset_folder, dataset_train, 'images', [annotations_train{i}.annotation.filename '.jpg']));

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        for j = 1:length(annotations_train{i}.annotation.object)
            % Crop images
            xmin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations_train{i}.annotation.object(j).bndbox.ymax)+1);
            img_crop = img(ymin:ymax,xmin:xmax,:);

            % Extract descriptors
            feat = extract_feature(feature, img_crop, c);
            x = zeros(1, dic_size);
            for k = 1:size(feat, 1)
                % Find the closest centroid
                closest_l = 0;
                min_dist = inf;
                for l = 1:dic_size
                    dist = dictionary(l,:) - feat(k,:);
                    dist = sum(dist .^ 2);
                    if dist < min_dist
                        min_dist = dist;
                        closest_l = l;
                    end
                end
                x(closest_l) = x(closest_l) + 1;
            end
            x = x ./ (sum(x) + eps); % normalize
            Xall(end+1,:) = x;

            % Get a label based on the annotation
            Yall(end+1,1) = str2double(annotations_train{i}.annotation.classes.(cls));
        end
    end
    save(filename, 'Xall', 'Yall');
end
fprintf('done\n');
end

%% Train a model
svm_options = '-s 2 -B 1 -c 1 -q';
model = train(Yall, sparse(Xall), svm_options);
models{iii} = model;

% Calculate confidences for the training data
[predicted_label, accuracy, prob] = predict(Yall, sparse(Xall), model);
min_prob = min(prob);
max_prob = max(prob);

%% Show classification scores
if 0
    idx_pos = find(Yall==1);
    idx_neg = find(Yall==0);
    prob_pos = prob(idx_pos);
    prob_neg = prob(idx_neg);
    [~,I_pos] = sort(prob_pos, 'descend');
    prob_pos = prob_pos(I_pos);
    idx_pos = idx_pos(I_pos);
    [~,I_neg] = sort(prob_neg, 'descend');
    prob_neg = prob_neg(I_neg);
    prob_all = [prob_pos;prob_neg];
    plot(prob_all, 'b.');
    hold on;
    plot(prob_pos, 'r.');
    ylabel('scores');
    xlabel('samples');
end

%% Show each classification
if 0
    figure;
    for i = 1:min(100,length(idx_pos))
        subplot(10,10,i);
        img = imread(instance_list{idx_pos(i)}.filename);
        bbox = instance_list{idx_pos(i)}.bbox;
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        img_crop = img(bbox(3):bbox(4), bbox(1):bbox(2),:);
        imshow(img_crop);
        title(sprintf('%f', prob_pos(i)));
    end
    figure;

    for i = 1:min(100,length(idx_pos))
        subplot(10,10,i);
        img = imread(instance_list{idx_pos(length(idx_pos)-i+1)}.filename);
        bbox = instance_list{idx_pos(length(idx_pos)-i+1)}.bbox;
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        img_crop = img(bbox(3):bbox(4), bbox(1):bbox(2),:);
        imshow(img_crop);
        title(sprintf('%f', prob_pos(i)));
    end
end

%% Evaluate the classifier with validation set
fprintf('evaluating the classifier with validation set...');
Xall = zeros(0,0);
Yall = zeros(0,0);
predicted_labels = zeros(length(mydata_val), 1);
gt_labels = zeros(length(mydata_val), 1);
for i = 1:length(mydata_val)
    prob_max = -inf;
    for j = 1:mydata_val{i}.num_box
        x = mydata_val{i}.feat{j};
        Yall(end+1,1) = mydata_val{i}.labels{j}(cls_idx);
        [~, ~, prob] = predict([1], sparse(x), model);
        if prob > prob_max
            prob_max = prob;
        end
    end
    predicted_labels(i,1) = prob_max;
    gt_labels(i,1) = mydata_val{i}.labels{1}(cls_idx);
end
AP = computeAP(predicted_labels, gt_labels, 1) * 100;
fprintf('done\n');
fprintf('AP(%d) = %f\n', iii, AP);

%% Sanitize the train set
num_pos = 0;
num_neg = 0;
for i = 1:length(mydata_train)
    for j = 1:mydata_train{i}.num_box
        x = mydata_train{i}.feat{j};
        [~, ~, prob] = predict([1], sparse(x), model, '-q'); % [1] is a dummy
        
        %FIXME: prob is need to be normalized
        prob = (prob - min_prob) / (max_prob - min_prob);
        if prob > 0.09;
            mydata_train{i}.labels{j}(cls_idx) = 1;
            num_pos = num_pos + 1;
        else
            mydata_train{i}.labels{j}(cls_idx) = 0;
            num_neg = num_neg + 1;
        end

    end
end
fprintf('#pos = %d, #neg = %d\n', num_pos, num_neg);

pause;

end
%End of iii

filename = fullfile(cache_folder, 'models.mat');
save(filename, 'models');

end

