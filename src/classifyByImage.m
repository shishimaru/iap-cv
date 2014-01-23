function [model] = classifyByImage(cls)

dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache/image';
dataset        = 'train';
% feature        = 'hog2x2';
feature        = 'gist'; % 20:X
desc_per_img   = 100; % shouldn't change
dic_size       = 1000; % should change

%% Load dataset
load(fullfile(devkit_folder, 'filelists.mat'));
load(fullfile(devkit_folder, 'classes.mat'));

fprintf('loading annotations...');
filename = '../cache/annotations_train.mat';
if exist(filename, 'file')
   load(filename);
else
   annotations_train = cellfun(@(x) VOCreadxml([dataset_folder '/' x]), train_data.annotations', 'UniformOutput', false);
   save(filename, 'annotations_train');
end
fprintf('done\n');

%% Extract features for dictionary
annotations = annotations_train;
num_imgs = length(annotations);
num_imgs = 1000;
c = conf();

fprintf('extracting features for dictionary...');
filename = fullfile(cache_folder, 'descriptors.mat');
if exist(filename, 'file')
    load(filename);
else
    descriptors = zeros(0, 0);
    for i = 1:num_imgs
        % Read an image
        img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        % Extract descriptors
        feat = extract_feature(feature, img, c);
        r = randperm(size(feat, 1));
        feat_shrinked = feat(r(1:min(length(r), desc_per_img)), :);
        descriptors(end+1:end+size(feat_shrinked,1),:) = feat_shrinked;
        fprintf('.');
    end
    save(filename, 'descriptors');
end
fprintf('done\n');

%% Whiten data
%TODO: not implemented yet!
?? sigma = cov(descriptors);
?? mu = mean(descriptors);
?? descriptors = sigma .^ (-0.5) * (descriptors - repmat(mu, size(descriptors,1), 1));


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
        img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        % Extract descriptors
        feat = extract_feature(feature, img, c);
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
        Yall(end+1,1) = str2double(annotations{i}.annotation.classes.(cls));
    end
    ?? Xall = (Xall - mu) / (sigma ^ 0.5);
    save(filename, 'Xall', 'Yall');
end
fprintf('done\n');

%% Train a model
svm_options = '-s 2 -B 1 -c 1 -q';
model = train(Yall, sparse(Xall), svm_options);

% Calculate confidences for the training data
svm_options = '-b 1';
[predicted_label, accuracy, prob] = predict(Yall, sparse(Xall), model, svm_options);

%% Show classification scores
if 1
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
end

%{
%% Show images with confidence order
[~,I] = sort(prob, 'descend');
prob_s = prob(I);
annotations_s = annotations(I);

for i = 1:num_imgs
  % Read an image
  filename = fullfile(dataset_folder, dataset, 'images', [annotations_s{i}.annotation.filename '.jpg'])
  img = imread(filename);
  
  % Make sure the image is color
  if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
  end

  figure(i);
  hold on;
  imshow(img);
  title(sprintf('FILE: %s, CONF: %f', filename, prob_s(i)));
  hold off
  pause;
  close(i);
end
%}

%% computeAP
evaluateResult([prob';prob';prob';prob';prob';prob';prob';prob';prob';prob';]);

end

