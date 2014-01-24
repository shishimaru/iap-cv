function [prob, AP] = evaluateModel(dataset, model, cls, dictionary, sigma_inv_half, mu)
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


if flag_debug
    num_imgs = debug_num_imgs;
else
    num_imgs = length(annotations);
end

c = conf();

%% Extract features for SVM validation
fprintf('extracting features for SVM validation...');
filename = fullfile(cache_folder, sprintf('x_%s_%s.mat', dataset, cls));
if exist(filename, 'file')
    load(filename);
else
    Xall = zeros(0,0);
    for i = 1:num_imgs
        % Read an image
        img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));

        % Make sure the image is color
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        
        % crop a image with one compounded bbox
        c_xmin = inf;
        c_xmax = -inf;
        c_ymin = inf;
        c_ymax = -inf;
        for j = 1:length(annotations{i}.annotation.object)
            xmin = round(str2double(annotations{i}.annotation.object(j).bndbox.xmin)+1);
            xmax = round(str2double(annotations{i}.annotation.object(j).bndbox.xmax)+1);
            ymin = round(str2double(annotations{i}.annotation.object(j).bndbox.ymin)+1);
            ymax = round(str2double(annotations{i}.annotation.object(j).bndbox.ymax)+1);
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


        % Extract descriptors
        feat = extract_feature(featureName, img, c);

		% Whiten descriptors
        if flag_white
          feat = sigma_inv_half * (feat' - repmat(mu', [1 size(feat,1)]));
          feat = feat';
        end

        x = zeros(1, size(dictionary, 1));
        for k = 1:size(feat, 1)
          % Find the closest centroid
          closest_l = 0;
          min_dist = inf;
          for l = 1:size(dictionary, 1)
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
    end

    save(filename, 'Xall');
end
fprintf('done\n');


%% Creating Y
fprintf('extracting features for SVM validation...');
filename = fullfile(cache_folder, sprintf('y_%s_%s.mat', dataset, cls));
if exist(filename, 'file')
    load(filename);
else
    Yall = zeros(0,0);
    for i = 1:num_imgs
        % Get a label based on the annotation
        if(strcmp(dataset, 'val') == 1)
            Yall(end+1,1) = str2double(annotations{i}.annotation.classes.(cls));
        else
            Yall(end+1,1) = 0;
        end
    end
    save(filename, 'Yall');
end
fprintf('done\n');

%% Calculate confidences for the validation data
[~, ~, prob] = predict(Yall, sparse(Xall), model);


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
    drawnow;
end


%% computeAP
predicted_labels = [prob';prob';prob';prob';prob';prob';prob';prob';prob';prob'];
load(fullfile(devkit_folder, 'classes.mat')); % load classes

average_precision = zeros(length(classes), 1);
  
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