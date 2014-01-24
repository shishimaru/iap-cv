function [dictionary, sigma_inv_half, mu] = buildDictionary()
globals();

% Build a dictionary using training data
%% Load dataset
filename = fullfile('../cache/annotations_train.mat');
if exist(filename, 'file')
   load(filename);
else
    error('cannot load %s', filename);
end

%% Extract features for dictionary
annotations = annotations_train;
num_imgs = length(annotations);
if flag_debug
    num_imgs = debug_num_imgs;
end
c = conf();

fprintf('extracting features for dictionary...');
filename = fullfile(cache_folder, 'descriptors.mat');
if exist(filename, 'file')
    load(filename);
else
    descriptors = zeros(0, 0);
    for i = 1:num_imgs
        % Read an image
        img = imread(fullfile(dataset_folder, 'train', 'images', [annotations{i}.annotation.filename '.jpg']));

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
        r = randperm(size(feat, 1));
        feat_shrinked = feat(r(1:min(length(r), desc_per_img)), :);
        descriptors(end+1:end+size(feat_shrinked,1),:) = feat_shrinked;
        fprintf('.');
    end
    save(filename, 'descriptors');
end
fprintf('done\n');

%% Whiten data
if flag_white
    filename = fullfile(cache_folder, 'white.mat');
    if exist(filename, 'file')
        load(filename);
    else
        % We need mu and sigma_inv_half for whitening
        sigma = cov(descriptors);
        mu = mean(descriptors);
        sigma_inv_half = sigma ^ (-0.5);
        save(filename, 'sigma_inv_half', 'mu');
    end
    descriptors = sigma_inv_half * (descriptors' - repmat(mu', [1 size(descriptors,1)]));
    descriptors = descriptors';

else
    % When we don't use whitening, the return values of sigma_inv_half and mu will be empty
    sigma_inv_half = zeros(0,0);
    mu = zeros(0,0);
end

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

end
%End of function