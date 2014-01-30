function [box_feature] = lr_extractFeature(dataset)

%% init
globals();
dataset_folder = '../viscomp/';
cache_folder = '../cache/';
%dataset = 'train';
classNames = {'bicycle', 'bird', 'bottle', 'car', 'chair',...
    'diningtable', 'dog', 'person', 'pottedplant', 'sofa'};
           
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
load(fullfile(devkit_folder, 'classes.mat')); % load classes
bndbox_ratio = 0.1; % square parcentage in one image

if(strcmp(dataset, 'train') == 1)
    annotations = annotations_train;
elseif(strcmp(dataset, 'val') == 1)
    annotations = annotations_val;
elseif(strcmp(dataset, 'test') == 1)
    annotations = annotations_test;
end

%%
fprintf('extracting feature from %s...', dataset);

filename_f = fullfile(cache_folder,...
    sprintf('bndbox_feature_%s.mat', dataset));

if exist(filename_f, 'file')
    load(filename_f);
else
    box_feature = [];
    for i=1:length(annotations)
        filename = annotations{i}.annotation.filename;
        img = imread(fullfile(dataset_folder, dataset, 'images', [filename '.jpg']));
        img_height = size(img,1);
        img_width = size(img,2);
        
        object = annotations{i}.annotation.object;
        
        sum_box_height = 0;
        sum_box_width = 0;
        sum_box_area = 0;
        for j=1:size(object, 2)
            xmin = str2double(object(j).bndbox.xmin);
            xmax = str2double(object(j).bndbox.xmax);
            ymin = str2double(object(j).bndbox.ymin);
            ymax = str2double(object(j).bndbox.ymax);
            height = ymax - ymin;
            width  = xmax - xmin;
            
            % aspect ratio
            sum_box_height = sum_box_height + height;
            sum_box_width = sum_box_width + width;
            
            % box area
            sum_box_area = sum_box_area + width * height;

        end
        % feature : the numberf of boxes
        f_box_num = size(object,2);
        
        % feature : aspect ratio
        f_box_aspect = sum_box_height / sum_box_width;
        
        % feature : box area
        if(f_box_num == 0)
            keyboard;
        end;
        f_box_area = sum_box_area / f_box_num;
        
        % integrate into the result
        box_feature(end+1,:) = [f_box_num, f_box_aspect, f_box_area];
    end
    save(filename_f, 'box_feature');
end
fprintf('done\n');
end