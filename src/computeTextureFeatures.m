function [X] = computeTextureFeatures(dataset)
globals();

%% Load annotations
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
if(strcmp(dataset, 'train') == 1)
    annotations = annotations_train;
elseif(strcmp(dataset, 'val') == 1)
    annotations = annotations_val;
elseif(strcmp(dataset, 'test') == 1)
    annotations = annotations_test;
end

%% Compute features
X = zeros(0,0);
for i = 1:length(annotations)
    % Read an image
    img = imread(fullfile(dataset_folder, dataset, 'images', [annotations{i}.annotation.filename '.jpg']));
    
    % Make sure the image is grayscale
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    if 0
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
    
    % Extract texture features
    GLCM2 = graycomatrix(img, 'Offset', [2 0;0 2]);
    stats = GLCM_Features1(GLCM2,0);
    feat = [stats.autoc(:); stats.contr(:); stats.corrm(:); stats.corrp(:); stats.cprom(:); stats.cshad(:); stats.dissi(:); stats.energ(:); stats.entro(:); stats.homom(:); stats.homop(:); stats.maxpr(:); stats.sosvh(:); stats.savgh(:); stats.svarh(:); stats.senth(:); stats.dvarh(:); stats.denth(:); stats.inf1h(:); stats.inf2h(:); stats.indnc(:); stats.idmnc(:)];
    feat = feat';
    X(end+1,:) = feat;
end

end
%End of function