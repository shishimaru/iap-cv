function Pr = computeCoocurrence()

globals();
featureDL_folder = '../feature-deeplearning';
[annotations_train, annotations_val, annotations_test] = loadAnnotations();
load(fullfile(devkit_folder, 'classes.mat')); % load classes

%% Load Data
Xtrain = []; Xval = []; Xtest = [];
ytrain = []; yval = [];

%  load train
fprintf('loading Xtrain...');
for i=1:40,
    filename = sprintf('%s/train_%d.mat', featureDL_folder, i);
    load(filename);
    Xtrain = [Xtrain; scores];
end
fprintf('done\n');
fprintf('loading ytrain...');
for i=1:size(annotations_train,1)
     clss = annotations_train{i}.annotation.classes;
     ytmp = [];
     for j=1:size(classes,2)
        ytmp(1,end+1) = str2double(clss.(classes{j}));
     end
     ytrain(end+1,:) = ytmp;
end
fprintf('done\n');

%  load validation
fprintf('loading Xval...');
for i=1:10,
    filename = sprintf('%s/val_%d.mat', featureDL_folder, i);
    load(filename);
    Xval = [Xval; scores];
end
fprintf('done\n');
fprintf('loading yval...');
for i=1:size(annotations_val,1)
     clss = annotations_val{i}.annotation.classes;
     ytmp = [];
     for j=1:size(classes,2)
        ytmp(1,end+1) = str2double(clss.(classes{j}));
     end
     yval(end+1,:) = ytmp;
end
fprintf('done\n');

%  load test
fprintf('loading Xtest...');
for i=1:25,
    filename = sprintf('%s/test_%d.mat', featureDL_folder, i);
    load(filename);
    Xtest = [Xtest; scores];
end
fprintf('done\n');

Pr = zeros(length(classes), length(classes));

for i = 1:length(classes)
    idx = find(ytrain(:,i) == 1);
    ytrain2 = ytrain(idx,:);
    Pr(i,:) = sum(ytrain2);
    Pr(i,:) = Pr(i,:) ./ Pr(i,i);
end
%for i = 1:length(classes)
%    Pr(:,i) = Pr(:,i) ./ sum(Pr(:,i));
%end

imagesc(Pr);
drawnow;

end

