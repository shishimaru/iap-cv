function [image_names, classes, confidence] = readUploadFile(filename)
% reading back the upload file

d = textread(filename, '%s');

tmp = load('classes.mat');
d = reshape(d, length(tmp.classes)+1, [])';
classes = d(1, 2:end);
image_names = d(2:end, 1);
confidence = str2double(d(2:end, 2:end));