function [] = createUploadFile(image_names, classes, confidence, outputname)

fid = fopen(outputname, 'w');
classes = ['image' classes];
cellfun(@(x) fprintf(fid, '%s\t', x), classes);
fprintf(fid, '\n');

for i=1:length(image_names)
    fprintf(fid, '%s\t', image_names{i});
    arrayfun(@(x) fprintf(fid, '%f\t', x), confidence(i, :));
    fprintf(fid, '\n');
end