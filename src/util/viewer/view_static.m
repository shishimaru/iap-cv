function view_static(dirname, imgNumber);
imgFile = fullfile(dirname, 'images', sprintf('%s.jpg', imgNumber));
annFile = fullfile(dirname, 'annotations', sprintf('%s.xml', imgNumber));

annotation = VOCreadxml(annFile);
classes = {'airplane','bicycle','car','cup_or_mug','dog','guitar',...
           'hamburger','sofa','traffic_light','person'};
class = ''
for i=1:length(classes),
  name = classes{i};
  flag = str2num(annotation.annotation.classes.(name));
  if(flag == 1),
    class = name;
  end;
end;


figure;
hold on;
imshow(imgFile);
fprintf('CLASS=%s\n', class);
title(sprintf('FILE: %s, CLASS: %s', imgFile,class));

% draw rectangle
for i=1:size(annotation.annotation.object,2),
    obj =  annotation.annotation.object(i);
    xmin = str2num(obj.bndbox.xmin);
    xmax = str2num(obj.bndbox.xmax);
    ymin = str2num(obj.bndbox.ymin);
    ymax = str2num(obj.bndbox.ymax);
    rectangle('Position',[xmin ymin xmax-xmin ymax-ymin], 'LineWidth',2, 'EdgeColor','b');
end;
hold off;

end
