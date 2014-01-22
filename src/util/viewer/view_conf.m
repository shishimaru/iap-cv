function view_conf(imgFile, bboxes);
%% show image with bounding boxes whose lines are changed on each confidence.
%% imgFile : filename of image
%% bboxes  : matrix of bbox with confidence
%%           bbox = [xmin xmax ymin ymax conf;
%%                   ...
%%                  ];

% range of color sheet
NC = 200;

figure;
hold on;
imshow(imgFile);
title(sprintf('FILE: %s', imgFile));

% draw bboxes
for i=1:size(bboxes,1),
  bbox = bboxes(i,:);
  xmin = bbox(1,1);
  xmax = bbox(1,2);
  ymin = bbox(1,3);
  ymax = bbox(1,4);
  conf = bbox(1,5);

  % set color based on confidence
  colorsheet = jet(NC);
  color_idx = round(conf * (NC-1) + 1);
  color = colorsheet(color_idx, :);

  % draw one bbox
  rectangle('Position',[xmin, ymin, xmax - xmin, ymax - ymin],...
            'LineWidth', 3, 'EdgeColor', color);
end;

hold off;
end
