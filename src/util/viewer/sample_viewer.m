function sample_viewer()
%{
view_static('../viscomp/train', '00001110');
pause;
view_static('../viscomp/train', '00001111');
pause;
view_static('../viscomp/train', '00001112');
pause;
%}
view_conf('../viscomp/train/images/00001110.jpg',...
          [100 300 100 200 0.1;...
           200 400 200 300 0.5;...
           300 500 300 400 1.0;]);
end
