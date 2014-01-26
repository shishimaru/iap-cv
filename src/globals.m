flag_debug          = false; % if enabled, we will use small subset of data
dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache';

featureName        = 'hog2x2'; % select only from hog2x2, hog3x3, sift(slow), ssim(too slow)
desc_per_img   = 100; % shouldn't change
dic_size       = 30; % should change

flag_white     = false;
debug_num_imgs = 100;

svm_options{1}  = '-s 2 -B 1 -c 0.06 -q'; % airplane
svm_options{2}  = '-s 2 -B 1 -c 0.139 -q'; % bicycle
svm_options{3}  = '-s 2 -B 1 -c 0.064 -q'; % car
svm_options{4}  = '-s 2 -B 1 -c 0.071 -q'; % cup_or_mug
svm_options{5}  = '-s 2 -B 1 -c 0.098 -q'; % dog
svm_options{6}  = '-s 2 -B 1 -c 0.064 -q'; % guitar
svm_options{7}  = '-s 2 -B 1 -c 0.128 -q'; % hamburger
svm_options{8}  = '-s 2 -B 1 -c 0.173 -q'; % sofa
svm_options{9}  = '-s 2 -B 1 -c 0.067 -q'; % traffic_light
svm_options{10} = '-s 2 -B 1 -c 0.060 -q'; % person
