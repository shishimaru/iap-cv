flag_debug          = true; % if enabled, we will use small subset of data
dataset_folder = '../viscomp/';
devkit_folder  = '../devkit/';
cache_folder   = '../cache';

feature        = 'hog2x2'; % select only from hog2x2, hog3x3, sift(slow), ssim(too slow)
desc_per_img   = 10; % shouldn't change
dic_size       = 5; % should change

flag_white     = true;
debug_num_imgs = 5;