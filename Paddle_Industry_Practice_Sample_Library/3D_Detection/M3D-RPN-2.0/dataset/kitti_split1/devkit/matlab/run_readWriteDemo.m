% clear and close everything
clear all; close all;
disp('======= KITTI DevKit Demo =======');

root_dir  = '/media/karlsruhe_data/kitti/2012_object';
train_dir = fullfile(root_dir,'/training/label_2');
test_dir  = '.'; % location of your testing dir

% read objects of first training image
train_objects = readLabels(train_dir,0);

% loop over all images
% ... YOUR TRAINING CODE HERE ...
% ... YOUR TESTING CODE HERE ...

% detect one object (car) in first test image
test_objects(1).type  = 'Car';
test_objects(1).x1    = 10;
test_objects(1).y1    = 10;
test_objects(1).x2    = 100;
test_objects(1).y2    = 100;
test_objects(1).alpha = pi/2;
test_objects(1).score = 0.5;

% write object to file
writeLabels(test_objects,test_dir,0);
disp('Test label file written!');
