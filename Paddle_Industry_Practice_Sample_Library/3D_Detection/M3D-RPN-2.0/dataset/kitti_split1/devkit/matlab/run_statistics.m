% KITTI OBJECT DETECTION AND ORIENTATION ESTIMATION BENCHMARK STATISTICS
clear all; close all; clc;
disp('======= KITTI DevKit Statistics =======');
disp('Computing statistics of Cars, Pedestrians and Cyclists in training set.');
disp('Please wait ...');

% options
root_dir = '/media/karlsruhe_data/kitti/2012_object';
  
% get label directory and number of images
label_dir = fullfile(root_dir,'training/label_2');
nimages = length(dir(fullfile(label_dir, '*.txt')));

% init statistics
cars.occ = zeros(1,4);
peds.occ = zeros(1,4);
cycs.occ = zeros(1,4);

% compute statistics
for j=1:nimages
  objects = readLabels(label_dir,j-1);
  for k=1:length(objects)
    if strcmp(objects(k).type,'Car')
      cars.occ(objects(k).occlusion+1)  = cars.occ(objects(k).occlusion+1)  + 1;
    end
    if strcmp(objects(k).type,'Pedestrian')
      peds.occ(objects(k).occlusion+1)  = peds.occ(objects(k).occlusion+1)  + 1;
    end
    if strcmp(objects(k).type,'Cyclist')
      cycs.occ(objects(k).occlusion+1)  = cycs.occ(objects(k).occlusion+1)  + 1;
    end
  end    
end

% plot statistics
fprintf('Cars: Not occluded: %d, partly occluded: %d, largely occluded: %d, unknown: %d\n',cars.occ);
fprintf('Pedestrians: Not occluded: %d, partly occluded: %d, largely occluded: %d, unknown: %d\n',peds.occ);
fprintf('Cyclists: Not occluded: %d, partly occluded: %d, largely occluded: %d, unknown: %d\n',cycs.occ);
