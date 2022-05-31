function objects = readLabels(label_dir,img_idx)

% parse input file
fid = fopen(sprintf('%s/%06d.txt',label_dir,img_idx),'r');
C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
fclose(fid);

% for all objects do
objects = [];
for o = 1:numel(C{1})

  % extract label, truncation, occlusion
  lbl = C{1}(o);                   % for converting: cell -> string
  objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
  objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
  objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
  objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])

  % extract 2D bounding box in 0-based coordinates
  objects(o).x1 = C{5}(o); % left
  objects(o).y1 = C{6}(o); % top
  objects(o).x2 = C{7}(o); % right
  objects(o).y2 = C{8}(o); % bottom

  % extract 3D bounding box information
  objects(o).h    = C{9} (o); % box width
  objects(o).w    = C{10}(o); % box height
  objects(o).l    = C{11}(o); % box length
  objects(o).t(1) = C{12}(o); % location (x)
  objects(o).t(2) = C{13}(o); % location (y)
  objects(o).t(3) = C{14}(o); % location (z)
  objects(o).ry   = C{15}(o); % yaw angle
end
