function [corners_2D,face_idx] = computeBox3D(object,P)
% takes an object and a projection matrix (P) and projects the 3D
% bounding box into the image plane.

% index for 3D bounding box faces
face_idx = [ 1,2,6,5   % front face
             2,3,7,6   % left face
             3,4,8,7   % back face
             4,1,5,8]; % right face

% compute rotational matrix around yaw axis
R = [+cos(object.ry), 0, +sin(object.ry);
                   0, 1,               0;
     -sin(object.ry), 0, +cos(object.ry)];

% 3D bounding box dimensions
l = object.l;
w = object.w;
h = object.h;

% 3D bounding box corners
x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
y_corners = [0,0,0,0,-h,-h,-h,-h];
z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];

% rotate and translate 3D bounding box
corners_3D = R*[x_corners;y_corners;z_corners];
corners_3D(1,:) = corners_3D(1,:) + object.t(1);
corners_3D(2,:) = corners_3D(2,:) + object.t(2);
corners_3D(3,:) = corners_3D(3,:) + object.t(3);

% only draw 3D bounding box for objects in front of the camera
if any(corners_3D(3,:)<0.1) 
  corners_2D = [];
  return;
end

% project the 3D bounding box into the image plane
corners_2D = projectToImage(corners_3D, P);
