function [orientation_2D] = computeOrientation3D(object,P)
% takes an object and a projection matrix (P) and projects the 3D
% object orientation vector into the image plane.

% compute rotational matrix around yaw axis
R = [cos(object.ry),  0, sin(object.ry);
     0,               1,              0;
     -sin(object.ry), 0, cos(object.ry)];

% orientation in object coordinate system
orientation_3D = [0.0, object.l
                  0.0, 0.0
                  0.0, 0.0];

% rotate and translate in camera coordinate system, project in image
orientation_3D      = R*orientation_3D;
orientation_3D(1,:) = orientation_3D(1,:) + object.t(1);
orientation_3D(2,:) = orientation_3D(2,:) + object.t(2);
orientation_3D(3,:) = orientation_3D(3,:) + object.t(3);

% vector behind image plane?
if any(orientation_3D(3,:)<0.1)
  orientation_2D = [];
  return;
end

% project orientation into the image plane
orientation_2D = projectToImage(orientation_3D,P);
