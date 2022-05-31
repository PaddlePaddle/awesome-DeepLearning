function pts_2D = projectToImage(pts_3D, P)
% PROJECTTOIMAGE projects 3D points in given coordinate system in the image
% plane using the given projection matrix P.
%
% Usage: pts_2D = projectToImage(pts_3D, P)
%   input: pts_3D: 3xn matrix
%          P:      3x4 projection matrix
%   output: pts_2D: 2xn matrix
%
% last edited on: 2012-02-27
% Philip Lenz - lenz@kit.edu


  % project in image
  pts_2D = P * [pts_3D; ones(1,size(pts_3D,2))];
  % scale projected points
  pts_2D(1,:) = pts_2D(1,:)./pts_2D(3,:);
  pts_2D(2,:) = pts_2D(2,:)./pts_2D(3,:);
  pts_2D(3,:) = [];
end