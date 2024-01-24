function P = readCalibration(calib_dir,img_idx,cam)

  % load 3x4 projection matrix
  P = dlmread(sprintf('%s/%06d.txt',calib_dir,img_idx),' ',0,1);
  P = P(cam+1,:);
  P = reshape(P ,[4,3])';
  
end
