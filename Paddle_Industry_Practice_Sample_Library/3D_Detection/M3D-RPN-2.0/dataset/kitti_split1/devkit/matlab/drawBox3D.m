function drawBox3D(h,object,corners,face_idx,orientation)

  % set styles for occlusion and truncation
  occ_col    = {'g','y','r','w'};
  trun_style = {'-','--'};
  trc        = double(object.truncation>0.1)+1;
  
  % draw projected 3D bounding boxes
  if ~isempty(corners)
    for f=1:4
      line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
           [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
           'parent',h(2).axes, 'color',occ_col{object.occlusion+1},...
           'LineWidth',3,'LineStyle',trun_style{trc});
      line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
           [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
           'parent',h(2).axes,'color','b','LineWidth',1);
    end
  end
  
  % draw orientation vector
  if ~isempty(orientation)
    line([orientation(1,:),orientation(1,:)]+1,...
         [orientation(2,:),orientation(2,:)]+1,...
         'parent',h(2).axes,'color','w','LineWidth',4);
    line([orientation(1,:),orientation(1,:)]+1,...
         [orientation(2,:),orientation(2,:)]+1,...
         'parent',h(2).axes,'color','k','LineWidth',2);
  end
end
