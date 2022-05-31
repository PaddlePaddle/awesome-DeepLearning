function writeLabels(objects,label_dir,img_idx)

% parse input file
fid = fopen(sprintf('%s/%06d.txt',label_dir,img_idx),'w');

% for all objects do
for o = 1:numel(objects)

  % set label, truncation, occlusion
  if isfield(objects(o),'type'),         fprintf(fid,'%s ',objects(o).type);
  else                                   error('ERROR: type not specified!'), end;
  if isfield(objects(o),'truncation'),   fprintf(fid,'%.2f ',objects(o).truncation);
  else                                   fprintf(fid,'-1 '); end; % default
  if isfield(objects(o),'occlusion'),    fprintf(fid,'%.d ',objects(o).occlusion);
  else                                   fprintf(fid,'-1 '); end; % default
  if isfield(objects(o),'alpha'),        fprintf(fid,'%.2f ',wrapToPi(objects(o).alpha));
  else                                   fprintf(fid,'-10 '); end; % default

  % set 2D bounding box in 0-based C++ coordinates
  if isfield(objects(o),'x1'),           fprintf(fid,'%.2f ',objects(o).x1);
  else                                   error('ERROR: x1 not specified!'); end;
  if isfield(objects(o),'y1'),           fprintf(fid,'%.2f ',objects(o).y1);
  else                                   error('ERROR: y1 not specified!'); end;
  if isfield(objects(o),'x2'),           fprintf(fid,'%.2f ',objects(o).x2);
  else                                   error('ERROR: x2 not specified!'); end;
  if isfield(objects(o),'y2'),           fprintf(fid,'%.2f ',objects(o).y2);
  else                                   error('ERROR: y2 not specified!'); end;

  % set 3D bounding box
  if isfield(objects(o),'h'),            fprintf(fid,'%.2f ',objects(o).h);
  else                                   fprintf(fid,'-1 '); end; % default
  if isfield(objects(o),'w'),            fprintf(fid,'%.2f ',objects(o).w);
  else                                   fprintf(fid,'-1 '); end; % default
  if isfield(objects(o),'l'),            fprintf(fid,'%.2f ',objects(o).l);
  else                                   fprintf(fid,'-1 '); end; % default
  if isfield(objects(o),'t'),            fprintf(fid,'%.2f %.2f %.2f ',objects(o).t);
  else                                   fprintf(fid,'-1000 -1000 -1000 '); end; % default
  if isfield(objects(o),'ry'),           fprintf(fid,'%.2f ',wrapToPi(objects(o).ry));
  else                                   fprintf(fid,'-10 '); end; % default

  % set score
  if isfield(objects(o),'score'),        fprintf(fid,'%.2f ',objects(o).score);
  else                                   error('ERROR: score not specified!'); end;

  % next line
  fprintf(fid,'\n');
end

% close file
fclose(fid);

function alpha = wrapToPi(alpha)

% wrap to [0..2*pi]
alpha = mod(alpha,2*pi);

% wrap to [-pi..pi]
idx = alpha>pi;
alpha(idx) = alpha(idx)-2*pi;

