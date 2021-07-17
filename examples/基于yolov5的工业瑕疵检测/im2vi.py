import os,sys
import cv2

path='demo_images'
pic=os.listdir(path)
pic.sort()

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter("a.avi", fourcc, 15, (512, 512))
for p in pic:
  im=cv2.imread(path+'/'+p)
  cv2.imshow('demo',im)
  out.write(im)
out.release()
