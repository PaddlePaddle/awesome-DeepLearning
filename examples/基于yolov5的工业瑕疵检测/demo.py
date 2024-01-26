from Detector import Detector
#import imutils
import cv2
import os
import time

if __name__ == '__main__':
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture('demo/a.avi')
    #cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
    print('FPS : ', fps)
    t = int(1000/fps)

    while cap.isOpened():
        ret,im = cap.read()
        if ret is False:
            break
        result, boxes = det.detect(im)
        for xyxy, conf, cls in boxes:
            x1, y1 = xyxy[0], xyxy[1]
            x2, y2 = xyxy[2], xyxy[3]
            x_center, y_center = int((x1+x2)/2), int((y1+y2)/2)
            #print("在({},{})处有{}缺陷，置信度为{:.2f}".format(x_center, y_center, cls, conf))
            
        cv2.imshow(name, result)
        if cv2.waitKey(t) & 0xFF == ord('q') :
            break
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()
