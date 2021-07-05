# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:08:03 2021

@author: Mr.BeHappy
"""

import cv2
cam=cv2.VideoCapture(1)
import time
cv2.namedWindow("RRwebPics")
img_counter=0;
while True:
    ret,frame=cam.read();
    if not ret:
        print("failed to grab cam")
        break;
    cv2.imshow("capture",frame);
    k=cv2.waitKey(1);
    if(k%256==27):
        print("esc hit")
        break;
    elif True :
        imgName="C:/Users/rajar/Documents/summercoding/ML/to train/face detection/IMAGES/1/captured{}.png".format(img_counter)
        cv2.imwrite(imgName,frame);
        print("{} written!".format(imgName))
        img_counter+=1;
    time.sleep(0.1)
    
        
cam.release()
cv2.destroyAllWindows()
