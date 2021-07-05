# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:47:31 2021

@author: Mr.BeHappy
"""


import cv2;
import os
import numpy as np
import time
from  face_recognition import *
def imshow(img):
    import IPython
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
   
#print("face detected",face_detected)
#####comment these lines
#faces,faceID=labels_for_training_images('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/IMAGES/')
#face_recognizer=train_classifier(faces,faceID)
#face_recognizer.write('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/trainingData.yml')

#uncomment these lines while running from  2nd time onwards
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/trainingData.yml')
print("starting camera")
name={1:'Mr.BeHappy',2:'babi',3:"Raji Reddy"}#creating dictionary containing names

cvv=cv2.VideoCapture(1)
while True:
    ret,test_img=cvv.read()
    #imshow(test_img)
    if ret:
        face_detected,gray_img=faceDetection(test_img)
        for face in face_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+int(w/2),x:x+h]
            label,confidence=face_recognizer.predict(roi_gray)#predicts
            
            draw_rec(test_img,face)
            predicted_name=name[label];
            print('confidence:{}',confidence)
            print("label:",label)
            if confidence>40:
                print("no confidence")
                continue
            
            put_text(test_img,predicted_name,x,y)
            #put_text(test_img,strings(confidence),x+h,y)
            print('confidence:{}',confidence)
            print("label:",label)
            #resized_image=cv2.resize(test_img,(500,700))
            resized_image=test_img
            imshow(test_img)
            #cv2.imshow("hii",resized_image)
    else :print ("not returning")
    time.sleep(0.1)
    k=cv2.waitKey(0);
    if(k%256==27) :break;
            
        
cv2.VideoCapture().release()
cv2.destroyAllWindows()

    
    

