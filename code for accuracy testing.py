# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:47:31 2021

@author: Mr.BeHappy
"""


import cv2;
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from  face_recognition import *
def imshow(img):
    import IPython
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
   
#print("face detected",face_detected)
#####comment these lines
faces,faceID=labels_for_training_images('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/IMAGES/')

#splitting the dataset
x_train,x_Test,y_train,y_test=train_test_split(faces,faceID,test_size=0.3,random_state=1);
print(type(faces),faces.dtype)
print(type(x_train),x_train.dtype)

face_recognizer=train_classifier(x_train,y_train)
face_recognizer.write('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/trainingData.yml')
#uncomment these lines while running from  2nd time onwards
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/trainingData.yml')

name={1:'Mr.BeHappy',2:'babi',3:"Raji Reddy"}#creating dictionary containing names
Y=np.array([]);
Y_pred=np.array([]);
print("started testing")
for i in  range (len(x_Test)):
        label,confidence=face_recognizer.predict(x_Test[i])#predicts
        predicted_name=name[label];
        print('confidence:{}',confidence)
        print("label:",label)
        if confidence>40:
            print("no confidence")
            continue
        #put_text(test_img,strings(confidence),x+h,y)
        print('confidence:{}',confidence)
        print("label:",label)
        Y=np.append(Y,y_test[i])
        Y_pred=np.append(Y_pred,label)
        #resized_image=cv2.resize(test_img,(500,700))
        imshow(x_Test[i])
        #cv2.imshow("hii",resized_image)
        #time.sleep(0.1)
            
accu=accuracy_score(Y,Y_pred);
print("accuracy=",accu)
    
    

