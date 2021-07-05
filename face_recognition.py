

import cv2;
import os
import numpy as np
import face_recognition as fr;

def imshow(img):
    import IPython
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
#given an image ,this function return rectangle for fave
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haarCascase=cv2.CascadeClassifier("C:/Users/rajar/Documents/summercoding/ML/to train/face detection/haarcascade_frontalface_default.xml")
    faces=face_haarCascase.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    #detectMultiScale returns the rectangles of faces
    return faces,gray_img;
#give labels to training images
def labels_for_training_images(directory):
    faces=[];
    faceID=[]
    for path,subDirNames,fileNames in os.walk(directory):
        for fileName in fileNames:
            if fileName.startswith('.'):
                continue;
            id=os.path.basename(path)
            img_path=os.path.join(path,fileName)#fetching the image path
            print("image path:",img_path)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print('image not loaded')
                continue;
            facese_rect,gray_image=faceDetection(test_img);
            if len(facese_rect)>2:
                continue;
            for (x,y,w,h) in facese_rect:
                roi_gray=gray_image[y:y+int(w/2),x:x+h]
                faces.append(roi_gray)
                imshow(roi_gray)
                print(id);
                print("lfti  ")
                faceID.append(int(id))
    return  faces,faceID


#Below function trains the classifier using the training images;
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rec(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),1)
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)
    
            