import cv2;
import os
import numpy as np
import time
from  face_recognition import *
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
def imshow(img):
    import IPython
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)
(X,Y)=(labels_for_training_images('C:/Users/rajar/Documents/summercoding/ML/to train/face detection/IMAGES'))

print(type(Y))
print(type(X))
print("X:",X.shape)
print("Y:",Y.shape)
print("X:",X.dtype)
print("Y:",Y.dtype)


print("dtype is: ",X.dtype)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

print("xtrain:",x_train.shape)
print("xtest:",x_test.shape)
print("ytrain:",y_train.shape)
print("ytest:",y_test.shape)
# for i in range(0,100):
#     cv2_imshow(x_train[i])
input_img_row=x_train[0].shape[0]
input_img_col=x_train[0].shape[1]
print(input_img_row)
print(input_img_col)
# for i in range(0,100):
#     cv2_imshow(x_train[i])
from tensorflow.keras.models import Sequential
x_train=x_train.reshape(x_train.shape[0],input_img_row,input_img_col,1)
x_test=x_test.reshape(x_test.shape[0],input_img_row,input_img_col,1)
print(x_test.shape)
input_shape=(input_img_row,input_img_col,1)
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")
#normalize the data
x_train=x_train/255
x_test=x_test/255
from tensorflow.python.keras.utils import np_utils
num_classes=10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print(y_train.shape)
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPool2D
from tensorflow.keras.optimizers import SGD
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(130,260,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer=SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
train=model.fit(x=x_train,y=y_train,batch_size=35,epochs=2,verbose=0,validation_data=(x_test,y_test))
score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
model.save("cnn_model1")


from tensorflow.keras.models import load_model
loaded_model=load_model("cnn_model1")
cvv=cv2.VideoCapture(1);
while True:
    ret,img=cvv.read();
    if ret:
        rect,gray_img=faceDetection(img)
        for (x,y,w,h) in rect:
            print("entered for loop")
            roi_gray=gray_img[y:y+130,x:x+260]
            imshow(roi_gray)
            roi_gray=roi_gray.reshape(1,130,260,1)
            results=str(loaded_model.predict_classes(roi_gray,batch_size=1,verbose=0))
            print(results)
            
    else :print ("not returning;")
cvv.release();
