#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys,os
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from keras.layers.core import  Dropout
from keras.layers import Dense, Conv2D, MaxPool2D ,ZeroPadding2D, Flatten
from tensorflow.keras.layers.experimental import preprocessing
from keras.optimizers import SGD
import cv2
import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


emotion_data = pd.read_csv('C:\\Users\\HP\\Desktop\\fer2013.csv')


# In[3]:


X_train = []
y_train = []
X_test = []
y_test = []
for index, row in emotion_data.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
#print(X_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(y_test.shape)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
#print(X_train.shape)

y_train= keras.utils.to_categorical(y_train, num_classes=7)
y_test =keras.utils.to_categorical(y_test, num_classes=7)


# In[4]:


model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(48, 48, 1)),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.05),
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation = 'softmax'),
])
gen = ImageDataGenerator()
train_gen = gen.flow(X_train,y_train,batch_size=512)
gen1= ImageDataGenerator()
test_gen=gen1.flow(X_test,y_test,batch_size=512)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_gen,validation_data=test_gen,batch_size=512,epochs = 20)
history_frame = pd.DataFrame(history.history)
# history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy']].plot();
fer_json = model.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model.save_weights("fer.h5")  


# In[5]:




from keras.models import load_model
from time import sleep
#from keras.models import model_from_jason
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#model = model_from_json(open("fer.json", "r").read())  
#load weights  
model.load_weights('fer.h5') 

face_classifier = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Emotion-recognition-master\haarcascade_files\haarcascade_frontalface_default.xml")
#classifier =load_model(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)




while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction =model.predict(roi)[0]
#             print(prediction)
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




