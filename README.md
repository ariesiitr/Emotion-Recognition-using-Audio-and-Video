# Emotion-Recognition-using-Audio-and-Video
Recruitment Project for 2nd Year

Human beings display their emotions using facial expressions. For humans it is very easy to
recognize
those emotions but for computers it is very challenging. Facial expressions vary from person to
person. Brightness, contrast and resolution of every random image is different. This is why
recognizing facial expressions is very difficult. Facial expression recognition is an active research
area. In this project, we worked on recognition of seven basic human emotions. These emotions
are angry, disgust, fear, happy, sad, surprise and neutral.
The model involves generally 3 steps training, testing and validation and after completion of these
steps we input our audio or video to predict emotion in the input.

## FER2013 Dataset
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically
registered so that the face is more or less centered and occupies about the same amount of space
in each image. The task is to categorize each face based on the emotion shown in the facial
expression into one of seven categories .
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
The training set consists of 28,709 examples. The public test set consists of 3,589 examples.

Firstly we have done emotion recognition using video in which we used basic CNN algorithm ,
trained and tested with the help of fer2013 dataset to predict emotion. Below you can see the code
and explanation of the video part. Emotion recognition using video has been completed. Now
coming to emotion recognition using the audio part , RAVDESS dataset
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio will be used in this part.
We have started studying RNN and research papers of deep speech to implement this part. And
then we will try to sum up both parts and predict emotion using audio and video both.

### Import required libraries:
Import the required libraries for building the network. The code for importing the libraries is written
below

```python

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
```

After importing all the libraries we have to import dataset which is to be used,we can give the
directory in small brackets where we have stored the dataset as shown in the code below.
```python
emotion_data = pd.read_csv('C:\\Users\\HP\\Desktop\\fer2013.csv')
```

### Extract Data 
We will then create a different list for testing and training image pixels. After this we will check if
pixels belong to the training set or testing set and we will append to the training or testing list
respectively.The code for this is written below.

```python
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
```
### Model 

Now it's time to design the CNN model for emotion detection with different layers. We start with the
initialization of the model followed by batch normalization layer and then different convents layers
with ReLu as an activation function, max pool layers, and dropouts to do learning efficiently.

```python

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
```

<div align="center">
<img width="525" alt="Screenshot 2021-05-21 at 8 37 37 PM" src="https://user-images.githubusercontent.com/57126154/119161977-7dd64100-ba77-11eb-8172-69d85fd19011.png">
</div>

We have also compiled the model using Adam as an optimizer, loss as categorical
cross-entropy, and metrics as accuracy as shown in the below code.

```python
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_gen,validation_data=test_gen,batch_size=512,epochs = 20)
```
### Testing the model in Real-time using OpenCV and WebCam

Libraries Required - 

```python
from keras.models import load_model
from time import sleep
#from keras.models import model_from_jason
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
```
After importing all the required libraries we will load the model weights that we saved earlier
after training. Use the below code to load your saved model. After importing the model weights
we have imported a haar cascade file that is designed by open cv to detect the frontal face.

```python

face_classifier = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Emotion-recognition-master\haarcascade_files\haarcascade_frontalface_default.xml")
```
After importing the haar cascade file we will have written a code to detect faces and classify
the desired emotions.

```python
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

```
<div align="center">
<img width="674" alt="Screenshot 2021-05-21 at 8 36 10 PM" src="https://user-images.githubusercontent.com/57126154/119161790-51bac000-ba77-11eb-8c36-121197017b33.png">
</div>
After 50 epochs we got accuracy as 62.7% and validation accuracy as 52.94%.

## Emotion Recognition from Audio

In this mini project,Building and training a simple Speech Emotion Recognizer that predicts human
emotions from audio files using Python, Sci-kit learn, librosa, and Keras. Firstly, we will load the
data (Ravdess dataset), extract features (MFCC) from it, and split it into training and testing sets.
Then, we will initialize CNN model as emotion classifiers and train them. Finally, we will calculate
the accuracy of our models.

The whole pipeline is as follows (as same as any machine learning pipeline):

1) Loading the Dataset: This process is about loading the dataset in Python which involves
extracting audio features, such as MFCC.

2) Training the Model: After we prepare and load the dataset, we simply train it on a suited model.

3) Testing the Model: Measuring how good our model is doing.


### RAVDESS Dataset

Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS. Full dataset of speech and song,
audio and video (24.8 GB) . 

Files:
This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The
RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched
statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry,
fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional
intensity (normal, strong), with an additional neutral expression.

Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier
(e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics.
Filename identifiers:

* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 =
disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the
'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

### Import Required libraries

```Python

import librosa
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
```

### Some Important terms

Librosa:
Librosa is a python package for music and audio analysis. It provides the building blocks
necessary to create music information retrieval systems.

Mel-frequency Cepstrum Coefficients (MFCC):
The sounds generated by a human are filtered by the shape of the vocal tract including tongue, teeth etc.
This shape determines what sound comes out. If we can determine the shape accurately, this should give us
an accurate representation of the phoneme being produced. The shape of the vocal tract manifests itself in
the envelope of the short time power spectrum, and the job of MFCCs is to accurately represent this
envelope.

A multilayer perceptron (MLP):
A multilayer perceptron (MLP) is a deep, artificial neural network. It is composed of more than one
perceptron. They are composed of an input layer to receive the signal, an output layer that makes a decision
or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the
true computational engine of the MLP.
<div align="center">
<img width="588" alt="Screenshot 2021-05-21 at 9 03 37 PM" src="https://user-images.githubusercontent.com/57126154/119162491-1240a380-ba78-11eb-8ae0-2c4073859782.png">

</div>

### Loading Dataset

```python
audio = r"C:\Users\HP\Desktop\dataset 2\audio_speech_actors_01-24"
actor_folders = os.listdir(audio) #list files in audio directory
actor_folders.sort() 
actor_folders[0:5]

emotion = []
gender = []
actor = []
file_path = []
for i in actor_folders:
    filename = os.listdir(audio + "\\" +i) #iterate over Actor folders
    for f in filename: # go through files in Actor folder
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        actor.append(int(part[6]))
        bg = int(part[6])
        if bg%2 == 0:
            bg = "female"
        else:
            bg = "male"
        gender.append(bg)
        file_path.append(audio + "\\" + i + "\\" + f)
```

Then we convert our data into data frame and then extract mfcc features , you can refer code for more info.

### Model 

```python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import RMSprop

#model = Sequential()
# First GRU layer with Dropout regularisation
#model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
#model.add(Dropout(0.2))
# Second GRU layer
#model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
#model.add(Dropout(0.2))
# Third GRU layer
#model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
#model.add(Dropout(0.2))
# Fourth GRU layer
#model.add(GRU(units=50, activation='tanh'))
#model.add(Dropout(0.2))
# The output layer
#model.add(Dense(units=8))

#model = Sequential()
#model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
#model.add(Dense(64))
#model.add(Dropout(0.4))
#model.add(Activation('relu'))
#model.add(Dense(32))
#model.add(Dropout(0.4))
#model.add(Activation('relu'))
#model.add(Dense(8))
#model.add(Activation('softmax'))

#BUILD 1D CNN LAYERS
model = tf.keras.Sequential()
model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Dropout(0.4))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(8, activation='sigmoid'))
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
```
```python
import tensorflow.keras as keras

# FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max', period=1, save_weights_only=True)

model_history=model.fit(X_train, y_train,batch_size=32, epochs=50, validation_data=(X_test, y_test),callbacks=[checkpoint])
```
<div align="center">
<img width="644" alt="Screenshot 2021-05-21 at 8 37 04 PM" src="https://user-images.githubusercontent.com/57126154/119161914-6bf49e00-ba77-11eb-9ec0-3f9870009f43.png">
</div>

First we tried it using RNN but accuracy was not that much that (RNN model is commented) and rest is CNN part which gives decent accuracy.

### Printing Confusion Matrix 

```python
cm = confusion_matrix(actual, predictions)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in lb.classes_] , columns = [i for i in lb.classes_])
ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig('Initial_Model_Confusion_Matrix.png')
plt.show()
```
<div align="center">
<img width="672" alt="Screenshot 2021-05-21 at 8 35 33 PM" src="https://user-images.githubusercontent.com/57126154/119160467-f0deb800-ba75-11eb-8e20-ad84cab7de52.png">
</div>

### Print classification report:

```python
print(classification_report(actual, predictions, target_names = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']))
```
After 50 epochs we got accuracy as 62.53% and validation accuracy as 54.51%.
