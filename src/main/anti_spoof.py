#!/usr/bin/env python
import cv2 as cv
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import the test() from Silent-Face-Anti-Spoofing module
# This Silent-Face-Anti-Spoofing directory must be in the PYTHONPATH environment variable
from test import test


''' Face Detector '''
haarcascade = cv.CascadeClassifier("../../data/models/haarcascade_frontalface_default.xml")
''' 
    @note the model used here is much faster than MTCNN which is good for real time
'''

''' Initializing video capture '''
'''
@note
    index 0 is for built-in webcam
    index 1 is for external webcam
'''
cap = cv.VideoCapture(0)
# Setting scale and res
cap.set(3, 1920)
cap.set(4, 1080)


''' Real time face detection and recognition '''
while cap.isOpened():
    _, frame = cap.read()     # Read video frames
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    # Bounding areas for detected faces
    for x,y,w,h in faces:
        ''' Anti-spoofing '''
        spf_label = test(
                image=frame,
                model_dir='../../resources/anti_spoof',
                device_id=1
                )
        
        ''' If the face detected is real.'''
        if spf_label == 1:            
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)                       # Green box (BGR)
            # Ender the name on screen real-time 
            cv.putText(frame, "Real face", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,         # Blue text             
                    1, (255,0,0), 3, cv.LINE_AA)   
            
            # FaceNet code goes under here.....same how it can recognize faces
   
        else: 
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)                        # Red box
            # Display "Unauthorized" text on screen real-time
            cv.putText(frame, "Invalid. Spoofing Detected!", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,        # Red text
                    1, (0,0,255), 3, cv.LINE_AA)   

        
    cv.imshow("Face Recognition and Spoofing Detection:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows