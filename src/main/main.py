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


''' FaceNet '''
facenet = FaceNet()
# Read the encoded x, y labels
face_embeddings = np.load("../../classes/faces_embeddings_done_4classes.npz")
# Get Y 
Y = face_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)



''' List all Authorized Names '''
AUTHORIZED_NAMES = []
for names in Y:
    # Remove extensions
    temp_name_rem_ext = names.rstrip(".png").rstrip(".jpeg") 
    final_auth_name = temp_name_rem_ext
    AUTHORIZED_NAMES.append(final_auth_name)


''' Face Detector '''
haarcascade = cv.CascadeClassifier("../../data/models/haarcascade_frontalface_default.xml")
''' 
    @note the model used here is much faster than MTCNN which is good for real time
'''

''' Model '''
model = pickle.load(open("../../data/models/svm_model_160x160.pkl", "rb"))


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
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)

        ''' Try and define a confidence score '''
        confidence = model.predict(facenet.embeddings(img))
        ''' Convert the confidence score '''
        confidence = int(100*(1-confidence/10))
        print("Confidence: ", confidence)

        ''' Anti-spoofing '''
        spf_label = test(
                image=frame,
                model_dir='../../resources/anti_spoof',
                device_id=1
                )
        
        ''' If Face detected is real '''
        if spf_label == 1: 
            ''' When the confidence is greater than 70% threshold, then authorized'''
            if (confidence > 70):
                ''' This is the recognition confidence from our Model '''
                embedding_scores = model.decision_function(ypred)
                print("Scores: ", embedding_scores)
                recognition_score = model.decision_function(ypred)[0]
                # final_recognition_score = int(100*(1-recognition_score/10))
                # print("Recognition score: ", final_recognition_score)
                
                ''' Fetching the names from the array '''
                final_name = encoder.inverse_transform(face_name)[0]
                ''' Remove the extensions '''
                rem_ext = final_name.rstrip(".png").rstrip(".jpeg")
                final_name = rem_ext

                # Check if the face recognized is authorized or not
                if final_name in AUTHORIZED_NAMES:
                    # Then, we proceed to render them real time and mark as Authorized
                    ''' Display to console the recognized face '''
                    print("Recognized: ", final_name)
                    print("Confidence: ", confidence)
                    print("\t ==> Status: Authorized")
                    cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)                        # Green box (BGR)
                    # Ender the name on screen real-time 
                    cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,       # Blue text             
                            1, (255,0,0), 3, cv.LINE_AA)        
                    
                else: 
                    print("\t ==> Status: Unauthorized")
                    cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)                        # Red box
                    # Display "Unauthorized" text on screen real-time
                    cv.putText(frame, "Unauthorized", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,        # Red text
                            1, (0,0,255), 3, cv.LINE_AA)
            


       
            
        else: 
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)                        # Red box
            # Display "Unauthorized" text on screen real-time
            cv.putText(frame, "Invalid. Spoofing Detected!", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,        # Red text
                    1, (0,0,255), 3, cv.LINE_AA)   
        
    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows