#!/usr/bin/env python
'''
    This FaceNet recognition implementation has a time-in security logging feature
'''

# Setup remote database connection
import os
from dotenv import load_dotenv
import MySQLdb

# Load up the .env file
try:
    if (load_dotenv()):
        print("Loaded .env variables")
    else:
        print("Error in loading variables or file not found!")

except Exception:
    print("Error in loading .env file")
## Connect to database
host = os.environ.get("DB_HOST")
user = os.environ.get("DB_USER")
pw = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_SCHEMA")
try:
    connection = MySQLdb.connect(host, user, pw, db_name)
    print("Successfully connected to ", db_name)
    connection.close()
except Exception:
    print("Error connecting!")

''' Instantiate database '''
# Create an instance for the db and cursor
db = MySQLdb.connect(host, user, pw, db_name)
cursor = db.cursor()


# =================================================================================================================== #


''' Running our FaceNet implementation just like in main.py '''
import cv2 as cv
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



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
cap = cv.VideoCapture(0)


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

        ''' When the confidence is greater than 70% threshold, then authorized'''
        if (confidence > 70):
            ''' This is the recognition confidence from our Model '''
            embedding_scores = model.decision_function(ypred)
            print("Scores: ", embedding_scores)
            recognition_score = model.decision_function(ypred)[0]
            final_recognition_score = int(100*(1-recognition_score[0]/10))
            print("Recognition score: ", final_recognition_score)
            
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
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 10)                        # Green box (BGR)
                # Ender the name on screen real-time 
                cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,       # Blue text             
                        1, (255,0,0), 3, cv.LINE_AA)   

                
                ''' @todo Insert into the database table for AUTHORIZED TENANTS, SECURITY_LOGS_TIME_OUT '''  
                # attributes: log_id (int), tenant_name (str), tenant_room (str), time_in (str), status (str), capture (blob)
                sql_get_tenant_room = "SELECT room_assign FROM TENANT WHERE full_name = '{}'".format(final_name)
                cursor.execute(sql_get_tenant_room)
                print(sql_get_tenant_room)
                # Fetch the assigned room data into a var
                res_tenant_room = cursor.fetchone()
                res_tenant_room = int(res_tenant_room[0])
                print("Assigned room: ", res_tenant_room)
                # Time
                from datetime import datetime
                current_time = datetime.now()
                time_in = current_time.strftime("%H:%M:%S")
                
                sql_log_time_in = "INSERT INTO SECURITY_LOGS_TIME_OUT (tenant_name, tenant_room, time_in, status, capture) VALUES (%s, %s, %s, %s, %b)"
                val = (final_name, res_tenant_room, time_in, "Authorized", rgb_img)
                # val = {
                #     'tenant_name': final_name,
                #     'tenant_room': res_tenant_room,
                #     'time_in': time_in,
                #     'status': 'Authorized',
                #     'capture': frame,
                # }
                try: 
                    cursor.execute(sql_log_time_in, val)
                    db.commit()
                    print("Inserted into security loggings")
                except Exception:
                    print("Something went wrong when inserting to security logs...")
            else: 
                print("\t ==> Status: Unauthorized")
                cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 10)                        # Red box
                # Display "Unauthorized" text on screen real-time
                cv.putText(frame, "Unauthorized", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,        # Red text
                        1, (0,0,255), 3, cv.LINE_AA)
            



 
        else: 
            ''' When confidence < 70 '''
            print("\t ==> Status: Unauthorized")
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 10)                        # Red box
            # Display "Unauthorized" text on screen real-time
            cv.putText(frame, "Unauthorized", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,        # Red text
                    1, (0,0,255), 3, cv.LINE_AA)
            

            ''' @todo Insert into the database table for UNAUTHORIZED entries '''
            sql_log_time_in_UNAUTHORIZED = "INSERT INTO SECURITY_LOGS_TIME_OUT (tenant_name, tenant_room, time_in, status, capture) VALUES (%s, %s, %s, %s, %b)"
            val2 = ('Unknown name', 'Unknown room', time_in, "Unauthorized", rgb_img)
            cursor.execute(sql_log_time_in_UNAUTHORIZED, val2)
            db.commit()
        
    cv.imshow("Face Recognition (TIME-OUTS):", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows