#!/usr/bin/env python
import cv2 as cv


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
    faces = haarcascade.detectMultiScale(rgb_img, 1.3, 5)
    # Bounding areas for detected faces
    for x,y,w,h in faces:
        # Draw boxes to all detected faces    
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)                       # Green box (BGR)
        
    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows