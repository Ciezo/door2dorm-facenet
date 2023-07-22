#!/usr/bin/env python
import cv2 as cv

'''
This script is to test running two cameras at the same time.
We need to make sure that frames are being read from two external camera sources

Where, 
camera is frame
camera2 is frame2
'''


cap = cv.VideoCapture(2)        # in-built webcam
cap2 = cv.VideoCapture(1)       # external webcam 1 (right USB)

# Setting scale and res
cap.set(3, 1920)
cap.set(4, 1080)
cap2.set(3, 1920)
cap2.set(4, 1080)

while cap.isOpened() and cap2.isOpened():
    _, frame = cap.read()     # Read video frames
    _, frame2 = cap2.read()     # Read video frames

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   
    rgb_img2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
    gray_img2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    cv.imshow("In-built webcam:", frame)
    cv.imshow("External webcam:", frame2)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release() 
cap2.release() 
cv.destroyAllWindows