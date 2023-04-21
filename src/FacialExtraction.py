import cv2 as cv
import os 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import MySQLdb
import numpy as np
import base64
from mtcnn.mtcnn import MTCNN



# Directory path
directory_path = 'out'

# Save all .png files into a dictionary 
save_png_files = {}

# Get a list of all the PNG files in the directory
png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

# Loop through the list of PNG files and open each file using cv2.imread()
for filename in png_files:
    file_path = os.path.join(directory_path, filename)
    save_png_files[filename] = file_path



''' Trying Facial Extraction '''
# Initialize the detector 
detector = MTCNN()
for images in enumerate(save_png_files):
    img_path = os.path.join(directory_path, filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    num_faces = len(results)

    # Store all detected areas
    detected_areas = {}

    # Then, create a list to store all
    bounding_box_x_y_w_h = [[]]

    for j in range(num_faces):
        x, y, width, height = results[j]['box']
        face = img[y:y+height, x:x+width]
        bounding_box_x_y_w_h.append(face)

for faces in save_png_files:
    # Store the bounding boxes for each face in a dictionary
    face_bounding_boxes = []
    for result in results:
        x, y, width, height = result['box']
        face_bounding_boxes.append((x, y, width, height))
        bounding_box_x_y_w_h.append(face_bounding_boxes)
    detected_areas[faces] = face_bounding_boxes



''' Cropping captured and detected face areas to 160x160 '''
# Initialize cropped_faces_img as an empty list
cropped_faces_img = []

# Loop through the images in save_png_files
for filename in save_png_files:
    img_path = os.path.join(directory_path, filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    areas_to_crop = len(results)

    for j in range(areas_to_crop):
        x, y, width, height = results[j]['box']
        cropped_img = img[y:y+height, x:x+width]
        cropped_img = cv.resize(cropped_img, (160, 160))
        print("Shape: ", cropped_img.shape)

        # Append the cropped image to cropped_faces_img
        cropped_faces_img.append(cropped_img)


''' Plotting '''
length_cropped_img = len(cropped_faces_img)
fig, axs = plt.subplots(nrows=length_cropped_img, ncols=2, figsize=(8, 8))
for i in range(length_cropped_img):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(cropped_faces_img[i])
plt.show()