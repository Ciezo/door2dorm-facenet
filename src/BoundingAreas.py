#!/usr/bin/env python
import cv2 as cv
import os 
from mtcnn.mtcnn import MTCNN



# Directory path
directory_path = 'out/train/images'

# Save all .png files into a dictionary 
save_png_files = {}

# Get a list of all the PNG files in the directory
png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

# Loop through the list of PNG files and open each file using cv2.imread()
for filename in png_files:
    file_path = os.path.join(directory_path, filename)
    save_png_files[filename] = file_path



''' Initialize detector for bounding areas '''
detector = MTCNN()
num_faces = 0
bounding_area = {}

for face in enumerate(save_png_files):
    img_path = os.path.join(directory_path, filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Detect faces using MTCNN
    results = detector.detect_faces(img)
    num_faces = len(face)
    print("Bounding area: ", results)
    bounding_area[face] = results

print(bounding_area)