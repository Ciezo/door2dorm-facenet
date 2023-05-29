#!/usr/bin/env python
import cv2 as cv
import os 
import matplotlib.pyplot as plt



# Define the path to the directory containing the PNG files
directory_path = 'out/train/images'

# Save all .png files into a dictionary 
save_png_files = {}

# Get a list of all the PNG files in the directory
png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

# Loop through the list of PNG files and open each file using cv2.imread()
for filename in png_files:
    file_path = os.path.join(directory_path, filename)
    save_png_files[filename] = file_path

for items in save_png_files:
    print("Checking saved local images")
    print(items)

'''
    Begin plotting images
'''
# Count the number of images
num_images = len(save_png_files)
# Set up the plot
fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(num_images*5,5))

for i, filename in enumerate(save_png_files):
    img_path = os.path.join(directory_path, filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)    # Set the images to RGB channel
    axes[i].imshow(img)
    axes[i].set_title(filename)
plt.show()