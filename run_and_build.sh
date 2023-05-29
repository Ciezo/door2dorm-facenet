#!/bin/sh

echo "Initiating running scripts"
echo "Going into source directory"
cd src
echo "Listing all Python scripts" 
echo "================================================================================="
ls -la *.py
echo "================================================================================="
echo "Fetching images from remote database...."
py SaveImgToLocal.py
echo "Plotting images....."
py PlotImages.py
echo "Extracting facial features...."
py FacialExtraction.py
echo "Defining bounding areas"
py BoundingAreas.py
echo "Implementing FaceNet and building Facial Recognition Model"
py FaceNet.py
echo "Attempting to run FaceNet"
cd main
echo "Running FaceNet....."
py main.py