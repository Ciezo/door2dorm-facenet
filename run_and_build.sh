#!/bin/sh

echo "Initiating running scripts"
echo "Going into source directory"
cd src
echo "Listing all Python scripts" 
echo "================================================================================="
ls -la *.py
echo "================================================================================="
echo "Fetching images from remote database...."
python SaveImgToLocal.py
echo "Plotting images....."
python PlotImages.py
echo "Extracting facial features...."
python FacialExtraction.py
echo "Defining bounding areas"
python BoundingAreas.py
echo "Implementing FaceNet and building Facial Recognition Model"
pthon FaceNet.py
echo "Attempting to run FaceNet"
cd main
echo "Running FaceNet....."
python main.py
