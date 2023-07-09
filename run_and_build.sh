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
python FaceNet.py
cd ..
echo "Going to root"
echo "Updating classes"
echo "Copying files from embed"
ls src/out/embed
cp src/out/embed/* classes
echo "Updating models"
ls src/out/model
cp src/out/model/* data/models
echo "Attempting to run FaceNet"
echo "Going to main"
cd src/main
echo "Running FaceNet....."
python main.py
