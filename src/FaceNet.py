import cv2 as cv
import os 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import MySQLdb
import numpy as np
import base64
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN



''' Template to load and extract faces '''
class FACELOADING:
    ''' We need to read through the ../out directory '''
    def __init__(self, directory):
        print("Going into, ", directory)
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()
    

    def extract_face(self, filename):
        print("Reading images ====> ", filename)
        img = cv.imread(filename)
        print("Converting to RGB Channels")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        print("Bounding area: ")
        print("\t ===================")
        print("\t x: ", x)
        print("\t y: ", y)
        print("\t width: ", w)
        print("\t height: ", h)
        print("\t ===================")
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        print("Resizing: ", self.target_size)
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    

    def load_faces(self, path):
      FACES = []
      print("Loading faces...")
      if not os.path.isfile(path):
          return FACES
      try:
          single_face = self.extract_face(path)
          FACES.append(single_face)
      except Exception as e:
          print("An error has occurred in loading facial capture images")
          pass
      return FACES


    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            print("Loading images from: ", path)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        print("Plotting images....")
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')



''' Checking and list .png files '''
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



''' MTCNN '''
detector = MTCNN()



''' Saving cropped face images '''
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



''' FaceNet '''
facenet = FaceNet()
def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0) 
    # 4D (Nonex160x160x3)
    yhat= facenet.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

faceloading = FACELOADING("out/train/images")
X, Y = faceloading.load_classes()
EMBEDDED_X = []

# Assign all our cropped images
for cropped_faces_img in X:
    EMBEDDED_X.append(get_embedding(cropped_faces_img))

EMBEDDED_X = np.asarray(EMBEDDED_X)


''' Saving class files '''
if((os.path.exists('out/embed'))):
    np.savez_compressed('out/embed/faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)
    print("File compressed")
else: 
    os.mkdir('out/embed')
    print("Embedded directory not found")
    print("Directory created!")
    np.savez_compressed('out/embed/faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)
    print("File compressed")



''' SVM Modelling '''
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
plt.plot(EMBEDDED_X[0]) 
plt.ylabel(Y[0])



''' Training and testing datasets '''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)



''' Data Modelling '''
# Training the model
from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_train, ypreds_train)

accuracy_score(Y_test,ypreds_test)



''' Testing the Model '''
t_im = cv.imread("out/test/images/Alfredo Vicente.png")
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']
print("Tested 1 image: ")
print("Results: ")
print("=================")
print("\t x: ", x)
print("\t y: ", y)
print("\t width: ", w)
print("\t height: ", h)
print("=================")

t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = get_embedding(t_im)
print("Embeddings results: ")
print(test_im)

test_im = [test_im]
ypreds = model.predict(test_im)



''' Saving the Model '''
import pickle
save_path = 'out/model'
if((os.path.exists(save_path))):
    with open('out/model/svm_model_160x160.pkl','wb') as f:
        print("Model saved to: ", save_path)
        pickle.dump(model,f)
else:
    os.mkdir(save_path)
    with open('out/model/svm_model_160x160.pkl','wb') as f:
        pickle.dump(model,f)