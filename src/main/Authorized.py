import cv2 as cv
import numpy as np
import os
import tensorflow as tf
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

class AUTHORIZED_NAMES:
    def get_AUTHORIZED_NAMES(self):
        LS_AUTHORIZED_NAMES = []
        for names in Y:
            # Remove extensions
            temp_name_rem_ext = names.rstrip(".png").rstrip(".jpeg") 
            final_auth_name = temp_name_rem_ext
            LS_AUTHORIZED_NAMES.append(final_auth_name)
        return LS_AUTHORIZED_NAMES
