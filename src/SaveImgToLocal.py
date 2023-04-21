import cv2 as cv
import os 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import MySQLdb
import numpy as np
import base64



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load up the .env file
try:
    if (load_dotenv()):
        print("Loaded .env variables")
    else:
        print("Error in loading variables or file not found!")

except Exception:
    print("Error in loading .env file")
## Connect to database
host = os.environ.get("DB_HOST")
user = os.environ.get("DB_USER")
pw = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_SCHEMA")
try:
    connection = MySQLdb.connect(host, user, pw, db_name)
    print("Successfully connected to ", db_name)
    connection.close()
except Exception:
    print("Error connecting!")



''' Instantiate database '''
# Create an instance for the db and cursor
db = MySQLdb.connect(host, user, pw, db_name)
cursor = db.cursor()




''' Trying to save all images from the database '''
# Define a global variable to store all IMG bytearray and data
face_capture = []
face_img = []

# Face Entity and all Facts from the table 
face_id = []
tenant_id = []
tenant_name = []
face_status = []


def fetch_all_FaceCaptures():
    '''
        Begin making queries to the database and SELECT our FACE_IMG table
        where all facial captures are stored
    '''
    # Create a query to select all facial captures
    sql_fetch_faceCaptures = "SELECT * FROM FACE_IMG"
    cursor.execute(sql_fetch_faceCaptures)
    # Fetch all data as rows
    rows = cursor.fetchall()

    # Iterate and assign data as to row
    for row in rows:
        face_id = row[0]
        tenant_id = row[1]
        tenant_name = row[2]
        face_status = row[3]
        face_capture = row[4]
        # Convert the face_capture column to a bytes-like object
        face_capture_bytes = bytearray(face_capture)
        # Use the cv.imdecode() function to decode the bytes-like object into an image
        face_img = cv.imdecode(np.frombuffer(face_capture_bytes, np.uint8), cv.IMREAD_COLOR)

        # Display some attribute data 
        print("==============================================================")
        print("Face ID: ", face_id)
        print("Tenant ID: ", tenant_id)
        print("Tenant Full Name: ", tenant_name)
        print("Status: ", face_status)
        print("Face IMG data: ", face_img)
        print(" ---> Type: ", type(face_img))
        print("==============================================================")

        # Render the image 
        try:
            # Display the image
            cv.imshow(f"{tenant_name}", face_img)
            cv.waitKey(0)
            try: 
                # encoded_img = cv.imencode(".png", face_img)
                # print("Converting image to .png")
                # face_img_png = encoded_img.tobytes()
                # with open(f'out/{tenant_name}.png', 'wb') as f:
                #     # Decode the base64-encoded PNG data
                #     decoded_data = base64.b64decode(face_img_png)
                #     print(f"Saved PNG file: out/{tenant_name}.png")
                #     # Write the decoded data to the file stream
                #     f.write(decoded_data)
                
                # Convert image to PNG format and store it in a variable
                cv.imwrite(f'out/{tenant_name}.png', face_img)
            except:
                print("An error has occurred converting the image")
        except: 
            print("There was an error occurred for rendering the image")

# Call the function to test it
fetch_all_FaceCaptures()