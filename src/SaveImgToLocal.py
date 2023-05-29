#!/usr/bin/env python
import cv2 as cv
import os 
from dotenv import load_dotenv
import MySQLdb
import numpy as np



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

    # Prompt the user if they want to preview the images
    option = input("Do you want to preview images? (y or n): ")
    option = option.islower()

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
            if (option == 'y'): 
                # Display the image
                print(">>>>>>>>> Previewing IMAGES <<<<<<<<< ")
                print("Number of images to display: ", len(row))
                cv.imshow(f"{tenant_name}", face_img)
                cv.waitKey(0)
            else: 
                print("Display option: no")
                print("Downloading files....")
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
                    if not os.path.exists('out/train/images'):
                        os.makedirs('out/train/images')
                        print("Folder has been created")

                    else: 
                        cv.imwrite(f'out/train/images/{tenant_name}.png', face_img)
                        print("Images downloaded to", f'out/train/images/{tenant_name}.png')

                except:
                    print("An error has occurred converting the image")
        except: 
            print("There was an error occurred for rendering the image")

# Call the function to test it
fetch_all_FaceCaptures()


''' This function creates a dedicated sub-directory based on tenant name '''
''' Example: 'Cloyd_Van_Secuya/Cloyd Van Secuya.png'''
# def fetch_all_FaceCaptures():
#     '''
#         Begin making queries to the database and SELECT our FACE_IMG table
#         where all facial captures are stored
#     '''
#     # Create a query to select all facial captures
#     sql_fetch_faceCaptures = "SELECT * FROM FACE_IMG"
#     cursor.execute(sql_fetch_faceCaptures)
#     # Fetch all data as rows
#     rows = cursor.fetchall()

#     # Prompt the user if they want to preview the images
#     option = input("Do you want to preview images? (y or n): ")
#     option = option.islower()

#     # Iterate and assign data as to row
#     for row in rows:
#         face_id = row[0]
#         tenant_id = row[1]
#         tenant_name = row[2]
#         face_status = row[3]
#         face_capture = row[4]
#         # Convert the face_capture column to a bytes-like object
#         face_capture_bytes = bytearray(face_capture)
#         # Use the cv.imdecode() function to decode the bytes-like object into an image
#         face_img = cv.imdecode(np.frombuffer(face_capture_bytes, np.uint8), cv.IMREAD_COLOR)

#         # Display some attribute data 
#         print("==============================================================")
#         print("Face ID: ", face_id)
#         print("Tenant ID: ", tenant_id)
#         print("Tenant Full Name: ", tenant_name)
#         print("Status: ", face_status)
#         print("Face IMG data: ", face_img)
#         print(" ---> Type: ", type(face_img))
#         print("==============================================================")

#         # Render the image 
#         try:
#             if (option == 'y'): 
#                 # Display the image
#                 print(">>>>>>>>> Previewing IMAGES <<<<<<<<< ")
#                 cv.imshow(f"{tenant_name}", face_img)
#                 cv.waitKey(0)
#             else: 
#                 print("Display option: no")
#                 print("Downloading files....")
#                 try: 
#                     # encoded_img = cv.imencode(".png", face_img)
#                     # print("Converting image to .png")
#                     # face_img_png = encoded_img.tobytes()
#                     # with open(f'out/{tenant_name}.png', 'wb') as f:
#                     #     # Decode the base64-encoded PNG data
#                     #     decoded_data = base64.b64decode(face_img_png)
#                     #     print(f"Saved PNG file: out/{tenant_name}.png")
#                     #     # Write the decoded data to the file stream
#                     #     f.write(decoded_data)
                    
#                     # Convert image to PNG format and store it in a variable
#                     if os.path.exists('out/train/images'):
#                         # Replace spaces with underscores in tenant name
#                         tenant_dir = tenant_name.replace(" ", "_")
#                         # Create sub-directory path
#                         sub_dir = os.path.join('out/train/images', tenant_dir)
#                         if not os.path.exists(sub_dir):
#                             os.makedirs(sub_dir)
#                         cv.imwrite(os.path.join(sub_dir, f'{tenant_name}.png'), face_img)
#                         print(f"Images downloaded to {sub_dir}/{tenant_name}.png")
#                     else: 
#                         print("Directory is not existing")
#                         print("Making directory....")
#                         os.makedirs('out/train/images')
#                         # Replace spaces with underscores in tenant name
#                         tenant_dir = tenant_name.replace(" ", "_")
#                         # Create sub-directory path
#                         sub_dir = os.path.join('out/train/images', tenant_dir)
#                         if not os.path.exists(sub_dir):
#                             os.makedirs(sub_dir)
#                         cv.imwrite(os.path.join(sub_dir, f'{tenant_name}.png'), face_img)
#                         print(f"Images downloaded to {sub_dir}/{tenant_name}.png")

#                 except:
#                     print("An error has occurred converting the image")
#         except: 
#             print("There was an error occurred for rendering the image")

# # Call the function to test it
# fetch_all_FaceCaptures()