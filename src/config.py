'''
    @author Cloyd Van S. Secuya
    @description This is a simple script to test the configured remote database
'''

import os
from dotenv import load_dotenv
import MySQLdb

# Load up the .env file
try:
    if (load_dotenv()):
        print("Loaded .env variables")
    else:
        print("Error in loading variables or file not found!")

except Exception:
    print("Error in loading .env file")

# DB credentials from .env
host = os.environ.get("DB_HOST")
user = os.environ.get("DB_USER")
pw = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_SCHEMA")

# Establish connection 
try:
    connection = MySQLdb.connect(host, user, pw, db_name)
    print("Successfully connected to ", db_name)
except Exception:
    print("Error connecting!")