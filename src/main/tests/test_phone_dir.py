''' @description This is a white-box unit testing to get all phone numbers from the database '''



''' CODE GOES HERE ''' 
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


# Fetch all phone numbers from the database
sql_get_all_phoneNumbers = "SELECT mobile_num FROM TENANT"
cursor.execute(sql_get_all_phoneNumbers)
rows = cursor.fetchall()

# Extract mobile_num values from the rows
phone_dir = [row[0] for row in rows]
def tenant_phone_directory():
    return phone_dir

# Close the cursor and database connection
cursor.close()
db.close()



''' RESULTS AND OUTPUTS HERE '''
numbers = tenant_phone_directory()
for items in numbers:
    print(items)