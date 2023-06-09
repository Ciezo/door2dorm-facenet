''' @description this is a white-box unit test to see if alerting all phone directories are working '''



''' @job Twilio SMS service '''
import os
from dotenv import load_dotenv
from twilio.rest import Client

print("UNIT TEST 1")
print("_________________________________________________________________________________________")

# Load up the .env file and get the Twilio keys
try:
    if (load_dotenv()):
        print("Twilio keys loaded")
    else:
        print("An error occurred in loading the Twilio keys")

except Exception:
    print("Error in loading .env file")


# Twilio keys
account_sid = os.environ.get("ACC_SID_TWILIO")
account_token = os.environ.get("AUTH_TOKEN_TWILIO")
account_number = os.environ.get("PHONE_NUM_TWILIO")

# Initialize an SMS REST Client
try: 
    client = Client(account_sid, account_token)
    print("SMS Client created!")
    print("SID: ", account_sid)
    print("Token: ", account_token)
    print("Twilio Number: ", account_number)
    print("\t ===> SMS messages and notifications are now ready!")

except Exception:
    print("An error occurred creating the SMS Client")
    print(Exception)


def sms_alert_msg(body, to_number):
    message = client.messages.create(
        body = body,
        from_ = account_number,
        to = to_number
    )

    print("Message has been sent to: ", to_number)
    return message
print(">>> END UNIT TEST 1")
print("_________________________________________________________________________________________")
# ================================================================================================ #








''' @job Create the phone directory based on the numbers of tenants from the database '''
import MySQLdb

print("UNIT TEST 2")
print("_________________________________________________________________________________________")

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

print(">> END UNIT TEST 2")
print("_________________________________________________________________________________________")








'''' @job To alert all tenants using the SMS messages '''

print("UNIT TEST 3")
print("_________________________________________________________________________________________")

numbers = tenant_phone_directory()
i = 0
for receivers in numbers:
    i = i + 1
    message = "Unauthorized Alert SMS Notification Test: '{}'".format(i)
    sms_alert_msg(message, receivers)
    print("\t\t Body: ", message)

print(">> END UNIT TEST 3")
print("_________________________________________________________________________________________")