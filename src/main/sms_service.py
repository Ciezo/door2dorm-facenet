'''
    @author Cloyd Van S. Secuya
    @description This is a script to use in order to function and send an SMS message and notification
                 to a target Philippine phone number
'''



import os
from dotenv import load_dotenv
from twilio.rest import Client

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