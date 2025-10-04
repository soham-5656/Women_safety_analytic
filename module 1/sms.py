import smtplib
import os  # Import os to check file existence
from email.message import EmailMessage
from twilio.rest import Client
import sys

EMAIL_ADDRESS = "sohamdawale@gmail.com"
EMAIL_PASSWORD = "jkna pnyk ztzb glny"
TO_EMAIL = "safetrck@gmail.com"

TWILIO_ACCOUNT_SID = 'AC442fefbe4fa2b53aae7f2504ceae1fe0'
TWILIO_AUTH_TOKEN = "4689f182b1f1b307069c0c1adf119a35"
TWILIO_PHONE_NUMBER = "+19283161327"
EMERGENCY_PHONE_NUMBER = "+918108465656"

print("📡 SMS script started...")

# Check for missing arguments
if len(sys.argv) < 4:
    print("⚠️ Warning: Not enough arguments provided! Using test values.")
    men_count = "1"
    women_count = "1"
    screenshot_path = "screenshot_test.jpg"
else:
    men_count = sys.argv[1]
    women_count = sys.argv[2]
    screenshot_path = sys.argv[3]

print(f"🔢 Men Count: {men_count}, Women Count: {women_count}")
print(f"📸 Screenshot Path: {screenshot_path}")

def send_email_alert(men_count, women_count, screenshot_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = "🚨 Emergency Alert!"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg.set_content(f"ALERT! {men_count} men detected. Immediate action required!")

        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as file:
                msg.add_attachment(file.read(), maintype='image', subtype='jpeg', filename="screenshot.jpg")
        else:
            print(f"⚠️ Warning: Screenshot not found at {screenshot_path}, skipping attachment.")

        print("📧 Connecting to email server...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email Alert Sent Successfully!")
    except smtplib.SMTPException as e:
        print(f"❌ Email Alert Failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def make_call_alert(men_count, women_count):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml=f'<Response><Say>Alert! {men_count} men detected. Immediate action required.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_PHONE_NUMBER
        )
        print("✅ Call Alert Sent Successfully!")
    except Exception as e:
        print(f"❌ Call Alert Failed: {e}")

print("📡 Sending Alerts...")
send_email_alert(men_count, women_count, screenshot_path)
make_call_alert(men_count, women_count)
print("📡 All Alerts Sent Successfully!")