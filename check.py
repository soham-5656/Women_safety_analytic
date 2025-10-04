import smtplib
server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10)
server.login("sohamdawale@gmail.com", "ffer bean dwkr alcu")
print("✅ SMTP connection successful!")
server.quit()
