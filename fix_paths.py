import sqlite3
import os

DB_FILE = "women_safety.db"
SCREENSHOT_FOLDER = os.path.join(os.getcwd(), "screenshots")  # Folder to store images

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("SELECT id, screenshot_path FROM cases")
cases = cursor.fetchall()

for case_id, filename in cases:
    if not os.path.isabs(filename):  # Convert relative filename to full path
        new_path = os.path.join(SCREENSHOT_FOLDER, filename)
        cursor.execute("UPDATE cases SET screenshot_path = ? WHERE id = ?", (new_path, case_id))

conn.commit()
conn.close()

print("✅ All screenshot paths have been updated to full paths!")
