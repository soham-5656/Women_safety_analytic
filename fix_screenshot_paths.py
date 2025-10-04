import sqlite3
import os

DB_FILE = "women_safety.db"
SCREENSHOT_FOLDER = os.path.join(os.getcwd(), "screenshots")

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("SELECT id, screenshot_path FROM cases")
cases = cursor.fetchall()

for case_id, old_path in cases:
    new_path = old_path  # Ensure new_path is always defined

    if not os.path.isabs(old_path):  # Convert relative path to absolute
        new_path = os.path.join(os.getcwd(), old_path)

    # Check if the file actually exists in the correct place
    if not os.path.exists(new_path):
        alt_path = os.path.join(SCREENSHOT_FOLDER, os.path.basename(old_path))
        if os.path.exists(alt_path):
            new_path = alt_path

    # Update the database only if the path has changed
    if new_path != old_path:
        cursor.execute("UPDATE cases SET screenshot_path = ? WHERE id = ?", (new_path, case_id))

conn.commit()
conn.close()

print("✅ All screenshot paths have been fixed!")
