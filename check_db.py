import sqlite3

DB_FILE = "women_safety_env/women_safety.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(cases)")  # Show table columns
columns = cursor.fetchall()

print("Table Columns:")
for column in columns:
    print(column)

conn.close()
