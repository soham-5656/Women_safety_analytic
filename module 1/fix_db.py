import sqlite3

DB_FILE = "women_safety_env/women_safety.db"

# Connect to the database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create the 'cases' table if it does not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        male_count INTEGER,
        female_count INTEGER,
        screenshot_path TEXT,
        location TEXT DEFAULT 'Unknown',
        status TEXT DEFAULT 'Pending'
    )
""")

conn.commit()
conn.close()

print("✅ Database table 'cases' is now fixed and ready!")
