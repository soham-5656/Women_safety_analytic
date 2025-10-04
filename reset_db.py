import sqlite3

DB_FILE = "women_safety.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# 🔴 Drop (delete) the existing table
cursor.execute("DROP TABLE IF EXISTS cases")

# ✅ Recreate the table with the correct structure
cursor.execute("""
    CREATE TABLE cases (
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

print("✅ Database reset! The 'cases' table has been recreated.")
