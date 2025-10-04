from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3

app = Flask(__name__)

# Initialize the database
def init_db():
    conn = sqlite3.connect("cases.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    conn = sqlite3.connect("cases.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases")
    cases = cursor.fetchall()
    conn.close()
    return render_template("index.html", cases=cases)

@app.route('/add', methods=['POST'])
def add_case():
    description = request.form['description']
    status = "Unsolved"
    conn = sqlite3.connect("cases.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cases (description, status) VALUES (?, ?)", (description, status))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

@app.route('/update/<int:case_id>/<status>')
def update_status(case_id, status):
    conn = sqlite3.connect("cases.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE cases SET status = ? WHERE id = ?", (status, case_id))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

@app.route('/delete/<int:case_id>')
def delete_case(case_id):
    conn = sqlite3.connect("cases.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
