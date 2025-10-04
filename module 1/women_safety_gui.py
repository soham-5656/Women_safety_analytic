import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import os
import csv
from PIL import Image, ImageTk
import webbrowser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

DB_FILE = "women_safety.db"

# Color Scheme
PRIMARY_COLOR = "#D63384"  # Pink
SECONDARY_COLOR = "#6C757D"  # Gray
BACKGROUND_COLOR = "#F8F9FA"  # Light Gray
TEXT_COLOR = "#333333"  # Dark Gray
WHITE = "#FFFFFF"
CARD_COLOR = "#E9ECEF"  # Light gray for cards
SUCCESS_COLOR = "#28A745"  # Green
WARNING_COLOR = "#DC3545"  # Red
HOVER_COLOR = "#FF85B3"  # Lighter pink

# Create Main Window
root = tk.Tk()
root.title("Women Safety System Dashboard")
root.geometry("1280x720")
root.configure(bg=BACKGROUND_COLOR)
root.resizable(True, True)

# Create Notebook
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Create Tab Frames
dashboard_tab = tk.Frame(notebook, bg=BACKGROUND_COLOR)
progress_tab = tk.Frame(notebook, bg=BACKGROUND_COLOR)
stats_tab = tk.Frame(notebook, bg=BACKGROUND_COLOR)

notebook.add(dashboard_tab, text=" Dashboard ")
notebook.add(progress_tab, text=" Case Progress ")
notebook.add(stats_tab, text=" Statistics ")

# Style Configuration
style = ttk.Style()
style.theme_use('clam')
style.configure("Treeview", font=('Arial', 11), rowheight=25, background=WHITE, foreground=TEXT_COLOR)
style.configure("Treeview.Heading", font=('Arial', 12, 'bold'), background=CARD_COLOR)
style.configure("TNotebook", background=BACKGROUND_COLOR)
style.configure("TNotebook.Tab", font=('Arial', 12), padding=[10, 5])

# Sidebar (At Top)
sidebar_frame = tk.Frame(dashboard_tab, bg=PRIMARY_COLOR, height=100)
sidebar_frame.pack(side="top", fill="x", pady=10, padx=10)

sidebar_header = tk.Frame(sidebar_frame, bg=PRIMARY_COLOR)
sidebar_header.pack(fill="x", pady=10, padx=10)

try:
    logo_path = r"C:\Users\Akshay\Downloads\Women-Safety-Analytics-new\Women-Safety-Analytics-main\module 1\logo.png"
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((80, 80), Image.Resampling.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(sidebar_header, image=logo_photo, bg=PRIMARY_COLOR)
        logo_label.image = logo_photo
        logo_label.pack(side="left", padx=10)
except Exception as e:
    print(f"Error loading logo: {str(e)}")

sidebar_label = tk.Label(sidebar_header, text="SAFE TRACK", font=("Arial", 18, "bold"),
                         fg=WHITE, bg=PRIMARY_COLOR)
sidebar_label.pack(side="left", pady=20)

# Main Content Frame
content_frame = tk.Frame(dashboard_tab, bg=WHITE, relief="flat", bd=1)
content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

# Title Label - Aligned to Left
title_label = tk.Label(content_frame, text="Women Safety Dashboard", font=("Arial", 20, "bold"),
                       fg=PRIMARY_COLOR, bg=WHITE, anchor="w")
title_label.pack(fill="x", pady=15, padx=10)

# Search and Filter Frame
filter_frame = tk.Frame(content_frame, bg=WHITE)
filter_frame.pack(fill="x", pady=10)

search_frame = tk.Frame(filter_frame, bg=WHITE)
search_frame.pack(side="left", padx=10)

search_label = tk.Label(search_frame, text="Search:", font=("Arial", 12, "bold"),
                        bg=WHITE, fg=TEXT_COLOR)
search_label.pack(side="left", padx=5)

search_var = StringVar()
search_entry = ttk.Entry(search_frame, textvariable=search_var, font=("Arial", 12), width=25)
search_entry.pack(side="left", padx=5)

search_by_var = StringVar(value="All Fields")
search_by_dropdown = ttk.Combobox(search_frame, textvariable=search_by_var,
                                  values=["All Fields", "Case ID", "Timestamp", "Status", "Priority"],
                                  state="readonly", width=15, font=("Arial", 11))
search_by_dropdown.pack(side="left", padx=5)

date_frame = tk.Frame(filter_frame, bg=WHITE)
date_frame.pack(side="left", padx=10)

tk.Label(date_frame, text="From:", font=("Arial", 12, "bold"), bg=WHITE, fg=TEXT_COLOR).pack(side="left", padx=5)
start_date_var = StringVar(value="YYYY-MM-DD")
start_date_entry = ttk.Entry(date_frame, textvariable=start_date_var, width=12, font=("Arial", 11))
start_date_entry.pack(side="left", padx=5)

tk.Label(date_frame, text="To:", font=("Arial", 12, "bold"), bg=WHITE, fg=TEXT_COLOR).pack(side="left", padx=5)
end_date_var = StringVar(value="YYYY-MM-DD")
end_date_entry = ttk.Entry(date_frame, textvariable=end_date_var, width=12, font=("Arial", 11))
end_date_entry.pack(side="left", padx=5)


# Buttons
def create_button(parent, text, command, bg_color=PRIMARY_COLOR):
    btn = tk.Button(parent, text=text, font=("Arial", 11, "bold"), fg=WHITE, bg=bg_color,
                    activebackground=HOVER_COLOR, padx=12, pady=6, command=command, relief="flat",
                    cursor="hand2")
    btn.bind("<Enter>", lambda e: btn.config(bg=HOVER_COLOR))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg_color))
    return btn


search_button = create_button(search_frame, "🔍 Search", lambda: search_cases())
search_button.pack(side="left", padx=5)

clear_button = create_button(search_frame, "❌ Clear", lambda: clear_search(), bg_color=SECONDARY_COLOR)
clear_button.pack(side="left", padx=5)

# Table Frame
frame = tk.Frame(content_frame, bg=WHITE)
frame.pack(fill="both", expand=True, pady=10)

no_data_label = tk.Label(frame, text="No data available", font=("Arial", 14, "italic"),
                         fg=WARNING_COLOR, bg=WHITE)

scroll_y = tk.Scrollbar(frame, orient="vertical")
scroll_x = tk.Scrollbar(frame, orient="horizontal")

case_table = ttk.Treeview(frame, columns=("ID", "Timestamp", "Male", "Female", "Screenshot", "Location",
                                          "Status", "Priority", "Notes"), show="headings",
                          yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

scroll_y.config(command=case_table.yview)
scroll_y.pack(side="right", fill="y")
scroll_x.config(command=case_table.xview)
scroll_x.pack(side="bottom", fill="x")

case_table.heading("ID", text="Case ID")
case_table.heading("Timestamp", text="Timestamp")
case_table.heading("Male", text="Male Count")
case_table.heading("Female", text="Female Count")
case_table.heading("Screenshot", text="Screenshot")
case_table.heading("Location", text="Location")
case_table.heading("Status", text="Status")
case_table.heading("Priority", text="Priority")
case_table.heading("Notes", text="Notes")

case_table.column("ID", width=60, anchor="center")
case_table.column("Timestamp", width=160, anchor="center")
case_table.column("Male", width=90, anchor="center")
case_table.column("Female", width=90, anchor="center")
case_table.column("Screenshot", width=100, anchor="center")
case_table.column("Location", width=100, anchor="center")
case_table.column("Status", width=90, anchor="center")
case_table.column("Priority", width=90, anchor="center")
case_table.column("Notes", width=120, anchor="center")

case_table.pack(fill="both", expand=True)

# Status Bar
status_bar = tk.Label(content_frame, text="Ready", font=("Arial", 10), bg=CARD_COLOR,
                      fg=TEXT_COLOR, bd=1, relief="sunken", anchor="w", padx=10)
status_bar.pack(side="bottom", fill="x")

# Action Buttons
button_frame = tk.Frame(content_frame, bg=WHITE)
button_frame.pack(side="bottom", fill="x", pady=10)

refresh_button = create_button(button_frame, "🔄 Refresh", lambda: load_cases())
refresh_button.pack(side="left", padx=5)

resolve_button = create_button(button_frame, "✔ Resolve", lambda: update_status())
resolve_button.pack(side="left", padx=5)

export_button = create_button(button_frame, "📤 Export CSV", lambda: export_to_csv())
export_button.pack(side="left", padx=5)

set_priority_button = create_button(button_frame, "⚡ Set Priority", lambda: set_priority())
set_priority_button.pack(side="left", padx=5)

add_note_button = create_button(button_frame, "✎ Add Note", lambda: add_case_note())
add_note_button.pack(side="left", padx=5)

delete_button = create_button(button_frame, "🗑 Delete All", lambda: delete_all_cases(), bg_color=WARNING_COLOR)
delete_button.pack(side="left", padx=5)

# Progress Tab
progress_tab.configure(bg=WHITE)

progress_header_frame = tk.Frame(progress_tab, bg=WHITE, padx=20, pady=15)
progress_header_frame.pack(fill="x")

progress_title = tk.Label(progress_header_frame, text="Case Progress Tracker", font=("Arial", 20, "bold"),
                          fg=PRIMARY_COLOR, bg=WHITE)
progress_title.pack(side="left")

progress_filter_frame = tk.Frame(progress_header_frame, bg=WHITE)
progress_filter_frame.pack(side="right")

tk.Label(progress_filter_frame, text="Filter by:", font=("Arial", 12, "bold"), bg=WHITE,
         fg=TEXT_COLOR).pack(side="left", padx=5)
progress_status_var = StringVar(value="All Status")
progress_status_dropdown = ttk.Combobox(progress_filter_frame, textvariable=progress_status_var,
                                        values=["All Status", "Open", "In Progress", "Resolved"],
                                        state="readonly", width=12, font=("Arial", 11))
progress_status_dropdown.pack(side="left", padx=5)

progress_priority_var = StringVar(value="All Priority")
progress_priority_dropdown = ttk.Combobox(progress_filter_frame, textvariable=progress_priority_var,
                                          values=["All Priority", "Low", "Normal", "High", "Urgent"],
                                          state="readonly", width=12, font=("Arial", 11))
progress_priority_dropdown.pack(side="left", padx=5)

progress_filter_button = create_button(progress_filter_frame, "Filter", lambda: filter_progress_cases())
progress_filter_button.pack(side="left", padx=5)

progress_clear_filter_button = create_button(progress_filter_frame, "Clear", lambda: reset_progress_filters(),
                                             bg_color=SECONDARY_COLOR)
progress_clear_filter_button.pack(side="left", padx=5)

progress_content = tk.PanedWindow(progress_tab, orient="horizontal", bg=WHITE, sashwidth=4, sashrelief="raised")
progress_content.pack(fill="both", expand=True, padx=20, pady=(0, 20))

progress_list_frame = tk.Frame(progress_content, bg=WHITE, bd=1, relief="solid")
progress_content.add(progress_list_frame, width=550)

progress_list_title = tk.Label(progress_list_frame, text="Active Cases", font=("Arial", 14, "bold"),
                               fg=TEXT_COLOR, bg=WHITE)
progress_list_title.pack(anchor="w", padx=15, pady=10)

progress_scroll_y = tk.Scrollbar(progress_list_frame, orient="vertical")
progress_scroll_x = tk.Scrollbar(progress_list_frame, orient="horizontal")

progress_no_data_label = tk.Label(progress_list_frame, text="No cases available", font=("Arial", 14, "italic"),
                                  fg=WARNING_COLOR, bg=WHITE)

progress_table = ttk.Treeview(progress_list_frame, columns=("ID", "Date", "Status", "Priority", "Progress"),
                              show="headings", yscrollcommand=progress_scroll_y.set,
                              xscrollcommand=scroll_x.set)

progress_scroll_y.config(command=progress_table.yview)
progress_scroll_y.pack(side="right", fill="y")
progress_scroll_x.config(command=progress_table.xview)
progress_scroll_x.pack(side="bottom", fill="x")

progress_table.heading("ID", text="Case ID")
progress_table.heading("Date", text="Date")
progress_table.heading("Status", text="Status")
progress_table.heading("Priority", text="Priority")
progress_table.heading("Progress", text="Progress")

progress_table.column("ID", width=60, anchor="center")
progress_table.column("Date", width=110, anchor="center")
progress_table.column("Status", width=100, anchor="center")
progress_table.column("Priority", width=100, anchor="center")
progress_table.column("Progress", width=200, anchor="center")

progress_table.pack(fill="both", expand=True)

case_detail_frame = tk.Frame(progress_content, bg=WHITE, bd=1, relief="solid")
progress_content.add(case_detail_frame)

detail_header = tk.Frame(case_detail_frame, bg=CARD_COLOR, padx=15, pady=10)
detail_header.pack(fill="x")

detail_title = tk.Label(detail_header, text="Case Details", font=("Arial", 16, "bold"),
                        fg=PRIMARY_COLOR, bg=CARD_COLOR)
detail_title.pack(side="left")

detail_content_frame = tk.Frame(case_detail_frame, bg=WHITE, padx=15, pady=15)
detail_content_frame.pack(fill="both", expand=True)

detail_labels = {
    "case_id": tk.Label(detail_content_frame, text="Case ID:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "timestamp": tk.Label(detail_content_frame, text="Timestamp:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "status": tk.Label(detail_content_frame, text="Status:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "priority": tk.Label(detail_content_frame, text="Priority:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "male_count": tk.Label(detail_content_frame, text="Male Count:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "female_count": tk.Label(detail_content_frame, text="Female Count:", font=("Arial", 12, "bold"), bg=WHITE,
                             anchor="w"),
    "progress": tk.Label(detail_content_frame, text="Progress:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w"),
    "notes": tk.Label(detail_content_frame, text="Notes:", font=("Arial", 12, "bold"), bg=WHITE, anchor="w")
}

detail_values = {
    key: tk.Label(detail_content_frame, text="", font=("Arial", 12), bg=WHITE, anchor="w")
    for key in detail_labels
}

for i, (key, label) in enumerate(detail_labels.items()):
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    detail_values[key].grid(row=i, column=1, padx=10, pady=5, sticky="w")

progress_bar_frame = tk.Frame(detail_content_frame, bg=WHITE)
progress_bar_frame.grid(row=6, column=1, padx=10, pady=5, sticky="w")

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(progress_bar_frame, variable=progress_var, maximum=100, length=200,
                               style="Horizontal.TProgressbar")
progress_bar.pack(side="left")

progress_percentage = tk.Label(progress_bar_frame, text="0%", font=("Arial", 12), bg=WHITE)
progress_percentage.pack(side="left", padx=5)

notes_text = tk.Text(detail_content_frame, height=5, width=40, wrap="word", font=("Arial", 11),
                     bg=CARD_COLOR, bd=1, relief="solid")
notes_text.grid(row=7, column=1, padx=10, pady=5, sticky="nsew")
notes_text.config(state="disabled")
detail_content_frame.rowconfigure(7, weight=1)

action_frame = tk.Frame(case_detail_frame, bg=CARD_COLOR, padx=15, pady=15)
action_frame.pack(fill="x", side="bottom")

action_row1 = tk.Frame(action_frame, bg=CARD_COLOR)
action_row1.pack(fill="x", pady=(0, 8))

action_row2 = tk.Frame(action_frame, bg=CARD_COLOR)
action_row2.pack(fill="x")

update_progress_button = create_button(action_row1, "📊 Update Progress", lambda: update_progress_dialog())
update_progress_button.pack(side="left", padx=5)

update_status_button = create_button(action_row1, "📝 Update Status", lambda: update_case_status())
update_status_button.pack(side="left", padx=5)

update_priority_button = create_button(action_row1, "⚡ Set Priority", lambda: update_case_priority())
update_priority_button.pack(side="left", padx=5)

edit_notes_button = create_button(action_row2, "✎ Edit Notes", lambda: toggle_notes_editing())
edit_notes_button.pack(side="left", padx=5)

view_screenshot_button = create_button(action_row2, "📸 View Screenshot", lambda: view_screenshot_detail())
view_screenshot_button.pack(side="left", padx=5)

view_location_button = create_button(action_row2, "📍 View Location", lambda: view_location_detail())
view_location_button.pack(side="left", padx=5)

# Statistics Tab
stats_frame = tk.Frame(stats_tab, bg=WHITE)
stats_frame.pack(fill="both", expand=True, padx=20, pady=20)

stats_header = tk.Frame(stats_frame, bg=WHITE)
stats_header.pack(fill="x", pady=10)

stats_title = tk.Label(stats_header, text="Case Statistics", font=("Arial", 20, "bold"),
                       fg=PRIMARY_COLOR, bg=WHITE)
stats_title.pack(side="left")

refresh_stats_button = create_button(stats_header, "🔄 Refresh", lambda: update_statistics())
refresh_stats_button.pack(side="right", padx=10)

charts_frame = tk.Frame(stats_frame, bg=WHITE)
charts_frame.pack(fill="both", expand=True, padx=20, pady=10)

stats_data = {
    "total_cases": 0,
    "resolved_cases": 0,
    "pending_cases": 0,
    "avg_progress": 0.0,
    "high_priority": 0,
    "recent_cases": 0,
    "high_priority_resolved": 0,
    "high_priority_pending": 0,
    "cases_by_date": {}
}


def update_charts():
    for widget in charts_frame.winfo_children():
        widget.destroy()

    main_container = tk.Frame(charts_frame, bg=WHITE)
    main_container.pack(fill="both", expand=True)

    # keep/try styles but don't crash if unavailable
    try:
        plt.style.use('ggplot')
    except Exception:
        plt.style.use('default')

    row1_frame = tk.Frame(main_container, bg=WHITE, bd=1, relief="solid")
    row1_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # --- Pie (Case Status Distribution) ---
    fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)
    # Ensure integers and handle None
    resolved = int(stats_data.get("resolved_cases") or 0)
    pending = int(stats_data.get("pending_cases") or 0)
    sizes = [resolved, pending]
    labels = ['Resolved', 'Pending']

    total = sum(sizes)
    if total == 0:
        # no data -> show placeholder text instead of a pie (avoids division by zero)
        ax1.text(0.5, 0.5, "No case data", ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.axis('off')
    else:
        # Safe plotting - ensure sizes are non-negative
        sizes = [max(0, s) for s in sizes]
        explode = (0.05, 0)
        ax1.pie(sizes, explode=explode, labels=labels, colors=[SUCCESS_COLOR, WARNING_COLOR],
                autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 10})
        ax1.axis('equal')
    ax1.set_title("Case Status Distribution", fontsize=12, pad=15, color=TEXT_COLOR)

    canvas1 = FigureCanvasTkAgg(fig1, master=row1_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(pady=10, padx=10)

    # --- Priority bar chart ---
    row1_frame2 = tk.Frame(main_container, bg=WHITE, bd=1, relief="solid")
    row1_frame2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    fig4, ax4 = plt.subplots(figsize=(5, 4), dpi=100)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT priority, COUNT(*) FROM cases GROUP BY priority")
    priority_data = cur.fetchall()
    conn.close()

    priorities = ['Low', 'Normal', 'High', 'Urgent']
    counts = [0] * len(priorities)
    for priority, count in priority_data:
        try:
            if priority in priorities:
                counts[priorities.index(priority)] = int(count)
        except Exception:
            pass

    # If all zeros, still draw bars but annotate 0
    bars = ax4.bar(priorities, counts, edgecolor='black')
    ax4.set_title("Cases by Priority", fontsize=12, pad=15)
    ax4.set_ylabel("Number of Cases", fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    canvas4 = FigureCanvasTkAgg(fig4, master=row1_frame2)
    canvas4.draw()
    canvas4.get_tk_widget().pack(pady=10, padx=10)

    # --- Polar progress chart (average progress) ---
    row2_frame = tk.Frame(main_container, bg=WHITE, bd=1, relief="solid")
    row2_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    fig3, ax3 = plt.subplots(figsize=(5, 4), subplot_kw={'projection': 'polar'}, dpi=100)
    progress = float(stats_data.get("avg_progress") or 0.0)
    # Clamp progress to 0..100
    progress = max(0.0, min(100.0, progress))
    theta = np.linspace(0, np.pi, 100)
    radius = np.ones_like(theta)
    ax3.fill(theta, radius, alpha=0.15)
    progress_angle = (progress / 100.0) * np.pi
    theta_progress = np.linspace(0, progress_angle, 100)
    ax3.fill(theta_progress, np.ones_like(theta_progress), alpha=0.7)
    ax3.set_yticklabels([])
    ax3.set_xticks(np.linspace(0, np.pi, 5))
    ax3.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    ax3.set_title(f"Avg Progress: {progress:.1f}%", fontsize=12, pad=20)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    plt.subplots_adjust(bottom=0.2)

    canvas3 = FigureCanvasTkAgg(fig3, master=row2_frame)
    canvas3.draw()
    canvas3.get_tk_widget().pack(pady=10, padx=10)

    # --- Line chart for cases last 7 days ---
    row2_frame2 = tk.Frame(main_container, bg=WHITE, bd=1, relief="solid")
    row2_frame2.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    fig6, ax6 = plt.subplots(figsize=(5, 4), dpi=100)
    dates = list(stats_data.get("cases_by_date", {}).keys())
    counts = list(stats_data.get("cases_by_date", {}).values())

    # If no data, show placeholder
    if not any(counts):
        ax6.text(0.5, 0.5, "No recent cases", ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
    else:
        # plotting string dates is fine; rotate x labels
        ax6.plot(dates, counts, marker='o', linewidth=2)
        ax6.set_xlabel("Date", fontsize=10)
        ax6.set_ylabel("Number of Cases", fontsize=10)
        ax6.tick_params(axis='x', rotation=45)
        for i, count in enumerate(counts):
            ax6.annotate(str(count), (dates[i], counts[i]), textcoords="offset points",
                         xytext=(0, 5), ha='center', fontsize=9)
    ax6.set_title(f"Cases Last 7 Days: {stats_data.get('recent_cases', 0)}", fontsize=12, pad=15)
    plt.tight_layout()

    canvas6 = FigureCanvasTkAgg(fig6, master=row2_frame2)
    canvas6.draw()
    canvas6.get_tk_widget().pack(pady=10, padx=10)

    main_container.grid_columnconfigure((0, 1), weight=1)
    main_container.grid_rowconfigure((0, 1), weight=1)



# Global Variables
current_case = {
    "id": None, "screenshot": None, "location": None, "progress": 0, "status": None,
    "priority": None, "notes": None, "timestamp": None, "male_count": None, "female_count": None
}
notes_editing = False


def initialize_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create the table if it doesn't exist with all required columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            male_count INTEGER,
            female_count INTEGER,
            screenshot_path TEXT,
            location TEXT,
            status TEXT,
            progress INTEGER DEFAULT 0,
            priority TEXT DEFAULT 'Normal',
            notes TEXT DEFAULT ''
        )
    """)

    # Check and add missing columns
    cursor.execute("PRAGMA table_info(cases)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'progress' not in columns:
        cursor.execute("ALTER TABLE cases ADD COLUMN progress INTEGER DEFAULT 0")
    if 'priority' not in columns:
        cursor.execute("ALTER TABLE cases ADD COLUMN priority TEXT DEFAULT 'Normal'")
    if 'notes' not in columns:
        cursor.execute("ALTER TABLE cases ADD COLUMN notes TEXT DEFAULT ''")
    if 'resolved_timestamp' not in columns:
        cursor.execute("ALTER TABLE cases ADD COLUMN resolved_timestamp TEXT DEFAULT NULL")

    conn.commit()
    conn.close()


def view_screenshot(image_path):
    if os.path.exists(image_path):
        os.startfile(image_path)
    else:
        messagebox.showerror("Error", f"Screenshot not found at: {image_path}")


def view_screenshot_detail():
    if current_case["screenshot"] and os.path.exists(current_case["screenshot"]):
        view_screenshot(current_case["screenshot"])
    else:
        messagebox.showerror("Error", "No screenshot available")


def open_location(location):
    if location and location != "No Location":
        try:
            latitude, longitude = location.split(",")
            webbrowser.open(f"https://www.google.com/maps?q={latitude.strip()},{longitude.strip()}")
        except ValueError:
            messagebox.showerror("Error", "Invalid location format")
    else:
        messagebox.showerror("Error", "No valid location")


def view_location_detail():
    if current_case["location"] and current_case["location"] != "No Location":
        open_location(current_case["location"])
    else:
        messagebox.showerror("Error", "No location data")


def toggle_no_data_message(show=False):
    if show:
        case_table.pack_forget()
        no_data_label.pack(expand=True, fill="both")
    else:
        no_data_label.pack_forget()
        case_table.pack(fill="both", expand=True)


def toggle_progress_no_data(show=False):
    if show:
        progress_table.pack_forget()
        progress_no_data_label.pack(expand=True, fill="both")
    else:
        progress_no_data_label.pack_forget()
        progress_table.pack(fill="both", expand=True)


def load_cases():
    case_table.delete(*case_table.get_children())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes FROM cases")
    rows = cursor.fetchall()
    if not rows:
        toggle_no_data_message(True)
    else:
        toggle_no_data_message(False)
        for row in rows:
            case_id, timestamp, male, female, screenshot, location, status, progress, priority, notes = row
            screenshot_text = "View" if os.path.exists(screenshot) else "Missing"
            location_text = "Map" if location and "," in location else "None"
            notes_text = "View" if notes else "None"
            case_table.insert("", "end", values=(case_id, timestamp, male, female, screenshot_text,
                                                 location_text, status, priority, notes_text),
                              tags=(screenshot, location, str(progress)))
    conn.close()
    update_status_bar()
    load_progress_cases()
    update_statistics()


def load_progress_cases():
    progress_table.delete(*progress_table.get_children())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes FROM cases"
    cursor.execute(query)
    rows = cursor.fetchall()
    if not rows:
        toggle_progress_no_data(True)
        clear_case_details()
    else:
        toggle_progress_no_data(False)
        for row in rows:
            case_id, timestamp, male, female, screenshot, location, status, progress, priority, notes = row
            date_display = timestamp.split(" ")[0] if " " in timestamp else timestamp
            progress_display = f"{progress}%"
            progress_table.insert("", "end", values=(case_id, date_display, status, priority, progress_display),
                                  tags=(str(case_id), timestamp, str(male), str(female), screenshot, location,
                                        status, str(progress), priority, notes))
    conn.close()


def filter_progress_cases():
    progress_table.delete(*progress_table.get_children())
    status_filter = progress_status_var.get()
    priority_filter = progress_priority_var.get()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query = """SELECT id, timestamp, male_count, female_count, screenshot_path, 
               location, status, progress, priority, notes 
               FROM cases WHERE 1=1"""
    params = []

    if status_filter != "All Status":
        query += " AND status = ?"
        params.append(status_filter)
    if priority_filter != "All Priority":
        query += " AND priority = ?"
        params.append(priority_filter)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        toggle_progress_no_data(True)
        clear_case_details()
    else:
        toggle_progress_no_data(False)
        for row in rows:
            case_id, timestamp, male, female, screenshot, location, status, progress, priority, notes = row
            date_display = timestamp.split(" ")[0] if " " in timestamp else timestamp
            progress_display = f"{progress}%"
            progress_table.insert("", "end", values=(case_id, date_display, status, priority, progress_display),
                                  tags=(str(case_id), timestamp, str(male), str(female), screenshot, location,
                                        status, str(progress), priority, notes))
    conn.close()


def reset_progress_filters():
    progress_status_var.set("All Status")
    progress_priority_var.set("All Priority")
    load_progress_cases()


def clear_search():
    search_var.set("")
    start_date_var.set("YYYY-MM-DD")
    end_date_var.set("YYYY-MM-DD")
    load_cases()


def search_cases():
    search_term = search_var.get().strip().lower()
    search_by = search_by_var.get()
    start_date = start_date_var.get()
    end_date = end_date_var.get()
    case_table.delete(*case_table.get_children())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes FROM cases WHERE 1=1"
    params = []
    if search_term:
        if search_by == "Case ID":
            try:
                query += " AND id = ?"
                params.append(int(search_term))
            except ValueError:
                messagebox.showwarning("Invalid Input", "Case ID must be a number")
                return
        elif search_by == "Timestamp":
            query += " AND LOWER(timestamp) LIKE ?"
            params.append(f"%{search_term}%")
        elif search_by == "Status":
            query += " AND LOWER(status) LIKE ?"
            params.append(f"%{search_term}%")
        elif search_by == "Priority":
            query += " AND LOWER(priority) LIKE ?"
            params.append(f"%{search_term}%")
        else:
            query += " AND (LOWER(timestamp) LIKE ? OR LOWER(status) LIKE ? OR LOWER(priority) LIKE ? OR id = ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"])
            try:
                params.append(int(search_term))
            except ValueError:
                params.pop()
    if start_date != "YYYY-MM-DD" and end_date != "YYYY-MM-DD":
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        except ValueError:
            messagebox.showwarning("Invalid Date", "Please use YYYY-MM-DD format")
            return
    cursor.execute(query, params)
    rows = cursor.fetchall()
    if not rows:
        toggle_no_data_message(True)
    else:
        toggle_no_data_message(False)
        for row in rows:
            case_id, timestamp, male, female, screenshot, location, status, progress, priority, notes = row
            screenshot_text = "View" if os.path.exists(screenshot) else "Missing"
            location_text = "Map" if location and "," in location else "None"
            notes_text = "View" if notes else "None"
            case_table.insert("", "end", values=(case_id, timestamp, male, female, screenshot_text,
                                                 location_text, status, priority, notes_text),
                              tags=(screenshot, location, str(progress)))
    conn.close()
    update_status_bar()


def update_status():
    selected = case_table.selection()
    if not selected:
        messagebox.showwarning("Warning", "Please select a case")
        return
    case_id = case_table.item(selected)["values"][0]
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE cases SET status = 'Resolved', progress = 100, resolved_timestamp = ? WHERE id = ?",
                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), case_id))
    conn.commit()
    conn.close()
    load_cases()
    messagebox.showinfo("Success", f"Case ID {case_id} marked as resolved")


def update_progress_dialog():
    if not current_case["id"]:
        messagebox.showwarning("Warning", "No case selected")
        return
    dialog = tk.Toplevel(root)
    dialog.title("Update Progress")
    dialog.geometry("400x200")
    dialog.configure(bg=WHITE)
    dialog.transient(root)
    dialog.grab_set()
    tk.Label(dialog, text=f"Case #{current_case['id']}", font=("Arial", 14, "bold"),
             bg=WHITE, fg=PRIMARY_COLOR).pack(pady=15)
    frame = tk.Frame(dialog, bg=WHITE)
    frame.pack(pady=10)
    tk.Label(frame, text="Progress (0-100%):", font=("Arial", 12), bg=WHITE).pack(side="left", padx=10)
    progress_value = tk.StringVar(value=str(current_case["progress"]))
    progress_entry = ttk.Spinbox(frame, from_=0, to=100, textvariable=progress_value, width=5,
                                 font=("Arial", 11))
    progress_entry.pack(side="left", padx=10)

    def save():
        try:
            progress = int(progress_value.get())
            if 0 <= progress <= 100:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                status = "Resolved" if progress == 100 else "In Progress"
                resolved_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if progress == 100 else None
                cursor.execute("UPDATE cases SET progress = ?, status = ?, resolved_timestamp = ? WHERE id = ?",
                               (progress, status, resolved_timestamp, current_case["id"]))
                conn.commit()
                conn.close()
                load_cases()
                load_case_details(current_case["id"])
                dialog.destroy()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")

    tk.Button(dialog, text="Save", bg=PRIMARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=save, relief="flat").pack(pady=10)
    tk.Button(dialog, text="Cancel", bg=SECONDARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=dialog.destroy, relief="flat").pack(pady=5)


def update_case_status():
    if not current_case["id"]:
        messagebox.showwarning("Warning", "No case selected")
        return
    dialog = tk.Toplevel(root)
    dialog.title("Update Status")
    dialog.geometry("350x180")
    dialog.configure(bg=WHITE)
    dialog.transient(root)
    dialog.grab_set()
    tk.Label(dialog, text=f"Set Status for Case #{current_case['id']}", font=("Arial", 14, "bold"),
             bg=WHITE, fg=PRIMARY_COLOR).pack(pady=15)
    status_var = StringVar(value=current_case["status"] or "Open")
    ttk.Combobox(dialog, textvariable=status_var, values=["Open", "In Progress", "Resolved"],
                 state="readonly", font=("Arial", 11)).pack(pady=10)

    def save():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        new_status = status_var.get()
        resolved_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status == "Resolved" else None
        cursor.execute("UPDATE cases SET status = ?, resolved_timestamp = ? WHERE id = ?",
                       (new_status, resolved_timestamp, current_case["id"]))
        if new_status == "Resolved":
            cursor.execute("UPDATE cases SET progress = 100 WHERE id = ?", (current_case["id"],))
        conn.commit()
        conn.close()
        load_cases()
        load_case_details(current_case["id"])
        dialog.destroy()

    tk.Button(dialog, text="Save", bg=PRIMARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=save, relief="flat").pack(pady=10)


def update_case_priority():
    if not current_case["id"]:
        messagebox.showwarning("Warning", "No case selected")
        return
    dialog = tk.Toplevel(root)
    dialog.title("Set Priority")
    dialog.geometry("350x180")
    dialog.configure(bg=WHITE)
    dialog.transient(root)
    dialog.grab_set()
    tk.Label(dialog, text=f"Set Priority for Case #{current_case['id']}", font=("Arial", 14, "bold"),
             bg=WHITE, fg=PRIMARY_COLOR).pack(pady=15)
    priority_var = StringVar(value=current_case["priority"] or "Normal")
    ttk.Combobox(dialog, textvariable=priority_var, values=["Low", "Normal", "High", "Urgent"],
                 state="readonly", font=("Arial", 11)).pack(pady=10)

    def save():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE cases SET priority = ? WHERE id = ?", (priority_var.get(), current_case["id"]))
        conn.commit()
        conn.close()
        load_cases()
        load_case_details(current_case["id"])
        dialog.destroy()

    tk.Button(dialog, text="Save", bg=PRIMARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=save, relief="flat").pack(pady=10)


def toggle_notes_editing():
    global notes_editing
    if not current_case["id"]:
        messagebox.showwarning("Warning", "No case selected")
        return
    if not notes_editing:
        notes_text.config(state="normal")
        edit_notes_button.config(text="💾 Save Notes")
        notes_editing = True
    else:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        new_notes = notes_text.get("1.0", tk.END).strip()
        cursor.execute("UPDATE cases SET notes = ? WHERE id = ?", (new_notes, current_case["id"]))
        conn.commit()
        conn.close()
        notes_text.config(state="disabled")
        edit_notes_button.config(text="✎ Edit Notes")
        notes_editing = False
        load_cases()
        load_case_details(current_case["id"])


def set_priority():
    selected = case_table.selection()
    if not selected:
        messagebox.showwarning("Warning", "Please select a case")
        return
    case_id = case_table.item(selected)["values"][0]
    dialog = tk.Toplevel(root)
    dialog.title("Set Priority")
    dialog.geometry("350x180")
    dialog.configure(bg=WHITE)
    dialog.transient(root)
    dialog.grab_set()
    tk.Label(dialog, text=f"Set Priority for Case #{case_id}", font=("Arial", 14, "bold"),
             bg=WHITE, fg=PRIMARY_COLOR).pack(pady=15)
    priority_var = StringVar(value="Normal")
    ttk.Combobox(dialog, textvariable=priority_var, values=["Low", "Normal", "High", "Urgent"],
                 state="readonly", font=("Arial", 11)).pack(pady=10)

    def save():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE cases SET priority = ? WHERE id = ?", (priority_var.get(), case_id))
        conn.commit()
        conn.close()
        load_cases()
        dialog.destroy()

    tk.Button(dialog, text="Save", bg=PRIMARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=save, relief="flat").pack(pady=10)


def add_case_note():
    selected = case_table.selection()
    if not selected:
        messagebox.showwarning("Warning", "Please select a case")
        return
    case_id = case_table.item(selected)["values"][0]
    dialog = tk.Toplevel(root)
    dialog.title("Case Notes")
    dialog.geometry("450x350")
    dialog.configure(bg=WHITE)
    dialog.transient(root)
    dialog.grab_set()
    tk.Label(dialog, text=f"Notes for Case #{case_id}", font=("Arial", 14, "bold"),
             bg=WHITE, fg=PRIMARY_COLOR).pack(pady=15)
    note_text = tk.Text(dialog, height=10, width=40, font=("Arial", 11), wrap="word")
    note_text.pack(pady=10, padx=20)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT notes FROM cases WHERE id = ?", (case_id,))
    current_note = cursor.fetchone()[0]
    if current_note:
        note_text.insert("1.0", current_note)
    conn.close()

    def save():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE cases SET notes = ? WHERE id = ?", (note_text.get("1.0", tk.END).strip(), case_id))
        conn.commit()
        conn.close()
        load_cases()
        if current_case["id"] == case_id:
            load_case_details(case_id)
        dialog.destroy()

    tk.Button(dialog, text="Save", bg=PRIMARY_COLOR, fg=WHITE, font=("Arial", 11, "bold"),
              command=save, relief="flat").pack(pady=10)


def export_to_csv():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes FROM cases")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        messagebox.showwarning("Warning", "No data to export")
        return
    with open("cases_export.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp", "Male Count", "Female Count", "Screenshot", "Location",
                         "Status", "Progress", "Priority", "Notes"])
        writer.writerows(rows)
    messagebox.showinfo("Success", "Data exported to cases_export.csv")


def delete_all_cases():
    if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete all cases?"):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS cases")
        conn.commit()
        conn.close()
        initialize_db()
        load_cases()
        clear_case_details()
        messagebox.showinfo("Success", "All cases deleted successfully")


def load_case_details(case_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes FROM cases WHERE id = ?",
        (case_id,))
    row = cursor.fetchone()
    if row:
        case_id, timestamp, male_count, female_count, screenshot_path, location, status, progress, priority, notes = row
        detail_values["case_id"].config(text=case_id)
        detail_values["timestamp"].config(text=timestamp)
        detail_values["status"].config(text=status)
        detail_values["priority"].config(text=priority)
        detail_values["male_count"].config(text=male_count)
        detail_values["female_count"].config(text=female_count)
        detail_values["progress"].config(text=f"{progress}%")
        progress_var.set(progress)
        progress_percentage.config(text=f"{progress}%")
        notes_text.config(state="normal")
        notes_text.delete("1.0", tk.END)
        notes_text.insert("1.0", notes if notes else "No notes available")
        notes_text.config(state="disabled")
        current_case.update({
            "id": case_id, "screenshot": screenshot_path, "location": location, "progress": progress,
            "status": status, "priority": priority, "notes": notes, "timestamp": timestamp,
            "male_count": male_count, "female_count": female_count
        })
    conn.close()


def clear_case_details():
    for key in detail_values:
        detail_values[key].config(text="")
    progress_var.set(0)
    progress_percentage.config(text="0%")
    notes_text.config(state="normal")
    notes_text.delete("1.0", tk.END)
    notes_text.insert("1.0", "No case selected")
    notes_text.config(state="disabled")
    current_case.update({
        "id": None, "screenshot": None, "location": None, "progress": 0, "status": None,
        "priority": None, "notes": None, "timestamp": None, "male_count": None, "female_count": None
    })


def update_statistics():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM cases")
    stats_data["total_cases"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE status = 'Resolved'")
    stats_data["resolved_cases"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE status != 'Resolved'")
    stats_data["pending_cases"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(progress) FROM cases")
    stats_data["avg_progress"] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM cases WHERE priority IN ('High', 'Urgent')")
    stats_data["high_priority"] = cursor.fetchone()[0]

    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM cases WHERE timestamp >= ?", (seven_days_ago,))
    stats_data["recent_cases"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE priority IN ('High', 'Urgent') AND status = 'Resolved'")
    stats_data["high_priority_resolved"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cases WHERE priority IN ('High', 'Urgent') AND status != 'Resolved'")
    stats_data["high_priority_pending"] = cursor.fetchone()[0]

    cursor.execute("SELECT timestamp FROM cases WHERE timestamp >= ?", (seven_days_ago,))
    rows = cursor.fetchall()
    cases_by_date = {}
    for i in range(7, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        cases_by_date[date] = 0
    for row in rows:
        timestamp = row[0]
        date = timestamp.split(" ")[0] if " " in timestamp else timestamp
        if date in cases_by_date:
            cases_by_date[date] += 1
    stats_data["cases_by_date"] = cases_by_date

    conn.close()
    update_charts()


def on_item_click(event):
    selected = case_table.selection()
    if selected:
        screenshot_path, location, _ = case_table.item(selected, "tags")
        column = case_table.identify_column(event.x)
        if column == "#5":
            view_screenshot(screenshot_path)
        elif column == "#6":
            open_location(location)
        elif column == "#9":
            add_case_note()


def on_progress_item_click(event):
    selected = progress_table.selection()
    if selected:
        case_id = progress_table.item(selected, "tags")[0]
        load_case_details(case_id)


def update_status_bar():
    item_count = len(case_table.get_children())
    status_text = f"Displaying {item_count} case{'s' if item_count != 1 else ''}"
    if search_var.get().strip() or start_date_var.get() != "YYYY-MM-DD":
        status_text += " (Filtered)"
    status_bar.config(text=status_text)


case_table.bind("<ButtonRelease-1>", on_item_click)
progress_table.bind("<ButtonRelease-1>", on_progress_item_click)
search_entry.bind("<Return>", lambda event: search_cases())

initialize_db()
load_cases()
clear_case_details()

root.mainloop()