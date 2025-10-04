import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# Color Scheme (Matching your dashboard)
PRIMARY_COLOR = "#D63384"  # Pink
SECONDARY_COLOR = "#6C757D"  # Gray
BACKGROUND_COLOR = "#F8F9FA"  # Light Gray
TEXT_COLOR = "#333333"  # Dark Gray
WHITE = "#FFFFFF"
HOVER_COLOR = "#FF85B3"  # Lighter pink

# Admin Credentials
ADMIN_ID = "admin"
ADMIN_PASSWORD = "admin123"

# Women Safety GUI file path
GUI_FILE = os.path.abspath("C:\Users\Soham\Desktop\final year\module 1\women_safety_gui.py")

def login():
    """Handles login authentication."""
    user_id = id_entry.get().strip()
    password = password_entry.get().strip()

    if user_id == ADMIN_ID and password == ADMIN_PASSWORD:
        messagebox.showinfo("Login Successful", "Welcome to the Women Safety System!",
                           parent=login_window)
        login_window.destroy()  # Close login window

        # Open the Women Safety GUI file
        try:
            subprocess.run([sys.executable, GUI_FILE], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open dashboard:\n{e}",
                                parent=None)  # No parent since login_window is destroyed
    else:
        messagebox.showerror("Login Failed", "Invalid ID or Password!",
                            parent=login_window)

def create_button(parent, text, command):
    """Creates a styled button with hover effect."""
    btn = tk.Button(parent, text=text, font=("Arial", 11, "bold"), fg=WHITE, bg=PRIMARY_COLOR,
                    activebackground=HOVER_COLOR, padx=20, pady=8, command=command,
                    relief="flat", cursor="hand2")
    btn.bind("<Enter>", lambda e: btn.config(bg=HOVER_COLOR))
    btn.bind("<Leave>", lambda e: btn.config(bg=PRIMARY_COLOR))
    return btn

# --- LOGIN WINDOW ---
login_window = tk.Tk()
login_window.title("Admin Login - Women Safety System")
login_window.geometry("450x300")
login_window.configure(bg=BACKGROUND_COLOR)
login_window.resizable(False, False)  # Prevent resizing for a polished look

# Header Frame
header_frame = tk.Frame(login_window, bg=PRIMARY_COLOR)
header_frame.pack(fill="x")

tk.Label(header_frame, text="Women Safety System", font=("Arial", 18, "bold"),
         fg=WHITE, bg=PRIMARY_COLOR).pack(pady=15)

# Main Content Frame
content_frame = tk.Frame(login_window, bg=BACKGROUND_COLOR)
content_frame.pack(expand=True, pady=20)

# Login Title
tk.Label(content_frame, text="Admin Login", font=("Arial", 16, "bold"),
         fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR).pack(pady=10)

# ID Field
id_frame = tk.Frame(content_frame, bg=BACKGROUND_COLOR)
id_frame.pack(pady=5)
tk.Label(id_frame, text="Admin ID:", font=("Arial", 12), bg=BACKGROUND_COLOR,
         fg=TEXT_COLOR).pack(side="left", padx=5)
id_entry = tk.Entry(id_frame, font=("Arial", 12), width=25, bg=WHITE, fg=TEXT_COLOR,
                    relief="solid", bd=1)
id_entry.pack(side="left", padx=5)

# Password Field
password_frame = tk.Frame(content_frame, bg=BACKGROUND_COLOR)
password_frame.pack(pady=5)
tk.Label(password_frame, text="Password:", font=("Arial", 12), bg=BACKGROUND_COLOR,
         fg=TEXT_COLOR).pack(side="left", padx=5)
password_entry = tk.Entry(password_frame, font=("Arial", 12), width=25, bg=WHITE,
                          fg=TEXT_COLOR, show="*", relief="solid", bd=1)
password_entry.pack(side="left", padx=5)

# Login Button
login_button = create_button(content_frame, "Login", login)
login_button.pack(pady=20)

# Bind Enter key to login
login_window.bind("<Return>", lambda event: login())

login_window.mainloop()