import cv2
import sqlite3
import numpy as np
import time
import os
import tensorflow as tf
import tkinter as tk
import datetime
import mediapipe as mp
from module_utlis1 import SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints
import smtplib
from email.message import EmailMessage
from twilio.rest import Client
import threading
import queue

# Email and Twilio Credentials
EMAIL_ADDRESS = "sohamdawale@gmail.com"
EMAIL_PASSWORD = "jkna pnyk ztzb glny"
TO_EMAIL = "tkanse30@gmail.com"

TWILIO_ACCOUNT_SID = 'ACf23d449c6c23ab7ad7575f1b9c6fcbae'
TWILIO_AUTH_TOKEN = "9a8172c794f564364b25a19c76a2f7a2"
TWILIO_PHONE_NUMBER = "+12317742606"
EMERGENCY_PHONE_NUMBER = "+919594510584"

# Action Recognition Models
ACTIONS = ["non-grabbing", "grabbing"]
BEATING_ACTIONS = ["non-beating", "beating"]

# Load models on startup
print("Loading models...")
try:
    grabbing_model = tf.keras.models.load_model("grabbing_model.keras")
    beating_model = tf.keras.models.load_model("beating_model.keras")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Gender Detection Models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# GPU optimization
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU acceleration enabled")
    except:
        print("GPU acceleration failed to enable")

# Database Setup
DB_FILE = "women_safety.db"
FEMALE_THRESHOLD = 0  # Minimum females required to trigger an alert

# Background task queue
task_queue = queue.Queue()

# Tkinter UI Setup
def create_ui():
    root = tk.Tk()
    root.title("Action & Threat Detection")
    root.geometry("400x350")

    label_grab = tk.Label(root, text="Grabbing: ", font=("Helvetica", 14))
    label_grab_acc = tk.Label(root, text="Confidence: ", font=("Helvetica", 12))
    label_beat = tk.Label(root, text="Beating: ", font=("Helvetica", 14))
    label_beat_acc = tk.Label(root, text="Confidence: ", font=("Helvetica", 12))
    label_gender = tk.Label(root, text="Males: 0, Females: 0", font=("Helvetica", 12))
    label_fps = tk.Label(root, text="FPS: 0", font=("Helvetica", 12))

    label_grab.pack(pady=3)
    label_grab_acc.pack(pady=3)
    label_beat.pack(pady=3)
    label_beat_acc.pack(pady=3)
    label_gender.pack(pady=3)
    label_fps.pack(pady=3)

    return root, label_grab, label_grab_acc, label_beat, label_beat_acc, label_gender, label_fps

# Capture Screenshot
def capture_screenshot(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.jpg"
    screenshot_path = os.path.join(os.getcwd(), screenshot_filename)
    cv2.imwrite(screenshot_path, frame)
    return screenshot_path

# Register Threat in Database
def register_case(male_count, female_count, screenshot_path, location):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO cases (timestamp, male_count, female_count, screenshot_path, location, status)
            VALUES (?, ?, ?, ?, ?, ?)""",
                       (current_timestamp, male_count, female_count, screenshot_path, location, "Pending"))
        conn.commit()
        conn.close()
        print(f"✅ Case Registered at {current_timestamp}!")
    except Exception as e:
        print(f"Database error: {e}")

# Improved Gender Detection
def detect_gender(frame):
    male_count = 0
    female_count = 0
    display_frame = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.65:
            x1 = max(0, int(detections[0, 0, i, 3] * frameWidth))
            y1 = max(0, int(detections[0, 0, i, 4] * frameHeight))
            x2 = min(frameWidth, int(detections[0, 0, i, 5] * frameWidth))
            y2 = min(frameHeight, int(detections[0, 0, i, 6] * frameHeight))

            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 100:
                continue

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face = frame[max(0, y1 - int((y2 - y1) * 0.2)):min(y2 + int((y2 - y1) * 0.2), frameHeight),
                   max(0, x1 - int((x2 - x1) * 0.2)):min(x2 + int((x2 - x1) * 0.2), frameWidth)]

            if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                continue

            try:
                face_resized = cv2.resize(face, (227, 227))
                blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                genderNet.setInput(blob)
                gender_preds = genderNet.forward()
                gender = "Male" if gender_preds[0].argmax() == 0 else "Female"
                gender_confidence = gender_preds[0][gender_preds[0].argmax()] * 100

                if gender == "Male":
                    male_count += 1
                else:
                    female_count += 1

                label = f"{gender} ({gender_confidence:.1f}%)"
                cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if gender == "Female" else (0, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

    return display_frame, male_count, female_count

# Background Task Worker
def background_worker():
    while True:
        try:
            task, args = task_queue.get()
            if task is None:
                break
            task(*args)
            task_queue.task_done()
        except Exception as e:
            print(f"Background task error: {e}")

# Send Email Alert
def send_email_alert(men_count, women_count, screenshot_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = "🚨 Emergency Alert!"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg.set_content(f"ALERT! {men_count} men detected with {women_count} women. Immediate action required!")

        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as file:
                msg.add_attachment(file.read(), maintype='image', subtype='jpeg', filename="screenshot.jpg")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email Alert Sent Successfully!")
    except Exception as e:
        print(f"❌ Email Alert Failed: {e}")

# Make Call Alert
def make_call_alert(men_count, women_count):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml=f'<Response><Say>Alert! {men_count} men detected with {women_count} women. Immediate action required.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_PHONE_NUMBER
        )
        print("✅ Call Alert Sent Successfully!")
    except Exception as e:
        print(f"❌ Call Alert Failed: {e}")

# Real-time Detection
def real_time_detection():
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()

    sequence = []
    last_alert_time = time.time() - 30
    skip_detection_frames = 2
    frame_count = 0

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    gender_history = []
    history_size = 10

    ui_root, label_grab, label_grab_acc, label_beat, label_beat_acc, label_gender, label_fps = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    latest_grab = "Non-Detecting"
    latest_beat = "Non-Detecting"
    grab_confidence = 0.0
    beat_confidence = 0.0
    smoothed_male_count = 0
    smoothed_female_count = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue

            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            frame_count += 1
            process_ml = (frame_count % skip_detection_frames == 0)

            if process_ml:
                try:
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                    if len(sequence) == SEQ_LENGTH:
                        input_data = np.expand_dims(sequence, axis=0)
                        grab_pred = grabbing_model.predict(input_data, verbose=0)[0]
                        beat_pred = beating_model.predict(input_data, verbose=0)[0]

                        # Assuming binary classification with softmax output
                        grab_confidence = float(grab_pred[1])  # Probability of "grabbing"
                        beat_confidence = float(beat_pred[1])  # Probability of "beating"
                        latest_grab = ACTIONS[1 if grab_confidence > 0.5 else 0]
                        latest_beat = BEATING_ACTIONS[1 if beat_confidence > 0.5 else 0]

                        sequence = sequence[-(SEQ_LENGTH - 1):]
                except Exception as e:
                    print(f"Mediapipe or prediction error: {e}")

            try:
                frame, male_count, female_count = detect_gender(frame)
                gender_history.append((male_count, female_count))
                if len(gender_history) > history_size:
                    gender_history.pop(0)

                smoothed_male_count = int(sum([x[0] for x in gender_history]) / len(gender_history) + 0.5)
                smoothed_female_count = int(sum([x[1] for x in gender_history]) / len(gender_history) + 0.5)
            except Exception as e:
                print(f"Gender detection error: {e}")

            cv2.putText(frame, f"Males: {smoothed_male_count}, Females: {smoothed_female_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Grabbing: {latest_grab}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Beating: {latest_beat}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {current_fps}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            try:
                label_grab.config(text=f"Grabbing: {latest_grab}")
                label_grab_acc.config(text=f"Confidence: {grab_confidence * 100:.1f}%")
                label_beat.config(text=f"Beating: {latest_beat}")
                label_beat_acc.config(text=f"Confidence: {beat_confidence * 100:.1f}%")
                label_gender.config(text=f"Males: {smoothed_male_count}, Females: {smoothed_female_count}")
                label_fps.config(text=f"FPS: {current_fps}")
                ui_root.update()
            except tk.TclError:
                break

            current_time = time.time()
            if (smoothed_male_count > 0 and smoothed_female_count > 0 and
                    (latest_grab == "grabbing" or latest_beat == "beating") and
                    (current_time - last_alert_time >= 10)):
                last_alert_time = current_time
                screenshot_path = capture_screenshot(frame)
                task_queue.put((register_case, (smoothed_male_count, smoothed_female_count, screenshot_path, "12.9716,77.5946")))
                task_queue.put((send_email_alert, (smoothed_male_count, smoothed_female_count, screenshot_path)))
                task_queue.put((make_call_alert, (smoothed_male_count, smoothed_female_count)))
                print(f"🚨 Alert Triggered at {datetime.datetime.now()} 🚨")

            cv2.imshow("Action & Threat Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    task_queue.put((None, None))
    worker_thread.join(timeout=1.0)
    try:
        ui_root.destroy()
    except:
        pass

# Run
if __name__ == "__main__":
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            male_count INTEGER,
            female_count INTEGER,
            screenshot_path TEXT,
            location TEXT,
            status TEXT
        )
        ''')
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")
        exit(1)

    print("Starting real-time detection system...")
    real_time_detection()