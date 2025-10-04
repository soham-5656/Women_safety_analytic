import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import sys
import time
import mediapipe as mp
from module_utlis1 import SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints

# ✅ Load Face Detection Model (OpenCV DNN)
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

# ✅ Load Gender Detection Model (OpenCV DNN)
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
gender_net = cv2.dnn.readNet(GENDER_PROTO, GENDER_MODEL)
GENDER_LABELS = ["Male", "Female"]

# ✅ Actions List
ACTIONS = ["non-grabbing", "grabbing"]
BEATING_ACTIONS = ["non-beating", "beating"]

# ✅ Load trained action detection models
try:
    grabbing_model = tf.keras.models.load_model("grabbing_model.keras")
    beating_model = tf.keras.models.load_model("beating_model.keras")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ✅ Function to detect gender
def detect_gender(face):
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False, crop=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LABELS[np.argmax(gender_preds)]
    return gender

# ✅ Create Tkinter UI
def create_ui():
    """Create a Tkinter window to display the latest prediction and confidence score."""
    root = tk.Tk()
    root.title("Action Detection Info")
    root.geometry("400x350")

    label_gender = tk.Label(root, text="Detected Gender: ", font=("Helvetica", 14))
    label_grab = tk.Label(root, text="Grabbing Prediction: ", font=("Helvetica", 14))
    label_grab_acc = tk.Label(root, text="Confidence: ", font=("Helvetica", 12))
    label_beat = tk.Label(root, text="Beating Prediction: ", font=("Helvetica", 14))
    label_beat_acc = tk.Label(root, text="Confidence: ", font=("Helvetica", 12))
    label_actions = tk.Label(root, text="Confirmed Actions: ", font=("Helvetica", 12), wraplength=380)

    label_gender.pack(pady=3)
    label_grab.pack(pady=3)
    label_grab_acc.pack(pady=3)
    label_beat.pack(pady=3)
    label_beat_acc.pack(pady=3)
    label_actions.pack(pady=5)

    info = tk.Label(root,
                    text="Instructions:\nPress 'c' to confirm action\nPress 's' to submit\nPress 'q' to quit",
                    font=("Helvetica", 10))
    info.pack(pady=10)

    return root, label_gender, label_grab, label_grab_acc, label_beat, label_beat_acc, label_actions

# ✅ Function to take a screenshot
def capture_screenshot(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.jpg"
    screenshot_path = os.path.join(os.getcwd(), screenshot_filename)
    cv2.imwrite(screenshot_path, frame)
    print(f"📸 Screenshot saved: {screenshot_path}")

# ✅ Real-time detection function with gender detection
def real_time_detection(grabbing_model, beating_model):
    sequence = []
    threshold = 0.8  # Confidence threshold
    confirmed_actions = []

    # Initialize UI
    ui_root, label_gender, label_grab, label_grab_acc, label_beat, label_beat_acc, label_actions = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        sys.exit()

    expected_seq_length = SEQ_LENGTH

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        grab_confidence, beat_confidence = 0, 0
        latest_grab, latest_beat = "Uncertain", "Uncertain"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # ✅ Detect face and predict gender
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            detected_gender = "Unknown"
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        detected_gender = detect_gender(face)

            # ✅ Display Gender on UI
            label_gender.config(text=f"Detected Gender: {detected_gender}")
            ui_root.update_idletasks()
            ui_root.update()

            if len(sequence) == expected_seq_length:
                input_data = np.expand_dims(sequence, axis=0)  # Shape (1, SEQ_LENGTH, 1662)

                # ✅ Predict grabbing action
                grab_res = grabbing_model.predict(input_data)[0]
                grab_confidence = float(grab_res)
                grab_idx = int(grab_confidence > 0.5)
                latest_grab = ACTIONS[grab_idx] if grab_confidence > threshold else "Uncertain"

                # ✅ Predict beating action
                beat_res = beating_model.predict(input_data)[0]
                beat_confidence = float(beat_res)
                beat_idx = int(beat_confidence > 0.5)
                latest_beat = BEATING_ACTIONS[beat_idx] if beat_confidence > threshold else "Uncertain"

                sequence = sequence[-(expected_seq_length - 1):]

                # ✅ Update UI with predictions
                label_grab.config(text=f"Grabbing Prediction: {latest_grab}")
                label_grab_acc.config(text=f"Confidence: {grab_confidence * 100:.1f}%")
                label_beat.config(text=f"Beating Prediction: {latest_beat}")
                label_beat_acc.config(text=f"Confidence: {beat_confidence * 100:.1f}%")

            cv2.imshow('Action & Gender Detection', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    ui_root.destroy()

# ✅ Run the detection
if __name__ == "__main__":
    real_time_detection(grabbing_model, beating_model)
