import os
import cv2
import numpy as np
import tensorflow as tf
import time

from module_utlis1 import SEQ_LENGTH, NUM_SEQUENCES, mp_holistic, mp_drawing, mediapipe_detection, extract_keypoints  # Ensure this function extracts keypoints correctly

# ✅ Load trained models
try:
    grabbing_model = tf.keras.models.load_model("grabbing_model.keras")
    beating_model = tf.keras.models.load_model("beating_model.keras")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ✅ Define sequence storage for time-series data
SEQ_LENGTH = 45  # Adjust based on your model
sequence = []


def preprocess_frame(frame):
    """
    Extracts keypoints from the frame and prepares input for model prediction.
    Ensures shape is (1, SEQ_LENGTH, feature_size).
    """
    global sequence

    keypoints = extract_keypoints(frame)  # Ensure this extracts (1662,) features
    sequence.append(keypoints)

    if len(sequence) == SEQ_LENGTH:
        input_data = np.expand_dims(sequence, axis=0)  # Shape (1, SEQ_LENGTH, feature_size)
        sequence.pop(0)  # Maintain sliding window
        return input_data
    return None  # Not enough frames yet


def checkGrabbing(frame):
    """
    Checks if the grabbing action is detected using the model.
    """
    input_data = preprocess_frame(frame)
    if input_data is None:
        return False  # Not enough frames collected yet

    prediction = grabbing_model.predict(input_data)[0][0]  # Extract probability
    return prediction > 0.5  # Threshold decision


def checkBeating(frame):
    """
    Checks if the beating action is detected using the model.
    """
    input_data = preprocess_frame(frame)
    if input_data is None:
        return False  # Not enough frames collected yet

    prediction = beating_model.predict(input_data)[0][0]
    return prediction > 0.5


def capture_screenshot(frame):
    """
    Captures and saves a screenshot with a timestamp.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.jpg"
    screenshot_path = os.path.join(os.getcwd(), screenshot_filename)
    cv2.imwrite(screenshot_path, frame)
    print(f"📸 Screenshot saved: {screenshot_path}")


# ✅ Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Unable to open camera.")
    exit()

print("🎥 Real-time detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ✅ Run action detection
    grabbing_detected = checkGrabbing(frame)
    beating_detected = checkBeating(frame)

    # ✅ Display results and take screenshot if detected
    if grabbing_detected:
        cv2.putText(frame, "⚠ Grabbing Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("🚨 Grabbing detected! Taking screenshot...")
        capture_screenshot(frame)

    if beating_detected:
        cv2.putText(frame, "⚠ Beating Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("🚨 Beating detected! Taking screenshot...")
        capture_screenshot(frame)

    cv2.imshow("Action Detection", frame)

    # ✅ Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
