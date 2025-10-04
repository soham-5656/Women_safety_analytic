import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox  # For confirmation prompts.
import sys
from model_utils import SEQ_LENGTH, mp_holistic, mediapipe_detection, extract_keypoints
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

# ✅ Define possible actions (beating & non-beating)
ACTIONS = ["non-beating", "beating"]  # Ensure correct order

def create_ui():
    """
    Create a Tkinter window to display the latest prediction, confidence score,
    and the confirmed actions list.
    """
    root = tk.Tk()
    root.title("Action Detection Info")
    root.geometry("400x250")

    label_action = tk.Label(root, text="Latest Prediction: ", font=("Helvetica", 16))
    label_accuracy = tk.Label(root, text="Confidence: ", font=("Helvetica", 16))
    label_actions = tk.Label(root, text="Confirmed Actions: ", font=("Helvetica", 16), wraplength=380)

    label_action.pack(pady=5)
    label_accuracy.pack(pady=5)
    label_actions.pack(pady=5)

    info = tk.Label(root,
                    text="Instructions:\nPress 'c' to confirm action\nPress 's' to submit\nPress 'q' to quit",
                    font=("Helvetica", 12))
    info.pack(pady=10)

    return root, label_action, label_accuracy, label_actions

def real_time_detection(model):
    """
    Runs real-time action detection using the trained model.
    Captures video frames, extracts keypoints, and predicts the action.
    """
    sequence = []
    threshold = 0.8  # ✅ Confidence threshold
    confirmed_actions = []  # List of confirmed actions

    # Initialize UI
    ui_root, label_action, label_accuracy, label_actions = create_ui()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        sys.exit()

    expected_seq_length = model.input_shape[1] if model.input_shape and model.input_shape[1] is not None else SEQ_LENGTH

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        confidence = 0  # Confidence score
        latest_prediction = "Uncertain"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == expected_seq_length:
                # ✅ Predict the action (Fix applied)
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  # Returns a single probability
                confidence = float(res)  # Extract confidence directly
                predicted_idx = int(confidence > 0.5)  # Convert probability to 0 (non-beating) or 1 (beating)
                latest_prediction = ACTIONS[predicted_idx] if confidence > threshold else "Uncertain"

                # ✅ Keep the last frames for the next prediction
                sequence = sequence[-(expected_seq_length - 1):]

                # ✅ Update UI with predictions
                label_action.config(text=f"Latest Prediction: {latest_prediction}")
                label_accuracy.config(text=f"Confidence: {confidence * 100:.1f}%")
                ui_root.update_idletasks()
                ui_root.update()

            cv2.imshow('Action Detection', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # ✅ Confirm action if confidence is high
                if confidence > threshold:
                    confirmed_actions.append(latest_prediction)
                    label_actions.config(text=f"Confirmed Actions: {' '.join(confirmed_actions)}")
                    print(f"✅ Action confirmed: {latest_prediction}")
                    sequence = []  # Reset sequence
                else:
                    print("⚠️ Confidence too low to confirm.")
            elif key == ord('s'):
                # ✅ Submit detected actions
                if confirmed_actions:
                    answer = messagebox.askyesno("Action Confirmation",
                                                 "Is the action sequence complete? (Yes to finish, No to continue)")
                    if answer:
                        final_actions = ' '.join(confirmed_actions)
                        print("✅ Final Detected Actions:", final_actions)
                        confirmed_actions = []  # Clear actions
                        label_actions.config(text="Confirmed Actions: ")
                        break
                    else:
                        print("Continuing detection...")

    cap.release()
    cv2.destroyAllWindows()
    ui_root.destroy()
    return

if __name__ == "__main__":
    # ✅ Load the trained beating detection model
    trained_model_path = "beating_model.keras"
    try:
        model = tf.keras.models.load_model(trained_model_path)
        print(f"✅ Loaded trained model from {trained_model_path}")
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        sys.exit()

    real_time_detection(model)
