import os
import cv2
import numpy as np
import sys
import time
import threading
import tensorflow as tf
import mediapipe as mp

# Import organization-specific modules
from model_utils import SEQ_LENGTH, NUM_SEQUENCES, mp_holistic, mp_drawing, mediapipe_detection, extract_keypoints

# Set Actions
ACTIONS = ["beating", "non-beating"]
DATA_PATH = 'training_data1'
TARGET_FRAMES = 45

# -----------------------
# GPU Acceleration Setup
# -----------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except Exception as e:
        print(f"❌ Failed to set GPU memory growth: {e}")

# -----------------------
# Background Camera Capture
# -----------------------
current_frame = None
frame_lock = threading.Lock()
capture_running = True

def camera_capture_thread(cap):
    """ Continuously capture frames in a background thread. """
    global current_frame, capture_running
    while capture_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(0.01)

def get_current_frame():
    """ Get the latest camera frame safely. """
    with frame_lock:
        return None if current_frame is None else current_frame.copy()

# -----------------------
# Data Augmentation Function
# -----------------------
def augment_sequence(sequence, noise_std=0.01):
    """ Adds Gaussian noise to a sequence for data augmentation. """
    sequence_arr = np.array(sequence)
    noise = np.random.normal(0, noise_std, sequence_arr.shape)
    return sequence_arr + noise

# -----------------------
# Data Collection for a Single Action
# -----------------------
def record_action(action, cap, holistic):
    """ Record a dataset for the given action (beating or non-beating). """
    print(f"\n📢 Ready to record data for '{action}'")
    print("ℹ️ Press 's' anytime to skip.")

    session_count = 0
    current_seq = 0
    current_session_dir = None

    while True:
        # Start a new session if needed
        if current_session_dir is None or current_seq >= NUM_SEQUENCES:
            session_count += 1
            current_session_dir = os.path.join(DATA_PATH, action, f"session_{session_count}")
            os.makedirs(current_session_dir, exist_ok=True)
            current_seq = 0
            print(f"\n🔄 Starting session {session_count} for '{action}'")

        # Capture NUM_SEQUENCES sequences
        while current_seq < NUM_SEQUENCES:
            skip_current_word = False

            # Countdown before starting
            for countdown in range(3, 0, -1):
                frame = get_current_frame()
                if frame is None:
                    continue
                text = f"Get Ready: {countdown}"
                cv2.putText(frame, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow('Data Collection', frame)
                if cv2.waitKey(1000) & 0xFF == ord('s'):
                    print(f"⚠️ Skipping '{action}'")
                    skip_current_word = True
                    break
            if skip_current_word:
                break

            sequence = []
            paused = False

            while len(sequence) < TARGET_FRAMES:
                frame = get_current_frame()
                if frame is None:
                    continue

                if not paused:
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Status overlay
                    status_text = f"Session {session_count} | Seq {current_seq} | Frame {len(sequence) + 1}/{TARGET_FRAMES}"
                    cv2.putText(image, status_text, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                cv2.imshow('Data Collection', image)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    return False
                elif key == ord('d'):
                    print(f"🛑 Discarding sequence {current_seq}")
                    sequence = []
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️ Paused." if paused else "▶️ Resumed.")
                elif key == ord('s'):
                    print(f"⚠️ Skipping '{action}'")
                    skip_current_word = True
                    break
            if skip_current_word:
                break

            # Save sequence if completed
            if len(sequence) == TARGET_FRAMES:
                cv2.putText(image, "Keep this sequence? (y=Yes, d=Discard)", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Data Collection', image)
                print(f"✅ Sequence {current_seq} captured for '{action}'")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        return False
                    elif key == ord('y'):
                        np.save(os.path.join(current_session_dir, f"seq_{current_seq}.npy"), sequence)
                        print(f"💾 Saved sequence {current_seq} for '{action}'")

                        # Save augmented version
                        augmented_sequence = augment_sequence(sequence)
                        np.save(os.path.join(current_session_dir, f"seq_{current_seq}_aug.npy"), augmented_sequence)
                        print(f"💾 Saved augmented sequence {current_seq} for '{action}'")

                        current_seq += 1
                        break
                    elif key == ord('d'):
                        print("🔄 Re-recording sequence.")
                        break
                    elif key == ord('s'):
                        print(f"⚠️ Skipping '{action}'")
                        skip_current_word = True
                        break
            else:
                continue

        if skip_current_word:
            break

    return True


# -----------------------
# Main Interactive Data Collection
# -----------------------
def collect_data():
    global capture_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌ Cannot access the camera.")

    thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
    thread.start()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            selected = input("\nEnter action to record (beating/non-beating) or 'exit' to quit: ").strip()
            if selected.lower() == 'exit':
                break
            if selected not in ACTIONS:
                print("❌ Invalid action.")
                continue

            success = record_action(selected, cap, holistic)
            if not success:
                break

    capture_running = False
    thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_data()
