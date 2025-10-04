import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# ======================
# Configuration
# ======================
ACTIONS = ['beating', 'non-beating']  # Classes for training
DATA_PATH = 'training_data1'  # Directory containing recorded .npy sequences
SEQ_LENGTH = 30  # Number of frames per sequence

# **Ensure NUM_FEATURES matches extracted keypoints**
NUM_FEATURES = 1662  # Pose (33x4), Face (468x3), LH (21x3), RH (21x3)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ======================
# Common Functions
# ======================
def mediapipe_detection(image, model):
    """
    Process image with MediaPipe Holistic model.
    Converts image to RGB, processes it, and then returns a BGR image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def extract_keypoints(results):
    """
    Extract and flatten keypoints from MediaPipe results.
    Returns concatenated arrays for pose, face, and both hands.
    """
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                         for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z]
                         for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z]
                       for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z]
                       for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

        # Ensure the final output shape is always (1662)
        keypoints = np.concatenate([pose, face, lh, rh])
        return keypoints
    except Exception as e:
        print(f"❌ Keypoint Extraction Error: {e}")
        return np.zeros(NUM_FEATURES)  # Return zero array if error occurs


def create_model(input_shape):
    """
    Creates and returns an LSTM model with L2 regularization and dropout.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),

        LSTM(128, return_sequences=False, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ✅ Dynamically check input shape based on extracted keypoints
input_shape = (SEQ_LENGTH, NUM_FEATURES)
model = create_model(input_shape)

# ✅ Print model summary to verify structure
model.summary()
