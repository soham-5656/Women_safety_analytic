import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# ======================
# Configuration
# ======================
ACTIONS = ['beating', 'non-beating']  # Classes for training
DATA_PATH = 'training_data1'  # Directory containing recorded .npy sequences
SEQ_LENGTH = 30  # Number of frames per sequence
NUM_SEQUENCES = 30  # Number of sequences per class

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
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """
    Extract and flatten keypoints from MediaPipe results.
    Returns concatenated arrays for pose, face, and both hands.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def create_model(input_shape):
    l2_reg = tf.keras.regularizers.l2(0.01)  # Define L2 regularization

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape,
                             kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),  # Increase dropout

        tf.keras.layers.LSTM(128, return_sequences=False, kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model