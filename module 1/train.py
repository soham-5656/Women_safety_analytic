import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collect import augment_sequence  # ✅ Correct function import
from model_utils import ACTIONS, create_model

# ==============================
# CONFIGURATION
# ==============================
TARGET_FRAMES = 45
DATA_PATH = "training_data1"
INITIAL_LEARNING_RATE = 1e-4  # ✅ Prevent getting stuck at low LR

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """ Save the model when validation loss improves. """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return
        if current_loss < self.best:
            self.best = current_loss
            self.model.save(self.filepath)
            print(f"\n✅ Epoch {epoch + 1}: val_loss improved to {current_loss:.4f}, saving model.")

def load_training_sequences(action_label):
    """
    Load training sequences for a given action label.
    Includes data augmentation.
    """
    sequences, labels = [], []
    action_dir = os.path.join(DATA_PATH, action_label)

    if not os.path.exists(action_dir):
        print(f"⚠️ No training data found for '{action_label}'. Skipping...")
        return sequences, labels

    for session in os.listdir(action_dir):
        session_path = os.path.join(action_dir, session)
        if os.path.isdir(session_path):
            for seq_file in os.listdir(session_path):
                if seq_file.endswith('.npy'):
                    seq_path = os.path.join(session_path, seq_file)
                    sequence = np.load(seq_path)

                    if len(sequence) != TARGET_FRAMES:
                        print(f"⚠️ Skipping {seq_path}: {len(sequence)} frames instead of {TARGET_FRAMES}.")
                        continue

                    sequences.append(sequence)
                    labels.append(1 if action_label == "beating" else 0)  # "Beating" = 1, "Non-beating" = 0

                    # Augmented version with more variations
                    for _ in range(3):  # ✅ Create 3 augmented versions
                        augmented_sequence = augment_sequence(sequence)
                        sequences.append(augmented_sequence)
                        labels.append(1 if action_label == "beating" else 0)

    return sequences, labels

def train_model():
    """
    Load sequences for both "beating" and "non-beating", apply augmentation,
    and train an LSTM model for classification.
    """
    # Load both "beating" and "non-beating" data
    beating_sequences, beating_labels = load_training_sequences("beating")
    non_beating_sequences, non_beating_labels = load_training_sequences("non-beating")

    # Merge both datasets
    sequences = beating_sequences + non_beating_sequences
    labels = beating_labels + non_beating_labels

    if len(sequences) == 0:
        sys.exit("❌ No valid training data found. Please collect 'beating' and 'non-beating' data.")

    # Convert to NumPy arrays
    X = np.array(sequences)
    y = np.array(labels).reshape(-1, 1)  # Convert labels to binary format

    # ✅ Split into train and validation sets (20% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Apply data augmentation to training sequences
    X_train_augmented = np.array([augment_sequence(seq) for seq in X_train])
    X_train = np.concatenate([X_train, X_train_augmented])  # Augment training data
    y_train = np.concatenate([y_train, y_train])  # Duplicate labels for augmented data

    # ==============================
    # CLASS WEIGHT HANDLING
    # ==============================
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # ==============================
    # MODEL DEFINITION
    # ==============================
    def create_model(input_shape):
        l2_reg = tf.keras.regularizers.l2(0.01)  # ✅ Increased Regularization

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape,
                                 kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),  # ✅ Increased dropout

            tf.keras.layers.LSTM(128, return_sequences=False, kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),

            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
            tf.keras.layers.Dropout(0.7),

            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=INITIAL_LEARNING_RATE),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model = create_model((TARGET_FRAMES, X.shape[2]))

    # ==============================
    # CALLBACKS
    # ==============================
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    callbacks = [
        early_stopping,  # ⏳ Stop training early if validation loss doesn't improve
        CustomModelCheckpoint("beating_model.keras")
    ]

    # ==============================
    # TRAINING
    # ==============================
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,  # ✅ Added class weights
        callbacks=callbacks
    )

    return model

if __name__ == "__main__":
    train_model()
