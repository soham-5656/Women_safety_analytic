import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collect1 import augment_sequence  # ✅ Import augmentation function
from module_utlis1 import create_model  # ✅ Import model function

# ==============================
# CONFIGURATION
# ==============================
TARGET_FRAMES = 45
DATA_PATH = "training_data2"
INITIAL_LEARNING_RATE = 1e-4  # ✅ Stable learning rate

# Custom Model Checkpoint (Save best model)
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
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

# ==============================
# LOAD DATA
# ==============================
def load_training_sequences(action_label):
    """
    Load sequences for a given action label.
    Includes data augmentation for robustness.
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
                    labels.append(1 if action_label == "grabbing" else 0)  # "Grabbing" = 1, "Non-Grabbing" = 0

                    # ✅ Augment Data
                    for _ in range(3):  # Create 3 augmented versions
                        augmented_sequence = augment_sequence(sequence)
                        sequences.append(augmented_sequence)
                        labels.append(1 if action_label == "grabbing" else 0)

    return sequences, labels

# ==============================
# TRAIN MODEL
# ==============================
def train_model():
    """
    Load "grabbing" and "non-grabbing" data, apply augmentation,
    and train an LSTM model for classification.
    """
    grabbing_sequences, grabbing_labels = load_training_sequences("grabbing")
    non_grabbing_sequences, non_grabbing_labels = load_training_sequences("non-grabbing")

    # Merge datasets
    sequences = grabbing_sequences + non_grabbing_sequences
    labels = grabbing_labels + non_grabbing_labels

    if len(sequences) == 0:
        sys.exit("❌ No valid training data found. Please collect 'grabbing' and 'non-grabbing' data.")

    # Convert to NumPy arrays
    X = np.array(sequences)
    y = np.array(labels).reshape(-1, 1)  # Convert labels to binary format

    # ✅ Split into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Data Augmentation for Training
    X_train_augmented = np.array([augment_sequence(seq) for seq in X_train])
    X_train = np.concatenate([X_train, X_train_augmented])  # Duplicate training data
    y_train = np.concatenate([y_train, y_train])  # Duplicate labels

    # ==============================
    # CLASS WEIGHTS (Handle Imbalance)
    # ==============================
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # ==============================
    # MODEL DEFINITION
    # ==============================
    def create_model(input_shape):
        l2_reg = tf.keras.regularizers.l2(0.01)  # ✅ Regularization

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape,
                                 kernel_regularizer=l2_reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),  # ✅ High dropout for better generalization

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
        early_stopping,  # Stop training early if no improvement
        CustomModelCheckpoint("grabbing_model.keras")
    ]

    # ==============================
    # TRAINING
    # ==============================
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,  # ✅ Handle class imbalance
        callbacks=callbacks
    )

    return model

if __name__ == "__main__":
    train_model()
