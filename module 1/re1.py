import os
import numpy as np

TARGET_FRAMES = 45  # Ensure all sequences are of this length

def resample_sequence(sequence, target_length=TARGET_FRAMES):
    """
    Resamples the sequence to match the target number of frames.
    - If sequence is too long, it selects evenly spaced frames.
    - If sequence is too short, it pads with the last frame.
    """
    num_frames = len(sequence)

    if num_frames == target_length:
        return sequence  # No change needed

    elif num_frames > target_length:
        # Select evenly spaced frames
        indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
        return sequence[indices]

    else:
        # Pad with the last available frame
        pad = np.repeat(sequence[-1][np.newaxis, :], target_length - num_frames, axis=0)
        return np.concatenate([sequence, pad], axis=0)

def process_npy_files(directory, target_length=TARGET_FRAMES):
    """
    Processes all .npy files in the dataset directory, resampling them to the target length.
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.npy'):
                continue  # Skip non-npy files

            file_path = os.path.join(root, filename)
            sequence = np.load(file_path, allow_pickle=True)

            if sequence.ndim == 0:
                continue  # Skip empty files

            num_frames = len(sequence)
            if num_frames != target_length:
                resampled = resample_sequence(sequence, target_length)
                np.save(file_path, resampled)  # Overwrite with the new version
                print(f"✅ Resampled {file_path}: {num_frames} --> {target_length} frames.")

if __name__ == "__main__":
    directory_path = "C:/Users/Akshay/Women-Safety-Analytics/training_data2"
    process_npy_files(directory_path)
