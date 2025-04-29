import os
import ast
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tqdm import tqdm
from joblib import Parallel, delayed
from skmultilearn.model_selection import iterative_train_test_split

# Constants
AUDIO_DIR = "train_audio"
CSV_FILE = "train.csv"
TARGET_HEIGHT, TARGET_WIDTH = 128, 206  # Fixed spectrogram size
BATCH_SIZE = 8  # Batch size for dataset
AUTOTUNE = tf.data.AUTOTUNE
N_JOBS = -1  # Use all available CPU cores

# Create directories to save processed data
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(CSV_FILE)

# Get species labels
species_list = sorted(df["primary_label"].unique())
species_to_idx = {species: i for i, species in enumerate(species_list)}

def process_audio(file_path):
    """Load an audio file and convert it into a resized Mel-spectrogram."""
    try:
        audio, sr = librosa.load(file_path, sr=32000, mono=True, dtype=np.float32)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=TARGET_HEIGHT)
        spec = librosa.power_to_db(spec, ref=np.max)

        # Normalize between 0 and 1
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-6)

        # Resize
        spec_resized = tf.image.resize(spec[..., np.newaxis], (TARGET_HEIGHT, TARGET_WIDTH), method='bilinear')

        return spec_resized.numpy().squeeze()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_row(row):
    """Process a single dataframe row into (spectrogram, label_vector)."""
    file_path = os.path.join(AUDIO_DIR, row["filename"])
    if not os.path.exists(file_path):
        return None, None

    spectrogram = process_audio(file_path)
    if spectrogram is None:
        return None, None

    label_vector = np.zeros(len(species_list), dtype=np.float32)
    label_vector[species_to_idx[row["primary_label"]]] = 1

    if isinstance(row["secondary_labels"], str) and row["secondary_labels"].strip():
        try:
            secondary_labels = ast.literal_eval(row["secondary_labels"])
            if isinstance(secondary_labels, list):
                for sec_label in secondary_labels:
                    if sec_label in species_to_idx:
                        label_vector[species_to_idx[sec_label]] = 1
        except Exception as e:
            print(f"Warning: failed to parse secondary labels for {file_path}: {e}")

    return spectrogram, label_vector

def create_datasets(X_train, y_train, X_val, y_val):
    """Create TensorFlow datasets for training and validation."""
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return train_dataset, val_dataset

def main():
    print("Processing dataset...")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df))
    )

    results = [res for res in results if res[0] is not None and res[1] is not None]
    X, y = zip(*results)
    X = np.array(X)
    y = np.array(y)

    X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, 128, 206, 1)

    X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size=0.07)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Save preprocessed arrays
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train) #optional
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train) #optional
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_val.npy"), X_val) #optional
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_val.npy"), y_val) #optional

    print(f"Preprocessing complete. Files saved in '{PROCESSED_DATA_DIR}'.")

    # Create TensorFlow datasets
    train_dataset, val_dataset = create_datasets(X_train, y_train, X_val, y_val)
    print("Datasets created successfully.")

if __name__ == "__main__":
    main()
