import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
INPUT_SHAPE = (128, 206, 1)  # Mel-spectrogram dimensions
NUM_CLASSES = 206  # Number of bird species


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8):
        super().__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense = keras.Sequential([
            layers.Dense(channels // self.ratio, activation='relu', kernel_initializer='he_normal'),
            layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
        ])

    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        weights = self.shared_dense(gap)
        return inputs * weights


class SpectrogramTransformerEncoder(layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, ff_dim=256, dropout=0.1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim),
            layers.Activation('gelu'),
            layers.Dropout(self.dropout),
            layers.Dense(input_shape[-1])
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = layers.Dropout(self.dropout)
        self.dropout2 = layers.Dropout(self.dropout)

    def call(self, inputs):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.position_embeddings = layers.Embedding(
            input_dim=self.max_len,
            output_dim=self.d_model
        )
        self.dense = layers.Dense(self.d_model)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.position_embeddings(positions)
        pos_emb = self.dense(pos_emb)  # Learned scaling
        return inputs + pos_emb


def build_complete_model():
    # Input layer
    inputs = keras.Input(shape=INPUT_SHAPE)

    # --- Encoder Block 1 ---
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = ChannelAttention()(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # --- Encoder Block 2 ---
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = ChannelAttention()(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # --- Encoder Block 3 ---
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = ChannelAttention()(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.3)(x)

    # Prepare for Transformer
    seq_len = x.shape[1] * x.shape[2]
    channel_dim = x.shape[3]
    x = layers.Reshape((seq_len, channel_dim))(x)

    # Positional Embedding with learned scale
    x = PositionEmbedding(max_len=seq_len, d_model=channel_dim)(x)

    # Transformer Blocks
    for i in range(2):
        x = SpectrogramTransformerEncoder(
            num_heads=8,
            key_dim=64,
            ff_dim=1024,
            dropout=0.2,
            name=f'transformer_{i}'
        )(x)

    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Final Classifier
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model


def focal_loss_multilabel(alpha=0.25, gamma=2.0):
    """
    Focal Loss for Multi-Label Classification (Numerically Stable).
    """

    def loss_fn(y_true, y_pred):
        # Clipping to prevent log(0) issues and extreme gradients
        y_pred = K.clip(y_pred, 1e-6, 1 - 1e-6)

        # Compute binary cross-entropy
        bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)

        # Compute focal weight (numerically stable)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = (alpha * y_true + (1 - alpha) * (1 - y_true)) * ((1 - pt) + 1e-6) ** gamma

        # Compute final focal loss
        focal_loss = -focal_weight * bce
        return tf.reduce_mean(focal_loss)  # Mean loss over all samples

    return loss_fn


# Rebuild the model if needed
model = build_complete_model()
# Load the model weights (without optimizer)
model.load_weights('WildNet.keras')

# Configuration
AUDIO_DIR = "test_soundscapes/"
TARGET_HEIGHT, TARGET_WIDTH = 128, 206
SEGMENT_LENGTH = 5
SAMPLE_RATE = 32000
MAX_WORKERS = os.cpu_count()

# Load your trained model
model = model


def segment_audio(audio, sr, segment_length=SEGMENT_LENGTH):
    num_samples_per_segment = sr * segment_length
    num_segments = len(audio) // num_samples_per_segment
    segments = [audio[i * num_samples_per_segment:(i + 1) * num_samples_per_segment] for i in range(num_segments)]
    return segments


def generate_spectrogram(audio_segment, sr):
    try:
        spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=TARGET_HEIGHT)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-6)
        spec_resized = tf.image.resize(spec[..., np.newaxis], (TARGET_HEIGHT, TARGET_WIDTH), method='bilinear')
        return spec_resized.numpy().squeeze()
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


def process_file(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
        audio_segments = segment_audio(audio, sr)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        spectrograms = []
        row_ids = []
        for i, segment in enumerate(audio_segments):
            spec = generate_spectrogram(segment, sr)
            if spec is not None:
                spectrograms.append(spec)
                row_id = f"{base_name}_{(i + 1) * SEGMENT_LENGTH}"
                row_ids.append(row_id)

        return spectrograms, row_ids
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []


# Gather all files
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".ogg")])
file_paths = [os.path.join(AUDIO_DIR, f) for f in audio_files]
all_spectrograms = []
all_row_ids = []

# Process all files with multithreading and progress bar
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_file, file_path): file_path for file_path in file_paths}

    # Wrap tqdm around as_completed for progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files"):
        file_path = futures[future]
        try:
            specs, row_ids = future.result()
            all_spectrograms.extend(specs)
            all_row_ids.extend(row_ids)
        except Exception as e:
            print(f"Error with file {file_path}: {e}")

# Convert to tf_dataset and make predictions
input_array = np.array(all_spectrograms)
input_array = np.expand_dims(input_array, axis=-1)
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = (
    tf.data.Dataset.from_tensor_slices((input_array))
    .batch(8)
    .prefetch(AUTOTUNE)
)
predictions = model.predict(test_dataset)
# Species list
species_ids = ['1139490',
               '1192948',
               '1194042',
               '126247',
               '1346504',
               '134933',
               '135045',
               '1462711',
               '1462737',
               '1564122',
               '21038',
               '21116',
               '21211',
               '22333',
               '22973',
               '22976',
               '24272',
               '24292',
               '24322',
               '41663',
               '41778',
               '41970',
               '42007',
               '42087',
               '42113',
               '46010',
               '47067',
               '476537',
               '476538',
               '48124',
               '50186',
               '517119',
               '523060',
               '528041',
               '52884',
               '548639',
               '555086',
               '555142',
               '566513',
               '64862',
               '65336',
               '65344',
               '65349',
               '65373',
               '65419',
               '65448',
               '65547',
               '65962',
               '66016',
               '66531',
               '66578',
               '66893',
               '67082',
               '67252',
               '714022',
               '715170',
               '787625',
               '81930',
               '868458',
               '963335',
               'amakin1',
               'amekes',
               'ampkin1',
               'anhing',
               'babwar',
               'bafibi1',
               'banana',
               'baymac',
               'bbwduc',
               'bicwre1',
               'bkcdon',
               'bkmtou1',
               'blbgra1',
               'blbwre1',
               'blcant4',
               'blchaw1',
               'blcjay1',
               'blctit1',
               'blhpar1',
               'blkvul',
               'bobfly1',
               'bobher1',
               'brtpar1',
               'bubcur1',
               'bubwre1',
               'bucmot3',
               'bugtan',
               'butsal1',
               'cargra1',
               'cattyr',
               'chbant1',
               'chfmac1',
               'cinbec1',
               'cocher1',
               'cocwoo1',
               'colara1',
               'colcha1',
               'compau',
               'compot1',
               'cotfly1',
               'crbtan1',
               'crcwoo1',
               'crebob1',
               'cregua1',
               'creoro1',
               'eardov1',
               'fotfly',
               'gohman1',
               'grasal4',
               'grbhaw1',
               'greani1',
               'greegr',
               'greibi1',
               'grekis',
               'grepot1',
               'gretin1',
               'grnkin',
               'grysee1',
               'gybmar',
               'gycwor1',
               'labter1',
               'laufal1',
               'leagre',
               'linwoo1',
               'littin1',
               'mastit1',
               'neocor',
               'norscr1',
               'olipic1',
               'orcpar',
               'palhor2',
               'paltan1',
               'pavpig2',
               'piepuf1',
               'pirfly1',
               'piwtyr1',
               'plbwoo1',
               'plctan1',
               'plukit1',
               'purgal2',
               'ragmac1',
               'rebbla1',
               'recwoo1',
               'rinkin1',
               'roahaw',
               'rosspo1',
               'royfly1',
               'rtlhum',
               'rubsee1',
               'rufmot1',
               'rugdov',
               'rumfly1',
               'ruther1',
               'rutjac1',
               'rutpuf1',
               'saffin',
               'sahpar1',
               'savhaw1',
               'secfly1',
               'shghum1',
               'shtfly1',
               'smbani',
               'snoegr',
               'sobtyr1',
               'socfly1',
               'solsan',
               'soulap1',
               'spbwoo1',
               'speowl1',
               'spepar1',
               'srwswa1',
               'stbwoo2',
               'strcuc1',
               'strfly1',
               'strher',
               'strowl1',
               'tbsfin1',
               'thbeup1',
               'thlsch3',
               'trokin',
               'tropar',
               'trsowl',
               'turvul',
               'verfly',
               'watjac1',
               'wbwwre1',
               'whbant1',
               'whbman1',
               'whfant1',
               'whmtyr1',
               'whtdov',
               'whttro1',
               'whwswa1',
               'woosto',
               'y00678',
               'yebela1',
               'yebfly1',
               'yebsee1',
               'yecspi2',
               'yectyr1',
               'yehbla2',
               'yehcar1',
               'yelori1',
               'yeofly1',
               'yercac1',
               'ywcpar']  # ⬅️ (Keep your list of 206 species here)

# Create Predictions DataFrame
wildnet_predictions = pd.DataFrame(predictions, columns=species_ids)
wildnet_predictions.insert(0, "row_id", all_row_ids)

# Save to CSV
wildnet_predictions.to_csv("wildnet_predictions.csv", index=False)
print("Predictions file created: wildnet_predictions.csv")