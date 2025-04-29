import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


def build_WildNet():
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


# Example usage
model = build_WildNet()