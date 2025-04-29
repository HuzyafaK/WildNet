import tensorflow as tf
from tensorflow.keras import backend as K

#loss_fn
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

# Model compilation
model = build_WildNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss_multilabel(),  # Using custom focal loss function
    metrics=[
        tf.keras.metrics.AUC(name='auc', multi_label=True),
        tf.keras.metrics.PrecisionAtRecall(0.5, name='precision'),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.RecallAtPrecision(0.5, name='recall')
    ]
)

model.summary()

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)

#save_model
model.save('WildNet.keras') # Saves in .keras format
model.save('WildNet')  # Saves in the TensorFlow SavedModel format
