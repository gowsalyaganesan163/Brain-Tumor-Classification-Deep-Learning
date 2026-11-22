
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(
    r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset"
)

TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "brain_tumor_custom_cnn_v3.keras"

# ==========================================
# 2. TRAINING CONFIGURATION
# ==========================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 20

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 3. LOAD DATASETS
# ==========================================

train_data = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    label_mode="int"
)

class_names = train_data.class_names

val_data = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    class_names=class_names,
    label_mode="int"
)

test_data = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names,
    label_mode="int"
)

print("\nClass names:", class_names)
print("Training batches:", len(train_data))
print("Validation batches:", len(val_data))
print("Testing batches:", len(test_data))

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

# ==========================================
# 5. BUILD CUSTOM CNN V3
# ==========================================

model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # Apply augmentation during training
    data_augmentation,

    # First convolution block
    layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification head
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
], name="Custom_CNN_V3")

model.summary()

# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCustom CNN V3 compiled successfully!")

# ==========================================
# 7. TRAINING CALLBACKS
# ==========================================

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

print("Training callbacks configured successfully!")

# ==========================================
# 8. TRAIN MODEL
# ==========================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ==========================================
# 9. FINAL TRAINING SUMMARY
# ==========================================

print("\nCustom CNN V3 training completed!")

print("Best model saved at:", MODEL_PATH)

best_epoch = max(
    range(len(history.history["val_accuracy"])),
    key=lambda i: history.history["val_accuracy"][i]
) + 1

print("Best validation accuracy:",
      round(max(history.history["val_accuracy"]) * 100, 2), "%")

print("Best validation epoch:", best_epoch)
