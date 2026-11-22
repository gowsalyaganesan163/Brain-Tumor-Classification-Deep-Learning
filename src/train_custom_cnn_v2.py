
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = DATASET_DIR = Path(r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset")
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

# Training configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123

# Load training and validation datasets
train_data = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_data = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    class_names=train_data.class_names
)

# Load independent test dataset
test_data = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=train_data.class_names
)

# Display dataset information
print("Class names:", train_data.class_names)
print("Training batches:", len(train_data))
print("Validation batches:", len(val_data))
print("Testing batches:", len(test_data))


# Build a custom CNN from scratch

model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),

    # Normalize pixel values from 0-255 to 0-1
    layers.Rescaling(1./255),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

# Display model architecture
model.summary()



# Compile the CNN model

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("CNN model compiled successfully!")



# Save the best model based on validation accuracy
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=PROJECT_DIR / "models" / "brain_tumor_custom_cnn_v2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Stop training if validation loss stops improving
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("Training callbacks configured successfully!")



# Train the custom CNN
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[checkpoint, early_stopping]
)
