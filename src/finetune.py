
from pathlib import Path

import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(
    r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset"
)
MODEL_DIR = PROJECT_DIR / "models"

INITIAL_MODEL_PATH = MODEL_DIR / "brain_tumor_mobilenetv2_initial.keras"
FINE_TUNED_MODEL_PATH = MODEL_DIR / "brain_tumor_mobilenetv2_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Check that the backup exists
if not INITIAL_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Backup model not found: {INITIAL_MODEL_PATH}"
    )

# Load training and validation images
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Training",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Training",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

# Load the held-out test set without shuffling
test_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Testing",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=False,
)

# Efficient input pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Load the already-trained model; do not train from scratch
print("\nLoading the existing trained model...")
model = keras.models.load_model(INITIAL_MODEL_PATH)

# Find the MobileNetV2 backbone inside the saved model
base_model = None
for layer in model.layers:
    if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise RuntimeError(
        "Could not find the MobileNetV2 backbone in the saved model. "
        "Stop here and inspect model.summary() before proceeding."
    )

print("Loaded model:", INITIAL_MODEL_PATH)
print("Backbone:", base_model.name)

# Unfreeze the final 30 backbone layers, leaving earlier layers frozen
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Use a small learning rate to adjust pretrained weights gently
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nTrainable parameters after fine-tuning setup:")
model.summary()

# MLflow setup
MLFLOW_DB = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("Brain_Tumor_MRI_Classification")

# Fine-tune for up to 5 epochs; retain the best validation-loss weights
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(FINE_TUNED_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]

with mlflow.start_run(run_name="MobileNetV2_Fine_Tuning"):
    mlflow.log_param("fine_tune_layers", 30)
    mlflow.log_param("fine_tune_epochs_max", 5)
    mlflow.log_param("learning_rate", 1e-5)
    mlflow.log_param("batch_size", BATCH_SIZE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model with the best validation weights
    model.save(FINE_TUNED_MODEL_PATH)
    mlflow.log_artifact(str(FINE_TUNED_MODEL_PATH))

    print("\nFine-tuning finished.")
    print("Saved model:", FINE_TUNED_MODEL_PATH)

# Evaluate the saved fine-tuned model on the separate test set
print("\nEvaluating fine-tuned model on the test set...")
best_model = keras.models.load_model(FINE_TUNED_MODEL_PATH)

y_true = []
y_pred = []

for images, labels in test_ds:
    scores = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(scores.argmax(axis=1).tolist())

print("\nFine-tuned test accuracy:", f"{sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true):.4f}")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
