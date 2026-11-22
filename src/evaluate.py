from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset")
MODEL_PATH = PROJECT_DIR / "models" / "brain_tumor_mobilenetv2_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Evaluating model:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

test_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Testing",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=False,
)

# IMPORTANT: Do not apply preprocess_input here. It is embedded in the model
# built by train.py, so image_dataset_from_directory's raw 0..255 tensors are
# the correct inputs.
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print(f"\nTest accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))))
