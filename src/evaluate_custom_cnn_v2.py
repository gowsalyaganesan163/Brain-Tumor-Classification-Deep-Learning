
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(
    r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset"
)

TEST_DIR = DATASET_DIR / "Testing"
MODEL_PATH = PROJECT_DIR / "models" / "brain_tumor_custom_cnn_v2.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Use the same class order as training
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Load independent test dataset
test_data = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions
y_true = np.concatenate([
    labels.numpy() for images, labels in test_data
])

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# Confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
