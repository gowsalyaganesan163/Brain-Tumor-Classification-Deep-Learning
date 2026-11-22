import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATASET_DIR = Path(
    r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset"
)

TEST_DIR = DATASET_DIR / "Testing"

# Image settings
BATCH_SIZE = 32

# Class names
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]



# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES
)

print("\nTest dataset loaded successfully!")
print("Class names:", test_data.class_names)
print("Number of test images:", len(test_data.file_paths))


def evaluate_model(model, dataset, model_name):

    y_true = []
    y_pred = []

    for images, labels in dataset:

        # Generate predictions
        predictions = model.predict(images, verbose=0)

        # Convert probabilities into class indices
        predicted_classes = np.argmax(predictions, axis=1)

        # Store actual and predicted labels
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro"
    )

    recall = recall_score(
        y_true, y_pred, average="macro"
    )

    f1 = f1_score(
        y_true, y_pred, average="macro"
    )

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    print("Confusion Matrix:")

    print(confusion_matrix(y_true, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Model file paths
MODEL_PATHS = {
    "MobileNetV2": PROJECT_DIR / "models" / "brain_tumor_mobilenetv2_finetuned.keras",

    "Custom CNN V1": PROJECT_DIR / "models" / "brain_tumor_custom_cnn.keras",

    "Custom CNN V2": PROJECT_DIR / "models" / "brain_tumor_custom_cnn_v2.keras"
}

# Load trained models
models = {}

for model_name, model_path in MODEL_PATHS.items():

    print(f"\nLoading {model_name}...")

    model = tf.keras.models.load_model(model_path)

    models[model_name] = model

    print(f"{model_name} loaded successfully!")

    # Store comparison results
comparison_results = []

# Evaluate each model
for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    # Select correct image size
    if model_name == "MobileNetV2":
        image_size = (224, 224)
    else:
        image_size = (128, 128)

    # Resize test images for this model
    resized_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=CLASS_NAMES
    )

    # Evaluate model
    results = evaluate_model(
        model,
        resized_dataset,
        model_name
    )

    comparison_results.append(results)