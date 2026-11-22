
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset\Testing"
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_custom_cnn_v3.keras"
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# --------------------------------------------------
# 2. Check model file
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("\nLoading Custom CNN V3 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Custom CNN V3 loaded successfully!")

# --------------------------------------------------
# 3. Load independent testing dataset
# --------------------------------------------------

print("\nLoading testing dataset...")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
    class_names=CLASS_NAMES
)

print("\nClass names:", test_data.class_names)

# --------------------------------------------------
# 4. Evaluate model
# --------------------------------------------------

print("\nEvaluating Custom CNN V3...")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\n========== CUSTOM CNN V3 TEST RESULTS ==========")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_data
])

y_prob = model.predict(test_data, verbose=1)

y_pred = np.argmax(y_prob, axis=1)

# --------------------------------------------------
# 6. Calculate performance metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n========== OVERALL PERFORMANCE ==========")

print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1-score    : {f1:.4f}")

# --------------------------------------------------
# 7. Classification report
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# --------------------------------------------------
# 9. Display confusion matrix
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("Custom CNN V3 - Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nCustom CNN V3 evaluation completed!")
