
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset\Testing"

MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

BATCH_SIZE = 32

# --------------------------------------------------
# 2. Model paths
# --------------------------------------------------

MODEL_PATHS = {
    "MobileNetV2": os.path.join(
        MODEL_DIR,
        "brain_tumor_mobilenetv2_finetuned.keras"
    ),

    "Custom CNN V1": os.path.join(
        MODEL_DIR,
        "brain_tumor_custom_cnn.keras"
    ),

    "Custom CNN V2": os.path.join(
        MODEL_DIR,
        "brain_tumor_custom_cnn_v2.keras"
    ),

    "Custom CNN V3": os.path.join(
        MODEL_DIR,
        "brain_tumor_custom_cnn_v3.keras"
    )
}

# --------------------------------------------------
# 3. Load testing datasets
# --------------------------------------------------

def load_test_data(image_size):

    print(f"\nLoading test images at {image_size}...")

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=CLASS_NAMES
    )

    return dataset


# --------------------------------------------------
# 4. Evaluate one model
# --------------------------------------------------

def evaluate_model(model_name, model_path, image_size):

    print("\n" + "=" * 55)
    print(f"Evaluating {model_name}")
    print("=" * 55)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None

    print("Loading model...")

    model = tf.keras.models.load_model(model_path)

    test_data = load_test_data(image_size)

    # Actual labels
    y_true = np.concatenate([
        labels.numpy()
        for images, labels in test_data
    ])

    # Model predictions
    print("Generating predictions...")

    y_prob = model.predict(test_data, verbose=1)

    y_pred = np.argmax(y_prob, axis=1)

    # Calculate metrics
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

    print(f"\n{model_name} Results:")
    print(f"Accuracy         : {accuracy * 100:.2f}%")
    print(f"Macro Precision  : {precision:.4f}")
    print(f"Macro Recall     : {recall:.4f}")
    print(f"Macro F1-score   : {f1:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# --------------------------------------------------
# 5. Compare all four models
# --------------------------------------------------

def main():

    results = []

    # MobileNetV2 uses 224 x 224 images
    result = evaluate_model(
        "MobileNetV2",
        MODEL_PATHS["MobileNetV2"],
        (224, 224)
    )

    if result:
        results.append(result)

    # Custom CNN models use 128 x 128 images
    for model_name in [
        "Custom CNN V1",
        "Custom CNN V2",
        "Custom CNN V3"
    ]:

        result = evaluate_model(
            model_name,
            MODEL_PATHS[model_name],
            (128, 128)
        )

        if result:
            results.append(result)

    # --------------------------------------------------
    # 6. Final comparison table
    # --------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    if not results:
        print("No models were successfully evaluated.")
        return

    print(
        f"{'Model':<18}"
        f"{'Accuracy':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1-score':<12}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['Model']:<18}"
            f"{result['Accuracy']:<14.2f}"
            f"{result['Precision']:<14.4f}"
            f"{result['Recall']:<14.4f}"
            f"{result['F1-score']:<12.4f}"
        )

    # Identify best model by accuracy
    best_model = max(
        results,
        key=lambda item: item["Accuracy"]
    )

    print("\nBest model by test accuracy:")
    print(
        f"{best_model['Model']} - "
        f"{best_model['Accuracy']:.2f}%"
    )

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()
