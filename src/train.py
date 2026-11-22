from pathlib import Path

import mlflow
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers

# paths 
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(
    r"C:\Users\Poonakutty\Downloads\Brain_tumor_mri_dataset"
)
MODEL_DIR = PROJECT_DIR / "models"

# Image & training 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 
SEED = 42
NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# mlflow configure
MLFLOW_DB = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("Brain_Tumor_MRI_Classification")

print("MLflow tracking database:", MLFLOW_DB)


# Load training images: reserve 20% for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Training",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Training",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

# Keep the separate testing dataset for final evaluation
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "Testing",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    shuffle=False,
)

print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())
print("Detected class names:", train_ds.class_names if hasattr(train_ds, "class_names") else CLASS_NAMES)


# prepare datasets for efficient processing

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# load mobilenetv2 
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet",
)

# freeze pretrained layers for inital training
base_model.trainable = False

# build the classification model
inputs = keras.Input(shape=(224, 224, 3))

# mobileNetv2 expect pixel value to be -1 to 1
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x,training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(NUM_CLASSES,activation="softmax")(x)

model = keras.Model(inputs, outputs)

# model training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
 )

# model arch display
model.summary()

# training settings
EPOCHS = 10

# train the model & record the experiment
with mlflow.start_run(run_name="MobileNetV2_Initial_Taining"):
    mlflow.log_param("model","MobileNetV2")
    mlflow.log_param("image_size","224x224")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("dataset_calsses", CLASS_NAMES)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    # Save the trained model
    model_path = MODEL_DIR / "brain_tumor_mobilenetv2.keras"
    model.save(model_path)


    # Log final metrics
    final_train_accuracy = history.history["accuracy"][-1]
    final_validation_accuracy = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_validation_loss = history.history["val_loss"][-1]

    mlflow.log_metric("final_train_accuracy", final_train_accuracy)
    mlflow.log_metric("final_validation_accuracy", final_validation_accuracy)
    mlflow.log_metric("final_train_loss", final_train_loss)
    mlflow.log_metric("final_validation_loss", final_validation_loss)

    mlflow.log_artifact(str(model_path))

    print("\nTraining completed!")
    print(f"Final training accuracy: {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Model saved to: {model_path}")


# Fine-tuning: unfreeze the later MobileNetV2 layers
print("\nStarting fine-tuning...")

# Allow selected pretrained layers to be trained
base_model.trainable = True

# Keep earlier layers frozen; unfreeze only the final 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for more stable fine-tuning
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile with a much smaller learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

FINE_TUNE_EPOCHS = 5

with mlflow.start_run(run_name="MobileNetV2_Fine_Tuning"):
    mlflow.log_param("fine_tune_layers", 30)
    mlflow.log_param("fine_tune_epochs", FINE_TUNE_EPOCHS)
    mlflow.log_param("fine_tune_learning_rate", 0.00001)

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        verbose=1,
    )

    # Save as a separate model so the original remains available
    fine_tuned_path = MODEL_DIR / "brain_tumor_mobilenetv2_finetuned.keras"
    model.save(fine_tuned_path)

    mlflow.log_metric(
        "fine_tune_final_train_accuracy",
        fine_tune_history.history["accuracy"][-1],
    )
    mlflow.log_metric(
        "fine_tune_final_validation_accuracy",
        fine_tune_history.history["val_accuracy"][-1],
    )
    mlflow.log_artifact(str(fine_tuned_path))

    print("\nFine-tuning completed!")
    print(
        "Final fine-tuning validation accuracy: "
        f"{fine_tune_history.history['val_accuracy'][-1]:.4f}"
    )
    print(f"Fine-tuned model saved to: {fine_tuned_path}")
