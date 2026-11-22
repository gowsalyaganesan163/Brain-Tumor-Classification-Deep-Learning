
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from tensorflow import keras


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

DISCLAIMER = (
    "Educational research only. This AI output is not a medical diagnosis, "
    "and displayed scores are not calibrated medical probabilities. Do not "
    "use this application to make medical decisions. Consult a qualified "
    "medical professional for clinical interpretation."
)

# Test-set metrics from your completed model evaluation.
MODEL_INFO = {
    "mobilenet": {
        "name": "MobileNetV2",
        "filename": "brain_tumor_mobilenetv2_finetuned.keras",
        "image_size": (224, 224),
        "accuracy": 90.50,
    },
    "custom_cnn": {
        "name": "Custom CNN V1",
        "filename": "brain_tumor_custom_cnn.keras",
        "image_size": (128, 128),
        "accuracy": 86.06,
    },
}

MODEL_CHOICES = {
    "mobilenet": ["mobilenet"],
    "custom_cnn": ["custom_cnn"],
    "compare": ["mobilenet", "custom_cnn"],
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


# --------------------------------------------------
# 2. Load both trained models
# --------------------------------------------------

models = {}

for model_key, info in MODEL_INFO.items():
    model_path = MODELS_DIR / info["filename"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please check that the trained model file exists."
        )

    print(f"Loading {info['name']}...")
    models[model_key] = keras.models.load_model(model_path)
    print(f"{info['name']} loaded successfully.")


# --------------------------------------------------
# 3. Image preparation
# --------------------------------------------------

def prepare_uploaded_image(uploaded_file):
    """
    Read the uploaded MRI image and return a clean RGB image.

    Resizing is performed separately for each model because:
    MobileNetV2 uses 224 x 224.
    Custom CNN V1 uses 128 x 128.

    Both models have their preprocessing layers inside their
    respective model architectures. Do not preprocess again here.
    """
    try:
        with Image.open(uploaded_file.stream) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            report_image = image.copy()

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid or unsupported image file.") from exc

    return image, report_image


def prepare_image_for_model(image, model_key):
    """Resize an image to the input size required by a model."""
    image_size = MODEL_INFO[model_key]["image_size"]

    resized = image.resize(image_size)
    array = np.asarray(resized, dtype=np.float32)

    # Add batch dimension.
    array = np.expand_dims(array, axis=0)

    return array


# --------------------------------------------------
# 4. Prediction functions
# --------------------------------------------------

def run_single_prediction(image, model_key):
    """Run prediction using one selected model."""

    model = models[model_key]
    model_input = prepare_image_for_model(image, model_key)

    probabilities = model.predict(model_input, verbose=0)[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"{MODEL_INFO[model_key]['name']} returned an unexpected "
            "number of class scores."
        )

    predicted_index = int(np.argmax(probabilities))

    scores = {
        class_name: float(score)
        for class_name, score in zip(CLASS_NAMES, probabilities)
    }

    return {
        "model_name": MODEL_INFO[model_key]["name"],
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "scores": scores,
        "test_accuracy": MODEL_INFO[model_key]["accuracy"],
    }


def run_selected_predictions(image, model_choice):
    """Run one or both models, depending on the user's selection."""

    if model_choice not in MODEL_CHOICES:
        raise ValueError("Invalid model selection.")

    results = {}

    for model_key in MODEL_CHOICES[model_choice]:
        results[model_key] = run_single_prediction(image, model_key)

    return results


def read_model_choice():
    """Read and validate the model selection sent by the website."""

    choice = request.form.get("model_choice", "mobilenet").strip()

    if choice not in MODEL_CHOICES:
        raise ValueError("Please select a valid model option.")

    return choice


# --------------------------------------------------
# 5. Patient details for PDF report
# --------------------------------------------------

def read_patient_details():
    patient_name = (
        request.form.get("patientName", "").strip()[:100]
        or "Not provided"
    )

    reference_id = (
        request.form.get("referenceId", "").strip()[:100]
        or "Not provided"
    )

    age = request.form.get("age", "").strip()
    sex = request.form.get("sex", "").strip()

    if age and (not age.isdigit() or not 0 <= int(age) <= 120):
        raise ValueError("Age must be a whole number between 0 and 120.")

    if sex not in ("", "Female", "Male", "Other"):
        raise ValueError("Invalid sex selection.")

    return {
        "Patient name": patient_name,
        "Reference ID": reference_id,
        "Age": age or "Not provided",
        "Sex": sex or "Not provided",
    }


# --------------------------------------------------
# 6. Website home page
# --------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 7. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "available_models": {
            key: {
                "name": info["name"],
                "test_accuracy": info["accuracy"],
                "image_size": info["image_size"],
            }
            for key, info in MODEL_INFO.items()
        },
        "class_names": CLASS_NAMES,
    })


# --------------------------------------------------
# 8. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the 'image' field."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        model_choice = read_model_choice()
        image, _ = prepare_uploaded_image(uploaded_file)
        results = run_selected_predictions(image, model_choice)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Keep the old response fields for single-model compatibility.
    response = {
        "model_choice": model_choice,
        "results": results,
        "disclaimer": DISCLAIMER,
    }

    if len(results) == 1:
        single_result = next(iter(results.values()))
        response.update({
            "predicted_class": single_result["predicted_class"],
            "scores": single_result["scores"],
            "model_file": next(
                MODEL_INFO[key]["filename"] for key in results
            ),
        })

    return jsonify(response)


# --------------------------------------------------
# 9. PDF report generation
# --------------------------------------------------

@app.post("/report")
def generate_report():

    if "image" not in request.files:
        return jsonify({"error": "Please upload an MRI image."}), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        model_choice = read_model_choice()
        image, report_image = prepare_uploaded_image(uploaded_file)
        patient_details = read_patient_details()
        results = run_selected_predictions(image, model_choice)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Create the PDF in memory.
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph("Brain MRI Classification Report", styles["Title"]),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Generated: "
            + escape(
                datetime.now().astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
            ),
            styles["Normal"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Patient Information", styles["Heading2"]),
    ]

    # Patient information table.
    patient_rows = [["Field", "Value"]]
    patient_rows.extend(
        [[key, value] for key, value in patient_details.items()]
    )

    patient_table = Table(
        patient_rows,
        colWidths=[5 * cm, 11 * cm],
        repeatRows=1,
    )

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.extend([
        patient_table,
        Spacer(1, 0.5 * cm),
        Paragraph("Uploaded MRI Image", styles["Heading2"]),
        Spacer(1, 0.2 * cm),
    ])

    # Add uploaded image to PDF.
    image_buffer = BytesIO()
    report_image.thumbnail((1000, 1000))
    report_image.save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    story.append(
        ReportImage(
            image_buffer,
            width=8 * cm,
            height=8 * cm,
            kind="proportional",
        )
    )

    story.extend([
        Spacer(1, 0.5 * cm),
        Paragraph("Model Prediction Results", styles["Heading2"]),
        Spacer(1, 0.2 * cm),
    ])

    # Prediction results table.
    prediction_rows = [[
        "Model",
        "Predicted Class",
        "Confidence",
        "Test Accuracy",
    ]]

    for result in results.values():
        prediction_rows.append([
            result["model_name"],
            result["predicted_class"],
            f"{result['confidence'] * 100:.2f}%",
            f"{result['test_accuracy']:.2f}%",
        ])

    prediction_table = Table(
        prediction_rows,
        colWidths=[4.3 * cm, 4.2 * cm, 3.5 * cm, 4 * cm],
        repeatRows=1,
    )

    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.extend([
        prediction_table,
        Spacer(1, 0.4 * cm),
        Paragraph("Class Scores", styles["Heading2"]),
        Spacer(1, 0.2 * cm),
    ])

    # Include individual class scores for every selected model.
    for result in results.values():
        story.append(
            Paragraph(
                escape(result["model_name"]),
                styles["Heading3"],
            )
        )

        score_rows = [["Class", "Score"]]

        for class_name, score in result["scores"].items():
            score_rows.append([
                class_name,
                f"{score * 100:.2f}%",
            ])

        score_table = Table(
            score_rows,
            colWidths=[8 * cm, 8 * cm],
        )

        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([
            score_table,
            Spacer(1, 0.3 * cm),
        ])

    # Educational disclaimer.
    story.extend([
        Spacer(1, 0.3 * cm),
        Paragraph("Important Disclaimer", styles["Heading2"]),
        Paragraph(escape(DISCLAIMER), styles["Normal"]),
    ])

    # Build the PDF document.
    try:
        document.build(story)
        pdf_buffer.seek(0)

    except Exception:
        app.logger.exception("PDF generation failed.")
        return jsonify({
            "error": "Unable to generate the PDF report."
        }), 500

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="brain_mri_classification_report.pdf",
    )


# --------------------------------------------------
# 10. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
