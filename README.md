# 🧠 Brain Tumor Classification Using Deep Learning

## 📌 Overview
This project uses Deep Learning to classify brain MRI images into four categories: **Glioma, Meningioma, Pituitary, and No Tumor**. It implements a fine-tuned MobileNetV2 model and three custom CNN architectures (V1, V2, and V3) to experiment with and compare classification performance.

A Flask web application allows users to upload MRI images, select a model, view prediction results and confidence scores, compare models, and generate downloadable PDF reports.

---

## 🎯 Objectives
- Classify brain MRI images into four categories.
- Implement transfer learning using MobileNetV2.
- Build and experiment with three custom CNN architectures.
- Compare model performance using standard evaluation metrics.
- Develop an interactive web interface using Flask.
- Generate downloadable PDF prediction reports.

---

## 🧠 Models Implemented

| Model | Description | Test Accuracy |
|---|---|---:|
| MobileNetV2 | Fine-tuned transfer learning model | **90.50%** |
| Custom CNN V1 | Baseline custom CNN architecture | **86.06%** |
| Custom CNN V2 | Global Average Pooling variant | 67.25% |
| Custom CNN V3 | CNN with augmentation and regularization techniques | 68.94% |

### Custom CNN V3 Techniques
- Data Augmentation
- Batch Normalization
- Global Average Pooling
- Dropout
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing

---

## 📊 Evaluation Metrics
Models are evaluated using:
- Accuracy
-
