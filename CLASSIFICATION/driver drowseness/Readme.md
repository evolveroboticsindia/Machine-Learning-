# 🚗 Driver Drowsiness Detection System

A deep learning pipeline for real-time driver drowsiness detection using transfer learning. Trains and benchmarks three CNN architectures — **MobileNetV2**, **ResNet50**, and **InceptionV3** — then deploys the best-performing model for single-image inference.

## Overview

This project implements a binary image classification pipeline to detect whether a driver is **drowsy** or **alert**. It uses pre-trained ImageNet weights via transfer learning and freezes the base model layers, training only the custom classification head.

After training all three architectures, the model with the highest validation accuracy is automatically saved as the **champion model** and used for inference and explainability analysis.

## How It Works

The pipeline runs end-to-end in the following stages:

### 1. 📥 Data Ingestion
Images are loaded using `tf.keras.preprocessing.image_dataset_from_directory` with an 80/20 train/validation split. Both splits are capped and class-balanced to manage memory efficiently.

### 2. 🏗️ Model Construction
A `build_model()` factory function constructs a transfer learning model for each architecture. The pretrained base (MobileNetV2 / ResNet50 / InceptionV3) is loaded with frozen ImageNet weights. A custom classification head — `GlobalAveragePooling2D → BatchNorm → Dense(64) → Dropout → Dense(1, sigmoid)` — is stacked on top and trained from scratch.

### 3. 🚀 Training Loop
Each model is trained for 3 epochs using the Adam optimizer with a low learning rate (`1e-4`) and binary cross-entropy loss. After each run, the model is evaluated on the stratified validation set and its metrics (accuracy, precision, recall, F1) are recorded. The model is then cleared from memory before the next architecture is trained.

### 4. 🏆 Champion Selection
After all three models are trained, the one with the highest validation accuracy is automatically saved to `outputs/best_model.keras` and designated the **champion model** for inference and explainability.

### 5. 📊 Visualization
Training curves (accuracy & loss) for all models are plotted side by side. A confusion matrix heatmap is generated for the champion model to show true vs. predicted class distributions.

### 6. 🧠 Explainability (XAI)
Two interpretability methods are applied to a sample validation image using the champion model:
- **LIME** segments the image into superpixels and perturbs them to identify which regions most influenced the prediction.
- **Gradient Saliency** uses `tf.GradientTape` to backpropagate the prediction signal to the input pixels, producing a heatmap of pixel-level importance.

### 7. 🔮 Inference
The `predict_custom_image()` function accepts any local image path, resizes it to `128×128`, runs it through the champion model, and displays the predicted class with a confidence score.

## Features

- ✅ Multi-model training and comparison (MobileNetV2, ResNet50, InceptionV3)
- ✅ Stratified, balanced validation subsets
- ✅ Automated champion model selection and saving
- ✅ Training curves (accuracy & loss) for all models
- ✅ Confusion matrix heatmap for the best model
- ✅ LIME superpixel attribution explanations
- ✅ Saliency map (gradient-based) visualization
- ✅ Single-image inference function with confidence score


## Dataset

The dataset contains labeled driver face images split into two classes:
- `Drowsy`
- `Non Drowsy` (Alert)

Place the dataset folder in your working directory before running the script. An 80/20 train/validation split is applied. Subsets are capped and balanced to manage memory during training.


## Model Architectures

All three models follow the same structure:

```
Input (128×128×3)
  └── Rescaling (1/255)
  └── Pretrained Base Model (frozen, ImageNet weights)
  └── GlobalAveragePooling2D
  └── BatchNormalization
  └── Dense(64, ReLU)
  └── Dropout(0.2)
  └── Dense(1, Sigmoid)  →  Binary output
```

| Model        | Base Params | Notes                        |
|--------------|-------------|------------------------------|
| MobileNetV2  | ~2.3M       | Lightweight, fast            |
| ResNet50     | ~23.5M      | Deep residual connections    |
| InceptionV3  | ~21.8M      | Multi-scale feature capture  |



## Setup & Installation

### 1. Open the notebook

Upload `driver_drowsiness.py` or open the notebook in your preferred environment.

### 2. Install dependencies

The script auto-installs missing packages at runtime:

```bash
pip install lime
```


## Usage

### Train & Evaluate All Models

Run the full script to train, evaluate, and compare all three architectures:

```bash
python driver_drowsiness.py
```

Or run all cells in the notebook sequentially.

### Run Inference on a Custom Image

Upload your image and update the path at the bottom of the script:

```python
test_path = "/content/your_image.jpg"
predict_custom_image(test_path)
```

The function outputs:
- A plot of the image with the predicted label
- Confidence score (as a percentage)


## Inference Entrypoint

At the bottom of the script, the live inference engine is triggered automatically on a test image. This is the final execution block:

```python
# ============================================================
# LIVE SINGLE-IMAGE INFERENCE EXECUTION ENTRYPOINT
# ============================================================

# 1. Define the path to your local test image
# Place your image in the working directory first.
test_path = "/content/shutterstock-1844338192-1404x1112.webp"

# 2. Fire the custom inference evaluation engine
# (This utilizes the trained champion model saved in the outputs folder)
predict_custom_image(test_path)
```

**To use your own image:**

1. Place your image file (`.jpg`, `.png`, `.webp`, etc.) in your working directory.
2. Update `test_path` with your file's path and run the cell.

**What `predict_custom_image()` does internally:**

| Step | Action |
|------|--------|
| File check | Verifies the image path exists before proceeding |
| Decode | Reads the raw file with `tf.io.read_file` and decodes it into a 3-channel tensor |
| Resize | Scales the image to `128×128` pixels to match the model's input shape |
| Batch | Wraps the tensor in a batch dimension `(1, 128, 128, 3)` |
| Predict | Runs the champion model and extracts the raw sigmoid probability |
| Interpret | Maps probability > 0.5 → `Drowsy`, else → `Non Drowsy` |
| Display | Renders the image with the predicted label and confidence score as a plot |

**Example output:**

```
🔮 Processing user inference image targeting: /content/shutterstock-1844338192-1404x1112.webp

Prediction: Drowsy
Confidence: 91.47%
```


## Output Files

| File | Description |
|------|-------------|
| `best_model.keras` | The champion model saved for inference |
| `training_performance_curves.png` | Accuracy and loss curves across all models |
| `confusion_matrix_heatmap.png` | True vs. predicted class matrix |
| `lime_explanation_plot.png` | LIME superpixel attribution for a sample image |
| `shap_like_saliency_heatmap.png` | Gradient saliency overlay on a sample image |


## Explainability (XAI)

Two interpretability techniques are applied to the best model on a sample validation image:

### LIME (Local Interpretable Model-agnostic Explanations)
Highlights the image regions (superpixels) that most influenced the model's prediction.

### Gradient Saliency Map
Uses `tf.GradientTape` to compute the gradient of the prediction with respect to the input pixels, revealing which areas the model focused on.


## Configuration

Key hyperparameters are defined at the top of the script:

```python
IMG_SIZE   = 128     # Input image resolution (pixels)
BATCH_SIZE = 16      # Training batch size
EPOCHS     = 3       # Training epochs per model
SEED       = 123     # Random seed for reproducibility
```

Training data is capped at **120 batches** and validation at **60 batches** to keep runtime manageable.


## Requirements

| Package | Purpose |
|---------|---------|
| `tensorflow` | Model building and training |
| `numpy` | Array operations |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Metrics (accuracy, F1, etc.) |
| `lime` | LIME explainability |
| `scikit-image` | Boundary visualization |


