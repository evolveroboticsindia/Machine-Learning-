import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3

# Safe Dependency Verification Installs
try:
    import kagglehub
except ImportError:
    !pip install kagglehub
    import kagglehub

try:
    import lime
except ImportError:
    !pip install lime
    import lime

from lime import lime_image
from skimage.segmentation import mark_boundaries

# ============================================================
# CONFIGURATION & REFRESH OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "outputs"

if os.path.exists(OUTPUT_DIR):
    print("Resetting output directory...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 3
SEED = 123

# ============================================================
# DATA ENGINE ACCESS
# ============================================================

print("\n⚡ ACCESSING KAGGLE DATASTREAM...")
raw_download_path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")

if "Driver Drowsiness Dataset (DDD)" in os.listdir(raw_download_path):
    DATASET_PATH = os.path.join(raw_download_path, "Driver Drowsiness Dataset (DDD)")
else:
    DATASET_PATH = raw_download_path

print("\n📂 PARTITIONING BALANCED SUBSETS...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary', shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary', shuffle=True
)

CLASS_NAMES = train_ds.class_names

# ============================================================
# STRATIFIED MEMORY CAPPER
# ============================================================
print("⚖️ Balancing working subsets...")

train_ds = train_ds.take(120)

val_images, val_labels = [], []
for imgs, lbls in val_ds.take(60):
    val_images.append(imgs.numpy())
    val_labels.append(lbls.numpy())

val_images = np.concatenate(val_images, axis=0)
val_labels = np.concatenate(val_labels, axis=0).flatten()

idx_class0 = np.where(val_labels == 0)[0][:240]
idx_class1 = np.where(val_labels == 1)[0][:240]
balanced_indices = np.concatenate([idx_class0, idx_class1])
np.random.seed(SEED)
np.random.shuffle(balanced_indices)

stratified_val_images = val_images[balanced_indices]
stratified_val_labels = val_labels[balanced_indices]

final_val_ds = tf.data.Dataset.from_tensor_slices((stratified_val_images, stratified_val_labels)).batch(BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
final_val_ds = final_val_ds.prefetch(AUTOTUNE)

# ============================================================
# ARCHITECTURE BUILDER
# ============================================================

def build_model(base_init_fn):
    tf.keras.backend.clear_session()
    gc.collect()

    base_model = base_init_fn()
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

models_dict = {
    "MobileNetV2": lambda: MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'),
    "ResNet50": lambda: ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'),
    "InceptionV3": lambda: InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
}

# ============================================================
# MAIN TRAINING AND EVALUATION EXECUTION LOOP
# ============================================================

results = {}
histories = {}
best_acc = 0
best_name = ""
best_model_path = os.path.join(OUTPUT_DIR, "best_model.keras")

for name, base_init_fn in models_dict.items():
    print("\n" + "="*50)
    print(f"🚀 ACTIVE EXECUTION RUN: {name}")
    print("="*50)

    model = build_model(base_init_fn)
    history = model.fit(train_ds, validation_data=final_val_ds, epochs=EPOCHS, verbose=1)
    histories[name] = history.history

    print(f"📋 Running evaluation loop for {name}...")
    y_prob = model.predict(stratified_val_images, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = stratified_val_labels.astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    results[name] = metrics

    if metrics["Accuracy"] > best_acc:
        best_acc = metrics["Accuracy"]
        best_name = name
        model.save(best_model_path)

    del model
    tf.keras.backend.clear_session()
    gc.collect()

# ============================================================
# VISUALIZATION SUITE ENGINE
# ============================================================

print("\n📈 GENERATING PERFORMANCE GRAPHS...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name in histories:
    axes[0].plot(histories[name]['accuracy'], linestyle='--', label=f'{name} Train')
    axes[0].plot(histories[name]['val_accuracy'], linestyle='-', label=f'{name} Val')
axes[0].set_title('Architecture Accuracy Profiles')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy Score')
axes[0].legend()
axes[0].grid(True)

for name in histories:
    axes[1].plot(histories[name]['loss'], linestyle='--', label=f'{name} Train')
    axes[1].plot(histories[name]['val_loss'], linestyle='-', label=f'{name} Val')
axes[1].set_title('Architecture Loss Decay Curves')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss Magnitude')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_performance_curves.png"))
plt.show()

# --- RELOAD CHAMPION MODEL FOR PERFORMANCE MATRIX ---
best_model = tf.keras.models.load_model(best_model_path)
y_prob = best_model.predict(stratified_val_images, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)
y_true = stratified_val_labels.astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix Heatmap: {best_name}")
plt.xlabel("Predicted Label Status")
plt.ylabel("Ground Truth Target Status")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"), bbox_inches='tight')
plt.show()

# ============================================================
# EXPLAINABLE AI VISUALIZATIONS (LIME & SHAP-LIKE SALIENCY)
# ============================================================

print("\n🧠 GENERATING MODEL INTERPRETABILITY HEATMAPS...")
sample_image = stratified_val_images[0]

# LIME
explainer = lime_image.LimeImageExplainer()

def predict_fn(images):
    images = images.astype(np.float32) / 255.0
    preds = best_model.predict(images, verbose=0)
    return np.concatenate([1 - preds, preds], axis=1)

explanation = explainer.explain_instance(
    sample_image.astype(np.uint8), predict_fn, top_labels=1, hide_color=0, num_samples=40
)
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=False
)

lime_result = mark_boundaries(temp / 255.0, mask)
plt.figure(figsize=(6, 6))
plt.imshow(lime_result)
plt.axis('off')
plt.title(f"LIME Feature Attribution Superpixels ({best_name})")
plt.savefig(os.path.join(OUTPUT_DIR, "lime_explanation_plot.png"), bbox_inches='tight')
plt.show()

# SHAP-Like Saliency Map
img_batch = np.expand_dims(sample_image, axis=0)

with tf.GradientTape() as tape:
    inputs = tf.cast(img_batch, tf.float32)
    tape.watch(inputs)
    preds = best_model(inputs, training=False)

grads = tape.gradient(preds, inputs)
heatmap = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()[0]

heatmap = np.maximum(heatmap, 0)
if np.max(heatmap) != 0:
    heatmap = heatmap / np.max(heatmap)

plt.figure(figsize=(6, 6))
plt.imshow(sample_image.astype(np.uint8))
plt.imshow(heatmap, cmap='jet', alpha=0.45)
plt.axis('off')
plt.title(f"Saliency Map Gradient Track Vector ({best_name})")
plt.savefig(os.path.join(OUTPUT_DIR, "shap_like_saliency_heatmap.png"), bbox_inches='tight')
plt.show()

# ============================================================
# COMPREHENSIVE LEADERBOARD OVERVIEW
# ============================================================

print("\n" + "="*65)
print("🏆 COMPARISON MATRIX FOR EACH EXPERIMENTAL MODEL")
print("="*65)
print(f"{'Model Name':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)
for model_name, metrics in results.items():
    # FIX: Corrected format specifier placements to make sure it aligns floats cleanly without crashing
    print(f"{model_name:<18} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f}")
print("="*65)
print(f"🥇 DEPLOYMENT TARGET INSTANTIATED: {best_name}")
print("="*65)


# ============================================================
# LIVE SINGLE-IMAGE INFERENCE ENGINE FUNCTION
# ============================================================

def predict_custom_image(image_path):
    """Loads a single local raw image file and determines the drowsiness state"""
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Could not locate inference target file at: '{image_path}'")
        return

    print(f"\n🔮 Processing user inference image targeting: {image_path}")

    # Process file pathways raw image vector conversion sequences
    raw_img = tf.io.read_file(image_path)
    decoded_tensor = tf.image.decode_image(raw_img, channels=3, expand_animations=False)
    resized_tensor = tf.image.resize(decoded_tensor, [IMG_SIZE, IMG_SIZE])

    # Cast to batch shape array wrapper
    inference_batch = np.expand_dims(resized_tensor.numpy(), axis=0)

    # Generate prediction vector probability matrix
    raw_probability = best_model.predict(inference_batch, verbose=0)[0][0]

    # Map probability metrics across target class text spaces
    predicted_class_idx = int(raw_probability > 0.5)
    confidence_score = raw_probability if predicted_class_idx == 1 else (1.0 - raw_probability)
    predicted_label_text = CLASS_NAMES[predicted_class_idx]
    
    # Render final diagnostic plot visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(resized_tensor.numpy().astype(np.uint8))
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label_text}\nConfidence: {confidence_score*100:.2f}%", fontsize=12, fontweight='bold')
    plt.show()
test_path = "/content/shutterstock-1844338192-1404x1112.webp"
predict_custom_image(test_path)