# =========================================================
# 🌦️ INDUSTRY LEVEL WEATHER PREDICTION SYSTEM
# =========================================================

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import joblib
import os

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("data/weather_forecast_data.csv")

print("\n✅ Dataset Loaded Successfully")
print(df.head())

# =========================================================
# BASIC INFO
# =========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# REMOVE DUPLICATES
# =========================================================

df.drop_duplicates(inplace=True)

# =========================================================
# TARGET ENCODING
# =========================================================

encoder = LabelEncoder()

df["Rain"] = encoder.fit_transform(df["Rain"])

# Save encoder
joblib.dump(encoder, "models/label_encoder.pkl")

# =========================================================
# FEATURE & TARGET SPLIT
# =========================================================

X = df.drop("Rain", axis=1)
y = df["Rain"]

# =========================================================
# EDA - CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig(
    "outputs/correlation_heatmap.png"
)

plt.close()

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# HANDLE IMBALANCE USING SMOTE
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =========================================================
# MODEL DEFINITIONS
# =========================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

# =========================================================
# MODEL TRAINING & EVALUATION
# =========================================================

results = {}

best_model = None
best_f1 = 0

for name, model in models.items():

    print(f"\n==============================")
    print(f"Training: {name}")
    print(f"==============================")

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc = roc_auc_score(y_test, y_pred)

    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring='f1'
    ).mean()

    results[name] = {
        "Accuracy": accuracy,
        "F1": f1,
        "ROC-AUC": roc,
        "CV Score": cv_score
    }

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"CV Score : {cv_score:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    "models/best_weather_model.pkl"
)

print("\n✅ Best Model Saved Successfully")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X.columns,
        "Importance": best_model.feature_importances_

    }).sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance
    )

    plt.title("Feature Importance")

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()

# =========================================================
# CONFUSION MATRIX
# =========================================================

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# =========================================================
# FINAL RESULTS
# =========================================================

results_df = pd.DataFrame(results).T

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(results_df)

print("\n✅ Training Completed Successfully")