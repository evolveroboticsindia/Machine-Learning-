# # =========================================================
# # 🌦️ INDUSTRY LEVEL WEATHER PREDICTION SYSTEM (2026)
# # =========================================================

# import warnings
# warnings.filterwarnings("ignore")

# # =========================================================
# # IMPORT LIBRARIES
# # =========================================================

# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# import logging
# import joblib

# from sklearn.model_selection import (
#     train_test_split,
#     GridSearchCV,
#     cross_val_score,
#     StratifiedKFold
# )

# from sklearn.preprocessing import LabelEncoder

# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     roc_auc_score,
#     RocCurveDisplay
# )

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.pipeline import Pipeline

# from xgboost import XGBClassifier

# from imblearn.over_sampling import SMOTE

# # =========================================================
# # CREATE FOLDERS
# # =========================================================

# os.makedirs("models", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)

# # =========================================================
# # LOGGING CONFIGURATION
# # =========================================================

# logging.basicConfig(
#     filename="outputs/project.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logging.info("Weather Prediction Project Started")

# # =========================================================
# # LOAD DATASET
# # =========================================================

# df = pd.read_csv("data/weather_forecast_data.csv")

# print("\n✅ Dataset Loaded Successfully")
# print(df.head())

# logging.info("Dataset Loaded Successfully")

# # =========================================================
# # BASIC DATASET INFO
# # =========================================================

# print("\nDataset Shape:")
# print(df.shape)

# print("\nDataset Info:")
# print(df.info())

# print("\nMissing Values:")
# print(df.isnull().sum())

# logging.info(f"Dataset Shape: {df.shape}")

# # =========================================================
# # REMOVE DUPLICATES
# # =========================================================

# df.drop_duplicates(inplace=True)

# # =========================================================
# # TARGET ENCODING
# # =========================================================

# encoder = LabelEncoder()

# df["Rain"] = encoder.fit_transform(df["Rain"])

# joblib.dump(
#     encoder,
#     "models/label_encoder.pkl"
# )

# logging.info("Target Encoding Completed")

# # =========================================================
# # FEATURE & TARGET SPLIT
# # =========================================================

# X = df.drop("Rain", axis=1)
# y = df["Rain"]

# features = X.columns

# # =========================================================
# # EDA - DISTRIBUTION PLOTS
# # =========================================================

# for feature in features:

#     plt.figure(figsize=(6, 4))

#     sns.histplot(
#         df[feature],
#         kde=True
#     )

#     plt.title(f"{feature} Distribution")

#     plt.savefig(
#         f"outputs/{feature}_distribution.png"
#     )

#     plt.close()

# logging.info("Distribution Plots Saved")

# # =========================================================
# # EDA - BOXPLOTS
# # =========================================================

# for feature in features:

#     plt.figure(figsize=(6, 4))

#     sns.boxplot(
#         x=df["Rain"],
#         y=df[feature]
#     )

#     plt.title(f"{feature} vs Rain")

#     plt.savefig(
#         f"outputs/{feature}_boxplot.png"
#     )

#     plt.close()

# logging.info("Boxplots Saved")

# # =========================================================
# # CORRELATION HEATMAP
# # =========================================================

# plt.figure(figsize=(10, 6))

# sns.heatmap(
#     df.corr(),
#     annot=True,
#     cmap="coolwarm"
# )

# plt.title("Feature Correlation Heatmap")

# plt.savefig(
#     "outputs/correlation_heatmap.png"
# )

# plt.close()

# logging.info("Correlation Heatmap Saved")

# # =========================================================
# # TRAIN TEST SPLIT
# # =========================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )

# # =========================================================
# # HANDLE IMBALANCE USING SMOTE
# # =========================================================

# smote = SMOTE(random_state=42)

# X_train_smote, y_train_smote = smote.fit_resample(
#     X_train,
#     y_train
# )

# print("\nBefore SMOTE:")
# print(y_train.value_counts())

# print("\nAfter SMOTE:")
# print(pd.Series(y_train_smote).value_counts())

# logging.info("SMOTE Applied Successfully")

# # =========================================================
# # STRATIFIED K-FOLD
# # =========================================================

# skf = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# # =========================================================
# # MODEL DEFINITIONS
# # =========================================================

# models = {

#     "Decision Tree": DecisionTreeClassifier(
#         random_state=42
#     ),

#     "Random Forest": RandomForestClassifier(
#         random_state=42
#     ),

#     "XGBoost": XGBClassifier(
#         eval_metric='logloss',
#         random_state=42
#     )
# }

# # =========================================================
# # HYPERPARAMETER TUNING - RANDOM FOREST
# # =========================================================

# print("\n🔍 Hyperparameter Tuning Started...")

# rf_params = {

#     "n_estimators": [100, 200],

#     "max_depth": [5, 10, None],

#     "min_samples_split": [2, 5]
# }

# grid = GridSearchCV(

#     estimator=RandomForestClassifier(
#         random_state=42
#     ),

#     param_grid=rf_params,

#     cv=3,

#     scoring='f1',

#     n_jobs=-1
# )

# grid.fit(
#     X_train_smote,
#     y_train_smote
# )

# best_rf = grid.best_estimator_

# print("\n✅ Best Random Forest Parameters:")
# print(grid.best_params_)

# logging.info(f"Best RF Params: {grid.best_params_}")

# # Replace original Random Forest
# models["Random Forest"] = best_rf

# # =========================================================
# # MODEL TRAINING & EVALUATION
# # =========================================================

# results = {}

# best_model = None
# best_f1 = 0

# for name, model in models.items():

#     print(f"\n==============================")
#     print(f"Training: {name}")
#     print(f"==============================")

#     # =====================================================
#     # PIPELINE
#     # =====================================================

#     pipeline = Pipeline([
#         ("model", model)
#     ])

#     pipeline.fit(
#         X_train_smote,
#         y_train_smote
#     )

#     y_pred = pipeline.predict(X_test)

#     # =====================================================
#     # METRICS
#     # =====================================================

#     accuracy = accuracy_score(
#         y_test,
#         y_pred
#     )

#     f1 = f1_score(
#         y_test,
#         y_pred
#     )

#     roc = roc_auc_score(
#         y_test,
#         y_pred
#     )

#     cv_score = cross_val_score(
#         pipeline,
#         X,
#         y,
#         cv=skf,
#         scoring='f1'
#     ).mean()

#     # =====================================================
#     # SAVE RESULTS
#     # =====================================================

#     results[name] = {

#         "Accuracy": accuracy,

#         "F1 Score": f1,

#         "ROC-AUC": roc,

#         "CV Score": cv_score
#     }

#     # =====================================================
#     # PRINT RESULTS
#     # =====================================================

#     print(f"\nAccuracy : {accuracy:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#     print(f"ROC-AUC  : {roc:.4f}")
#     print(f"CV Score : {cv_score:.4f}")

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

#     logging.info(
#         f"{name} -> Accuracy: {accuracy}, F1: {f1}"
#     )

#     # =====================================================
#     # SAVE BEST MODEL
#     # =====================================================

#     if f1 > best_f1:

#         best_f1 = f1

#         best_model = pipeline

# # =========================================================
# # SAVE BEST MODEL
# # =========================================================

# joblib.dump(
#     best_model,
#     "models/best_weather_model.pkl"
# )

# print("\n✅ Best Model Saved Successfully")

# logging.info("Best Model Saved")

# # =========================================================
# # FEATURE IMPORTANCE
# # =========================================================

# model_obj = best_model.named_steps["model"]

# if hasattr(model_obj, "feature_importances_"):

#     importance = pd.DataFrame({

#         "Feature": X.columns,

#         "Importance": model_obj.feature_importances_

#     })

#     importance = importance.sort_values(
#         by="Importance",
#         ascending=False
#     )

#     print("\nFeature Importance:")
#     print(importance)

#     plt.figure(figsize=(10, 6))

#     sns.barplot(
#         x="Importance",
#         y="Feature",
#         data=importance
#     )

#     plt.title("Feature Importance")

#     plt.savefig(
#         "outputs/feature_importance.png"
#     )

#     plt.close()

# logging.info("Feature Importance Saved")

# # =========================================================
# # CONFUSION MATRIX
# # =========================================================

# y_pred = best_model.predict(X_test)

# cm = confusion_matrix(
#     y_test,
#     y_pred
# )

# plt.figure(figsize=(6, 5))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues"
# )

# plt.title("Confusion Matrix")

# plt.xlabel("Predicted")
# plt.ylabel("Actual")

# plt.savefig(
#     "outputs/confusion_matrix.png"
# )

# plt.close()

# logging.info("Confusion Matrix Saved")

# # =========================================================
# # ROC CURVE
# # =========================================================

# RocCurveDisplay.from_estimator(
#     best_model,
#     X_test,
#     y_test
# )

# plt.savefig(
#     "outputs/roc_curve.png"
# )

# plt.close()

# logging.info("ROC Curve Saved")

# # =========================================================
# # SAVE RESULTS CSV
# # =========================================================

# results_df = pd.DataFrame(results).T

# results_df.to_csv(
#     "outputs/model_results.csv"
# )

# # =========================================================
# # FINAL RESULTS
# # =========================================================

# print("\n==============================")
# print("FINAL MODEL COMPARISON")
# print("==============================")

# print(results_df)

# logging.info("Training Completed Successfully")

# print("\n✅ Training Completed Successfully")


# =========================================================
# 🌦️ NEXT-GEN INDUSTRY WEATHER PREDICTION SYSTEM (2026)
# =========================================================
#
# FEATURES:
# ✅ Advanced Missing Value Handling
# ✅ Feature Engineering
# ✅ Leakage-Safe ColumnTransformer Pipeline
# ✅ Label Encoding + OneHotEncoding
# ✅ SMOTE Balancing
# ✅ Before/After Imbalance Visualization
# ✅ Hyperparameter Optimization
# ✅ Cross Validation
# ✅ Multi-Model Comparison
# ✅ SHAP Explainability
# ✅ Feature Importance
# ✅ ROC Curve
# ✅ Confusion Matrix
# ✅ Logging System
# ✅ Model Persistence
# ✅ Production-Level Workflow
#
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import logging
import joblib
import shap

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    filename="outputs/project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Weather Prediction Project Started")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    r"C:\Users\USER\Desktop\PROJECT\weather_prediction_projects\data\weatherAUS.csv"
)

print("\n✅ Dataset Loaded Successfully")

print(df.head())

logging.info("Dataset Loaded Successfully")

# =========================================================
# DATASET INFO
# =========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# REMOVE DUPLICATES
# =========================================================

df.drop_duplicates(inplace=True)

# =========================================================
# REMOVE NULL TARGETS
# =========================================================

df = df.dropna(subset=["RainTomorrow"])

# =========================================================
# DROP USELESS COLUMNS
# =========================================================

drop_cols = ["Date", "row ID"]

for col in drop_cols:

    if col in df.columns:

        df.drop(col, axis=1, inplace=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\n⚙️ Performing Feature Engineering...")

df["TempDifference"] = (
    df["MaxTemp"] - df["MinTemp"]
)

df["PressureDifference"] = (
    df["Pressure9am"] - df["Pressure3pm"]
)

df["HumidityDifference"] = (
    df["Humidity9am"] - df["Humidity3pm"]
)

df["WindSpeedDifference"] = (
    df["WindSpeed3pm"] - df["WindSpeed9am"]
)

df["AvgTemperature"] = (
    df["Temp9am"] + df["Temp3pm"]
) / 2

logging.info("Feature Engineering Completed")

# =========================================================
# TARGET ENCODING
# =========================================================

df["RainTomorrow"] = df["RainTomorrow"].map({
    "No": 0,
    "Yes": 1
})

# =========================================================
# FEATURE & TARGET SPLIT
# =========================================================

X = df.drop("RainTomorrow", axis=1)

y = df["RainTomorrow"]

# =========================================================
# IDENTIFY FEATURE TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# =========================================================
# NUMERICAL PIPELINE
# =========================================================

numeric_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler())

])

# =========================================================
# CATEGORICAL PIPELINE
# =========================================================

categorical_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("encoder", OneHotEncoder(handle_unknown="ignore"))

])

# =========================================================
# PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(transformers=[

    ("num", numeric_transformer, numeric_features),

    ("cat", categorical_transformer, categorical_features)

])

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

# =========================================================
# APPLY PREPROCESSING
# =========================================================

print("\n⚙️ Applying Preprocessing Pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# =========================================================
# CLASS IMBALANCE CHECK
# =========================================================

plt.figure(figsize=(6,5))

sns.countplot(x=y_train)

plt.title("Class Distribution Before SMOTE")

plt.savefig("outputs/before_smote.png")

plt.close()

print("\nBefore SMOTE:")
print(y_train.value_counts())

# =========================================================
# APPLY SMOTE
# =========================================================

print("\n⚙️ Applying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(

    X_train_processed,
    y_train
)

# =========================================================
# AFTER SMOTE VISUALIZATION
# =========================================================

plt.figure(figsize=(6,5))

sns.countplot(x=y_train_smote)

plt.title("Class Distribution After SMOTE")

plt.savefig("outputs/after_smote.png")

plt.close()

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

logging.info("SMOTE Applied Successfully")

# =========================================================
# STRATIFIED K-FOLD
# =========================================================

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)

# =========================================================
# MODEL DEFINITIONS
# =========================================================

models = {

    "RandomForest": RandomForestClassifier(
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        verbose=0,
        random_state=42
    )
}

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

print("\n🔍 Hyperparameter Optimization Started...")

xgb_params = {

    "n_estimators": [100, 200],

    "max_depth": [5, 7, 10],

    "learning_rate": [0.03, 0.05, 0.1],

    "subsample": [0.8, 1.0]
}

random_search = RandomizedSearchCV(

    estimator=XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),

    param_distributions=xgb_params,

    n_iter=5,

    scoring='f1',

    cv=3,

    verbose=1,

    random_state=42,

    n_jobs=-1
)

random_search.fit(

    X_train_smote,
    y_train_smote
)

best_xgb = random_search.best_estimator_

print("\n✅ Best XGBoost Parameters:")
print(random_search.best_params_)

models["Optimized_XGBoost"] = best_xgb

# =========================================================
# MODEL TRAINING & EVALUATION
# =========================================================

results = {}

best_model = None
best_f1 = 0

for name, model in models.items():

    print(f"\n==============================")
    print(f"🚀 Training: {name}")
    print(f"==============================")

    model.fit(

        X_train_smote,
        y_train_smote
    )

    y_pred = model.predict(X_test_processed)

    # =====================================================
    # PROBABILITY
    # =====================================================

    y_prob = model.predict_proba(
        X_test_processed
    )[:,1]

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cv_score = cross_val_score(

        model,

        X_train_smote,
        y_train_smote,

        cv=skf,

        scoring='f1'
    ).mean()

    results[name] = {

        "Accuracy": round(accuracy,4),

        "F1 Score": round(f1,4),

        "ROC-AUC": round(roc_auc,4),

        "CV Score": round(cv_score,4)
    }

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"CV Score : {cv_score:.4f}")

    print("\nClassification Report:\n")

    print(classification_report(
        y_test,
        y_pred
    ))

    logging.info(
        f"{name} -> Accuracy:{accuracy}, F1:{f1}"
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name

        best_predictions = y_pred

# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    "models/best_weather_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\n✅ Best Model Saved Successfully")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(best_model, "feature_importances_"):

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(

        by="Importance",

        ascending=False
    )

    print("\nTop Important Features:")

    print(importance_df.head(20))

    plt.figure(figsize=(12,8))

    sns.barplot(

        data=importance_df.head(20),

        x="Importance",

        y="Feature"
    )

    plt.title("Top 20 Important Features")

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"
)

plt.title(f"{best_model_name} Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# =========================================================
# ROC CURVE
# =========================================================

RocCurveDisplay.from_estimator(

    best_model,

    X_test_processed,

    y_test
)

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

print("\n⚙️ Generating SHAP Explainability...")

explainer = shap.Explainer(best_model)

shap_values = explainer(
    X_test_processed[:500]
)

shap.summary_plot(

    shap_values,

    X_test_processed[:500],

    show=False
)

plt.savefig(
    "outputs/shap_summary.png"
)

plt.close()

logging.info("SHAP Explainability Generated")

# =========================================================
# SAVE RESULTS CSV
# =========================================================

results_df = pd.DataFrame(results).T

results_df.to_csv(
    "outputs/model_results.csv"
)

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n================================================")
print("🏆 FINAL MODEL COMPARISON")
print("================================================")

print(results_df)

print(f"\n🏆 BEST MODEL: {best_model_name}")

print(f"🏆 BEST F1 SCORE: {best_f1:.4f}")

logging.info("Training Completed Successfully")

# =========================================================
# GENERATED FILES
# =========================================================

print("\nGenerated Files:")

print("""

📊 VISUALIZATIONS
✔ before_smote.png
✔ after_smote.png
✔ feature_importance.png
✔ confusion_matrix.png
✔ roc_curve.png
✔ shap_summary.png

📁 MODELS
✔ best_weather_model.pkl
✔ preprocessor.pkl

📄 REPORTS
✔ model_results.csv
✔ project.log

""")

print("\n✅ NEXT-GEN INDUSTRY WEATHER AI SYSTEM COMPLETED")