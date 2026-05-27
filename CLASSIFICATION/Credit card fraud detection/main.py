import os
import warnings
import logging
import joblib

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_DIRS = [
    'saved_models',
    'saved_graphs',
    'results'
]

for directory in OUTPUT_DIRS:
    os.makedirs(directory, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_PATH = '/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv'

logging.info('Loading dataset...')

try:
    df = pd.read_csv(DATA_PATH)
    logging.info('Dataset loaded successfully')
except Exception as e:
    logging.error(f'Error loading dataset: {e}')
    raise

# ==========================================================
# DATA VALIDATION
# ==========================================================

logging.info(f'Dataset Shape: {df.shape}')

print('\n===================================')
print('DATASET INFORMATION')
print('===================================')

print('\nDataset Shape:')
print(df.shape)

print('\nMissing Values:')
print(df.isnull().sum())

print('\nDuplicate Rows:', df.duplicated().sum())

print('\nClass Distribution:')
print(df['Class'].value_counts())

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

initial_rows = df.shape[0]

df = df.drop_duplicates()

logging.info(f'Removed {initial_rows - df.shape[0]} duplicate rows')

# ==========================================================
# FEATURE / TARGET SPLIT
# ==========================================================

X = df.drop('Class', axis=1)
y = df['Class']

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train[['Amount', 'Time']] = scaler.fit_transform(
    X_train[['Amount', 'Time']]
)

X_test[['Amount', 'Time']] = scaler.transform(
    X_test[['Amount', 'Time']]
)

# SAVE SCALER

joblib.dump(
    scaler,
    'saved_models/scaler.pkl'
)

logging.info('Scaler saved successfully')

# ==========================================================
# BEFORE SMOTE
# ==========================================================

print('\n===================================')
print('BEFORE SMOTE')
print('===================================')
print(y_train.value_counts())

# ==========================================================
# SMOTE BALANCING
# ==========================================================

smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print('\n===================================')
print('AFTER SMOTE')
print('===================================')
print(y_train_smote.value_counts())

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

models = {

    'Random_Forest': RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),

    'XGBoost': XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
}

# ==========================================================
# RESULTS STORAGE
# ==========================================================

results = []

# ==========================================================
# ROC CURVE FIGURE
# ==========================================================

plt.figure(figsize=(8, 6))

# ==========================================================
# CROSS VALIDATION STRATEGY
# ==========================================================

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================================
# MODEL TRAINING LOOP
# ==========================================================

for model_name, model in models.items():

    print('\n===================================')
    print(f'TRAINING MODEL: {model_name}')
    print('===================================')

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    model.fit(X_train_smote, y_train_smote)

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_path = f'saved_models/{model_name}.pkl'

    joblib.dump(model, model_path)

    logging.info(f'{model_name} saved successfully')

    # ======================================================
    # PREDICTIONS
    # ======================================================

    y_prob = model.predict_proba(X_test)[:, 1]

    # INDUSTRIAL THRESHOLD

    threshold = 0.45

    y_pred = (y_prob >= threshold).astype(int)

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    cv_scores = cross_val_score(
        model,
        X_train_smote,
        y_train_smote,
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1
    )

    cv_mean = np.mean(cv_scores)

    # ======================================================
    # STORE RESULTS
    # ======================================================

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'CV F1 Mean': cv_mean
    })

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print(f'\nAccuracy: {accuracy:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1 Score: {f1:.6f}')
    print(f'ROC AUC: {roc_auc:.6f}')
    print(f'Cross Validation F1: {cv_mean:.6f}')

    print('\nClassification Report:\n')

    print(classification_report(y_test, y_pred))

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()

    plt.savefig(
        f'saved_graphs/{model_name}_confusion_matrix.png',
        dpi=300
    )

    plt.close()

    # ======================================================
    # ROC CURVE
    # ======================================================

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{model_name} (AUC={roc_auc:.4f})'
    )

# ==========================================================
# FINAL ROC CURVE
# ==========================================================

plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    'saved_graphs/roc_curve_comparison.png',
    dpi=300
)

plt.close()

# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='F1 Score',
    ascending=False
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    'results/model_comparison_results.csv',
    index=False
)

print('\n===================================')
print('MODEL COMPARISON RESULTS')
print('===================================')

print(results_df)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

rf_model = models['Random_Forest']

importance = rf_model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

# ==========================================================
# FEATURE IMPORTANCE VISUALIZATION
# ==========================================================

plt.figure(figsize=(12, 8))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df.head(15),
    palette='magma'
)

plt.title('Top 15 Important Features')

plt.tight_layout()

plt.savefig(
    'saved_graphs/feature_importance.png',
    dpi=300
)

plt.close()

# ==========================================================
# LEARNING CURVE
# ==========================================================

train_sizes, train_scores, validation_scores = learning_curve(

    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    X_train_smote,
    y_train_smote,

    cv=cv_strategy,

    scoring='f1',

    train_sizes=np.linspace(0.1, 1.0, 5),

    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)

plt.figure(figsize=(8, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker='o',
    label='Training Score'
)

plt.plot(
    train_sizes,
    validation_mean,
    marker='o',
    label='Validation Score'
)

plt.xlabel('Training Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve')

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    'saved_graphs/learning_curve.png',
    dpi=300
)

plt.close()

# ==========================================================
# FINAL PROJECT SUMMARY
# ==========================================================

print('\n===================================')
print('PROJECT COMPLETED SUCCESSFULLY')
print('===================================')

print('\nSaved Outputs:')
print('\n1. Models -> saved_models/')
print('2. Graphs -> saved_graphs/')
print('3. Results -> results/')

print('\nBEST MODELS USED:')
print('\n1. Random Forest')
print('2. XGBoost')