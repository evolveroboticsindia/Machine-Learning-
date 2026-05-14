# ============================================
# 🚀 CUSTOMER INTELLIGENCE SYSTEM
# ============================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from scipy.stats import ks_2samp

sns.set_style("whitegrid")

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv(
    "/kaggle/input/datasets/abhishekrp1517/online-retail-transactions-dataset/Online Retail.csv",
    encoding='ISO-8859-1'
)

print("\n===== DATA OVERVIEW =====")
print(df.info())
print(df.head())

# ============================================
# 📊 EDA VISUALS
# ============================================

plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

plt.figure(figsize=(8,4))
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title("Top Customer Countries")
plt.show()

# ============================================
# 2. CLEANING
# ============================================

df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("\nCleaned Shape:", df.shape)

# ============================================
# 📊 TOP PRODUCTS
# ============================================

top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,4))
top_products.plot(kind='bar')
plt.title("Top 10 Products Sold")
plt.show()

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

agg_df = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'InvoiceNo': 'nunique',
    'StockCode': 'nunique',
    'InvoiceDate': ['min', 'max']
}).reset_index()

agg_df.columns = [
    'CustomerID','Total Spend','Items Purchased',
    'Total Orders','Unique Products',
    'First Purchase','Last Purchase Date'
]

today = df['InvoiceDate'].max()

agg_df['Recency'] = (today - agg_df['Last Purchase Date']).dt.days
agg_df['Lifetime'] = (agg_df['Last Purchase Date'] - agg_df['First Purchase']).dt.days

agg_df['Avg Spend Per Item'] = agg_df['Total Spend']/(agg_df['Items Purchased']+1)
agg_df['Purchase Frequency'] = agg_df['Total Orders']/(agg_df['Lifetime']+1)
agg_df['Spending Velocity'] = agg_df['Total Spend']/(agg_df['Lifetime']+1)

customer_df = agg_df.copy()

# ============================================
# 🔥 RFM SCORING
# ============================================

customer_df['R'] = pd.qcut(
    customer_df['Recency'],
    4,
    labels=[4, 3, 2, 1],
    duplicates='drop'
).astype(int)

customer_df['F'] = pd.qcut(
    customer_df['Total Orders'].rank(method='first'),
    q=4,
    labels=[1, 2, 3, 4],
    duplicates='drop'
).astype(int)

customer_df['M'] = pd.qcut(
    customer_df['Total Spend'],
    q=4,
    labels=[1, 2, 3, 4],
    duplicates='drop'
).astype(int)

# ============================================
# 📊 RFM VISUALS
# ============================================

sns.countplot(x=customer_df['R'])
plt.title("Recency Score Distribution")
plt.show()

# ============================================
# 4. CHURN LABEL
# ============================================

customer_df['Churn'] = (customer_df['Recency'] > 90).astype(int)

sns.countplot(x=customer_df['Churn'])
plt.title("Churn Distribution")
plt.show()

print("\nChurn Rate:", customer_df['Churn'].mean())

# ============================================
# 5. FEATURES
# ============================================

features = [
    'Total Spend','Items Purchased','Total Orders','Unique Products',
    'Avg Spend Per Item','Purchase Frequency','Spending Velocity',
    'R','F','M'
]

X = customer_df[features]
y = customer_df['Churn']

# ============================================
# 6. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

sns.countplot(x=y_train)
plt.title("Class Distribution (Before SMOTE)")
plt.show()

# ============================================
# 🔥 AFTER SMOTE VISUAL (ADDED)
# ============================================

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

sns.countplot(x=y_res)
plt.title("Class Distribution (After SMOTE)")
plt.show()

# ============================================
# 7. MODEL PIPELINE
# ============================================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(random_state=42, verbose=-1))
])

param_grid = {
    'model__n_estimators': [200, 300, 500],
    'model__max_depth': [4, 6, 8],
    'model__learning_rate': [0.01, 0.05, 0.1]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=8,
    scoring='roc_auc',
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\nBest Params:", search.best_params_)

# ============================================
# 8. MODEL COMPARISON + GRAPH (ADDED)
# ============================================

models = {
    "LightGBM": LGBMClassifier(verbose=-1),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0)
}

print("\n===== MODEL COMPARISON =====")

model_names = []
model_scores = []

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test[features])[:,1]

    score = roc_auc_score(y_test, probs)

    model_names.append(name)
    model_scores.append(score)

    print(f"{name}: ROC-AUC = {score:.4f}")

# 🔥 MODEL COMPARISON BAR CHART (ADDED)
plt.figure(figsize=(6,4))
sns.barplot(x=model_names, y=model_scores)
plt.title("Model Comparison (ROC-AUC)")
plt.ylabel("ROC-AUC Score")
plt.show()

# ============================================
# 9. EVALUATION
# ============================================

y_probs = best_model.predict_proba(X_test[features])[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.show()

print("\nROC-AUC:", roc_auc_score(y_test, y_probs))

# ============================================
# 📉 CALIBRATION
# ============================================

prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.title("Calibration Curve")
plt.show()

# ============================================
# 💰 ROI OPTIMIZATION
# ============================================

best_roi, best_threshold = -np.inf, 0.5

for t in np.linspace(0.1,0.9,50):
    preds = (y_probs > t).astype(int)
    at_risk = X_test[preds==1]

    cost = len(at_risk)*50
    saved = at_risk['Total Spend'].sum()*0.3

    roi = (saved-cost)/(cost+1e-6)

    if roi > best_roi:
        best_roi = roi
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best ROI:", best_roi)

# ============================================
# 🔍 DRIFT DETECTION
# ============================================

print("\n===== FEATURE DRIFT =====")

for col in features:
    stat, p = ks_2samp(X_train[col], X_test[col])
    print(col, ":", "Drift ⚠️" if p<0.05 else "Stable ✅")

# ============================================
# 🔍 SHAP
# ============================================

model = best_model.named_steps['model']
explainer = shap.TreeExplainer(model)

sample = X_test.sample(300)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample)

# ============================================
# 🔥 CLUSTERING
# ============================================

cluster_data = customer_df[['Recency','Total Orders','Total Spend']]
scaled = StandardScaler().fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(scaled)

sns.countplot(x='Cluster', data=customer_df)
plt.title("Customer Segments")
plt.show()

print(customer_df.groupby('Cluster')[['Total Spend','Total Orders']].mean())

# ============================================
# 💰 BUSINESS SUMMARY
# ============================================

print("\n===== BUSINESS SUMMARY =====")
print("Customers:", len(customer_df))
print("Churn Rate:", customer_df['Churn'].mean())
print("Revenue Loss:", customer_df[customer_df['Churn']==1]['Total Spend'].sum())
print("High Risk Customers:", sum(y_probs > best_threshold))

# ============================================
# 💾 SAVE MODEL
# ============================================

joblib.dump({
    "model": best_model,
    "threshold": best_threshold
}, "customer_intelligence_model.pkl")

