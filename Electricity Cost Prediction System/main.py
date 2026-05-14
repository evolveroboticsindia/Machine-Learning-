# ============================================
# ⚡ ELECTRICITY COST PREDICTION
# ============================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. LOAD DATA
# ============================================
path = "/kaggle/input/datasets/imtkaggleteam/household-power-consumption/household_power_consumption.csv"
df = pd.read_csv(path)

# ============================================
# 2. CLEAN COLUMN NAMES
# ============================================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ============================================
# 3. CLEAN DATA
# ============================================
df.replace("?", np.nan, inplace=True)

for col in df.columns:
    if col not in ['date', 'time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================
# 4. DATETIME ENGINEERING
# ============================================
df['datetime'] = pd.to_datetime(
    df['date'].astype(str) + " " + df['time'].astype(str),
    errors='coerce'
)

df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime')
df.set_index('datetime', inplace=True)

df.drop(['date', 'time'], axis=1, inplace=True)

# ============================================
# 5. MISSING VALUE HANDLING
# ============================================
df = df.interpolate(method='time')
df.fillna(df.median(), inplace=True)

# ============================================
# 6. TIME FEATURES
# ============================================
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday

# ============================================
# 7. TARGET CREATION
# ============================================
power_col = [col for col in df.columns if "active" in col][0]
df['electricity_cost'] = df[power_col] * 5

# ============================================
# 8. LAG FEATURES (VERY IMPORTANT FOR TOP 1%)
# ============================================
df['lag_1'] = df[power_col].shift(1)
df['lag_2'] = df[power_col].shift(2)

df['rolling_mean_3'] = df[power_col].rolling(3).mean()
df['rolling_std_3'] = df[power_col].rolling(3).std()

df = df.dropna()

# ============================================
# 9. FEATURES
# ============================================
features = [
    power_col,
    'global_reactive_power',
    'voltage',
    'global_intensity',
    'sub_metering_1',
    'sub_metering_2',
    'sub_metering_3',
    'hour', 'day', 'month', 'weekday',
    'lag_1', 'lag_2',
    'rolling_mean_3',
    'rolling_std_3'
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df['electricity_cost']

# ============================================
# 10. TIME SERIES SPLIT (NO LEAKAGE)
# ============================================
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================
# 11. PREPROCESSING PIPELINE
# ============================================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), features)
])

# ============================================
# 12. MODELS (TOP 1% STACK)
# ============================================
models = {
    "Ridge": Pipeline([
        ("prep", preprocess),
        ("model", Ridge())
    ]),

    "RandomForest": Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(random_state=42))
    ]),

    "GradientBoosting": Pipeline([
        ("prep", preprocess),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
}

# ============================================
# 13. HYPERPARAMETER TUNING (BEST MODEL ONLY)
# ============================================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 10]
}

grid = GridSearchCV(
    models["RandomForest"],
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
models["RandomForest"] = best_rf

print("\nBEST RF PARAMS:", grid.best_params_)

# ============================================
# 14. MODEL TRAINING + EVALUATION
# ============================================
results = {}

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = [mae, rmse, r2]

    print("\n", name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

for name, model in models.items():
    evaluate(name, model)

# ============================================
# 15. RESULTS TABLE
# ============================================
results_df = pd.DataFrame(results, index=["MAE", "RMSE", "R2"]).T
print("\nMODEL COMPARISON:\n", results_df)

# ============================================
# 16. BEST MODEL SELECTION
# ============================================
best_model_name = results_df.sort_values("R2", ascending=False).index[0]
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# ============================================
# 17. MODEL SAVE (KAGGLE LEVEL)
# ============================================
joblib.dump(best_model, "electricity_model_top1.pkl")

# ============================================
# 18. SHAP EXPLAINABILITY (TOP 1% FEATURE INSIGHT)
# ============================================
if "RandomForest" in best_model_name or "GradientBoosting" in best_model_name:

    explainer = shap.Explainer(best_model.named_steps["model"], X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)

# ============================================
# 19. FINAL PREDICTION
# ============================================
sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)

print("\nPREDICTED ELECTRICITY COST:", prediction[0])