# 🚀 Customer Intelligence System
## AI-Powered Customer Churn Prediction & Customer Analytics Platform
# 📌 Project Overview

The Customer Intelligence System is an end-to-end industry-grade Machine Learning project designed to analyze customer purchasing behavior, predict customer churn, segment customers, and optimize business retention strategies.

This project combines advanced data analytics, customer behavior modeling, explainable AI, and business intelligence techniques to create a real-world customer analytics solution similar to systems used in modern e-commerce, retail, fintech, and subscription-based companies.

The system performs:

- Customer churn prediction
- RFM customer analysis
- ROI optimization
- Customer segmentation
- Drift detection
- Explainable AI analysis
- Ensemble model comparison

The project demonstrates a complete machine learning workflow from raw transactional data to business-ready insights.

# 🎯 Business Problem

Customer retention is one of the most critical challenges in modern businesses.

Companies lose significant revenue when customers stop purchasing products or services. Identifying high-risk customers early allows businesses to:

- Reduce customer churn
- Improve retention campaigns
- Increase revenue
- Optimize marketing costs
- Improve customer satisfaction

This project helps businesses predict which customers are likely to churn using transactional and behavioral data.

# 🧠 Key Features

## ✅ Data Cleaning & Preprocessing
- Removed cancelled transactions
- Removed invalid quantities and prices
- Handled missing values
- Converted date features into datetime format


## ✅ Advanced Feature Engineering
Generated powerful business-centric customer features:

- Total Spend
- Purchase Frequency
- Spending Velocity
- Average Spend Per Item
- Customer Lifetime
- Unique Product Count
- Recency Analysis


## ✅ RFM Analysis
Implemented RFM (Recency, Frequency, Monetary) scoring for customer behavior analysis.

### RFM Metrics:
- Recency → How recently a customer purchased
- Frequency → How often a customer purchases
- Monetary → How much a customer spends

## ✅ Customer Churn Prediction
Built a machine learning pipeline to predict customer churn using:

- LightGBM
- XGBoost
- CatBoost

The system identifies high-risk customers likely to stop purchasing.


## ✅ Imbalanced Data Handling
Implemented SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance effectively.


## ✅ Hyperparameter Optimization
Used RandomizedSearchCV with Stratified K-Fold Cross Validation for robust model tuning.


## ✅ Explainable AI (XAI)
Implemented SHAP Explainability to understand feature importance and model decision-making.

This improves:
- Transparency
- Trustworthiness
- Business interpretability

## ✅ Drift Detection
Used KS-Test based feature drift detection to identify distribution changes between training and testing data.

This is an important concept in production-grade machine learning systems.


## ✅ Customer Segmentation
Applied KMeans Clustering to divide customers into different behavioral groups for:
- Marketing strategies
- Personalized recommendations
- Business targeting

## ✅ ROI Optimization
Implemented ROI-based threshold optimization instead of relying only on accuracy.

This helps businesses maximize retention profit and minimize campaign costs.


# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost

### Imbalanced Learning
- SMOTE

### Explainable AI
- SHAP

### Model Saving
- Joblib


# 📊 Machine Learning Workflow

```text
Raw Retail Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
RFM Analysis
       ↓
Train-Test Split
       ↓
SMOTE Balancing
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Evaluation
       ↓
SHAP Explainability
       ↓
Drift Detection
       ↓
Customer Segmentation
       ↓
Business Insights
```


# 📈 Models Used

| Model | Purpose |
|---|---|
| LightGBM | Fast Gradient Boosting |
| XGBoost | Advanced Ensemble Learning |
| CatBoost | High-performance Gradient Boosting |


# 📊 Evaluation Metrics

The project evaluates models using:

- ROC-AUC Score
- ROC Curve
- Calibration Curve
- Confusion Matrix
- Classification Report

# 🔍 Explainable AI

SHAP Summary Plots are used to explain:
- Which features influence churn
- Feature impact on predictions
- Business behavior interpretation


# 💼 Business Applications

This system can be used in:

- E-commerce Platforms
- Retail Analytics
- Subscription Businesses
- Banking & FinTech
- CRM Systems
- Customer Retention Teams


# 📦 Dataset

Dataset Used:
Online Retail Transaction Dataset

Contains:
- Customer purchase history
- Product information
- Invoice data
- Transaction timestamps
- Country information


# 🚀 Future Improvements

Potential future enhancements:

- Streamlit Dashboard
- Real-Time Prediction API
- FastAPI Deployment
- MLflow Experiment Tracking
- Docker Containerization
- Cloud Deployment (AWS/GCP/Azure)
- Real-Time Drift Monitoring



# 💾 Output

The system generates:

- Customer churn predictions
- ROC curves
- Calibration curves
- SHAP explainability plots
- Customer clusters
- Business analytics summary
- Saved ML model (.pkl)

---

# 🏆 Project Highlights

✅ End-to-End Machine Learning Pipeline  
✅ Industry-Level Customer Analytics  
✅ Advanced Feature Engineering  
✅ Explainable AI Integration  
✅ Business ROI Optimization  
✅ Drift Detection  
✅ Customer Segmentation  
✅ Ensemble Learning Models  
✅ Production-Ready Model Saving  


# ⭐ Final Note

This project demonstrates strong practical knowledge in:
- Data Science
- Machine Learning
- Business Analytics
- Explainable AI
- Customer Intelligence Systems

It is designed as a professional portfolio project suitable for showcasing advanced AI and analytics skills.