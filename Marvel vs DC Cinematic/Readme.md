# Marvel vs DC Cinematic Intelligence Dashboard

A premium industry-style data analytics dashboard built using Python, Streamlit, Plotly, Pandas, and Seaborn to compare the financial and audience performance of Marvel and DC cinematic universes.

This project performs:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Financial Analytics
* ROI Analysis
* Interactive Dashboard Visualization
* Business Intelligence Reporting


# Project Preview

## Features Included

* Interactive Streamlit Dashboard
* Premium Dark-Themed UI
* Financial KPI Tracking
* ROI Analytics
* Box Office Trend Analysis
* Audience & Critics Rating Comparison
* Movie Search & Filtering
* Downloadable Cleaned Dataset
* Automated Visualization Generation


# Technologies Used

| Technology | Purpose               |
| ---------- | --------------------- |
| Python     | Core Programming      |
| Pandas     | Data Processing       |
| NumPy      | Numerical Computation |
| Matplotlib | Static Visualizations |
| Seaborn    | Statistical Charts    |
| Plotly     | Interactive Charts    |
| Streamlit  | Dashboard Development |


# Dashboard Modules

## Main Dashboard

Displays:

* Total Box Office Revenue
* Average ROI
* IMDb Rating Comparison
* Interactive Scatter Plot

### Key Visualization

* Budget vs Box Office Performance
* Bubble Size = ROI Strength


## Financial Analytics

Displays:

* Budget Comparison
* Profit Comparison
* ROI Distribution
* Revenue Trends Over Time

### Business Insights

* Financial efficiency comparison
* Profitability analysis
* Franchise revenue evolution


## Reception Analysis

Displays:

* IMDb Distribution
* Rotten Tomatoes Distribution
* Top Rated Movies
* Highest ROI Movies
* Highest Grossing Movies

### Audience Intelligence

* Audience sentiment comparison
* Critical reception analytics


## 4. Dataset Explorer

Features:

* Interactive search
* Director filtering
* Download cleaned dataset
* Dynamic table exploration


# Data Cleaning Pipeline

The dataset undergoes several preprocessing steps:

## Cleaning Operations

* Duplicate row removal
* Franchise normalization
* Budget column cleaning
* Rotten Tomatoes score formatting
* Numeric conversion
* Missing value handling


# Feature Engineering

Two major business metrics are generated:

## Profit Calculation

Profit = BoxOffice - Budget

## ROI Calculation

ROI% = \frac{(Profit)}{Budget} \times 100

Generated Features:

* `Profit_Million`
* `ROI_Pct`


# Visualizations Generated

## Static Visualizations

Saved automatically inside `/plots`

### Included Charts

* Budget vs Box Office Scatter Plot
* Ratings Comparison
* ROI Distribution
* Box Office Trends


# Interactive Dashboard Features

## Filters Included

* Franchise Selection
* Release Year Range
* Minimum IMDb Rating
* Minimum Rotten Tomatoes Score


# Industry Relevance

This project simulates a real-world media analytics and business intelligence platform similar to tools used by:

* Netflix
* Disney
* Warner Bros. Discovery
* IMDb


# Business Use Cases

* Media Revenue Analysis
* Franchise Performance Tracking
* Investment Decision Analytics
* Audience Sentiment Monitoring
* Entertainment Market Intelligence


## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```


# Running the Project

## Run Analysis Pipeline

```bash
python analysis.py
```

This will:

* Clean dataset
* Generate engineered features
* Perform EDA
* Save visualizations

---

## Run Dashboard

```bash
streamlit run app.py
```


# Key KPIs Monitored

| KPI              | Description           |
| ---------------- | --------------------- |
| Total Box Office | Total global revenue  |
| ROI %            | Investment efficiency |
| Profit           | Revenue - Budget      |
| IMDb Rating      | Audience reception    |
| Rotten Tomatoes  | Critical reception    |


# Sample Insights

* Marvel demonstrates stronger financial consistency.
* DC exhibits higher revenue volatility.
* High-budget films do not always guarantee high ROI.
* Critically acclaimed films may not always achieve the highest box office performance.

# Future Improvements

## Planned Enhancements

* Machine Learning Revenue Prediction
* Real-Time Movie API Integration
* SQL Database Integration
* User Authentication
* Docker Deployment
* Cloud Hosting
* AI-Powered Recommendation System


