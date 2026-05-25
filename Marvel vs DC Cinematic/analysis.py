import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    print("=== Phase 1: Loading Data ===")
    df = pd.read_csv(file_path)
    print(f"Loaded raw dataset with shape: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    print("\nFirst 3 rows of raw data:")
    print(df.head(3))

    print("\n=== Phase 2: Data Cleaning ===")
    
    # 1. Remove duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_len - len(df)} duplicate row(s). New shape: {df.shape}")

    # 2. Standardize Franchise name casing
    df['Franchise'] = df['Franchise'].str.strip().str.capitalize()
    # E.g. 'marvel' -> 'Marvel', 'DC' -> 'Dc' (Wait, let's map 'Dc' to 'DC' and 'Marvel' to 'Marvel')
    df['Franchise'] = df['Franchise'].replace({'Dc': 'DC'})
    print(f"Standardized franchise categories: {df['Franchise'].unique()}")

    # 3. Clean Budget_Million: strip '$' and 'M', then convert to numeric
    # e.g., "$140M" -> 140.0, "$220" -> 220.0
    df['Budget_Million'] = (
        df['Budget_Million']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace('M', '', regex=False)
        .str.strip()
    )
    df['Budget_Million'] = pd.to_numeric(df['Budget_Million'], errors='coerce')

    # 4. Clean Rotten_Tomatoes_Pct: strip '%' and convert to numeric
    # e.g. "94%" -> 94
    df['Rotten_Tomatoes_Pct'] = (
        df['Rotten_Tomatoes_Pct']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.strip()
    )
    df['Rotten_Tomatoes_Pct'] = pd.to_numeric(df['Rotten_Tomatoes_Pct'], errors='coerce')

    # 5. Ensure Box_Office_Million is numeric
    df['Box_Office_Million'] = pd.to_numeric(df['Box_Office_Million'], errors='coerce')

    print("\nHandling missing values (if any)...")
    print(df.isnull().sum())
    # Fill any null values using median for numerical columns or drop them
    df['Budget_Million'] = df['Budget_Million'].fillna(df['Budget_Million'].median())
    df['Box_Office_Million'] = df['Box_Office_Million'].fillna(df['Box_Office_Million'].median())
    df['Rotten_Tomatoes_Pct'] = df['Rotten_Tomatoes_Pct'].fillna(df['Rotten_Tomatoes_Pct'].median())

    print("\n=== Phase 3: Feature Engineering ===")
    
    # Calculate Profit = Box Office Gross - Budget (in Millions)
    df['Profit_Million'] = df['Box_Office_Million'] - df['Budget_Million']
    
    # Calculate ROI % = (Profit / Budget) * 100
    df['ROI_Pct'] = (df['Profit_Million'] / df['Budget_Million']) * 100

    print("Added columns: 'Profit_Million' and 'ROI_Pct'")
    return df

def perform_exploratory_analysis(df):
    print("\n=== Phase 4: Exploratory Data Analysis (EDA) ===")
    
    # 1. Summary statistics per franchise
    franchise_summary = df.groupby('Franchise').agg(
        Movie_Count=('Movie', 'count'),
        Avg_Budget_M=('Budget_Million', 'mean'),
        Avg_BoxOffice_M=('Box_Office_Million', 'mean'),
        Avg_Profit_M=('Profit_Million', 'mean'),
        Avg_ROI_Pct=('ROI_Pct', 'mean'),
        Median_ROI_Pct=('ROI_Pct', 'median'),
        Avg_IMDb=('IMDb_Rating', 'mean'),
        Avg_RottenTomatoes=('Rotten_Tomatoes_Pct', 'mean')
    ).round(2)
    
    print("\nFranchise Performance Summary Table:")
    print(franchise_summary)
    
    # 2. Find top 3 highest ROI movies overall
    print("\nTop 3 Movies by Return on Investment (ROI %):")
    top_roi = df.sort_values(by='ROI_Pct', ascending=False).head(3)
    print(top_roi[['Movie', 'Franchise', 'Release_Year', 'Budget_Million', 'Box_Office_Million', 'ROI_Pct']])

    # 3. Find top 3 highest grossing movies overall
    print("\nTop 3 Highest-Grossing Movies (Global Box Office):")
    top_gross = df.sort_values(by='Box_Office_Million', ascending=False).head(3)
    print(top_gross[['Movie', 'Franchise', 'Release_Year', 'Budget_Million', 'Box_Office_Million', 'ROI_Pct']])

    # 4. Find bottom 3 lowest performing movies by ROI
    print("\nLowest 3 Movies by Return on Investment (ROI %):")
    bottom_roi = df.sort_values(by='ROI_Pct', ascending=True).head(3)
    print(bottom_roi[['Movie', 'Franchise', 'Release_Year', 'Budget_Million', 'Box_Office_Million', 'ROI_Pct']])

    return franchise_summary

def generate_and_save_visualizations(df, output_dir='plots'):
    print("\n=== Phase 5: Generating Visualizations ===")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Set style parameters for high quality, modern design
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    
    # Palette definition matching premium aesthetic: Marvel (Red/Crimson), DC (Deep Blue/Teal)
    palette = {'Marvel': '#E23636', 'DC': '#004B87'}

    # 1. Budget vs Box Office Scatter Plot with ROI representation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Budget_Million', 
        y='Box_Office_Million', 
        hue='Franchise', 
        palette=palette,
        size='ROI_Pct',
        sizes=(40, 400),
        alpha=0.8,
        edgecolor='black'
    )
    plt.title('Marvel vs DC: Budget vs Box Office (Size = ROI %)', pad=20)
    plt.xlabel('Budget (Millions USD)')
    plt.ylabel('Box Office Gross (Millions USD)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/budget_vs_boxoffice.png', dpi=300)
    plt.close()
    print("- Saved: budget_vs_boxoffice.png")

    # 2. Average Ratings Comparison: Subplots (IMDb & Rotten Tomatoes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: IMDb Average Rating
    sns.barplot(
        data=df,
        x='Franchise',
        y='IMDb_Rating',
        palette=palette,
        errorbar=None,
        ax=axes[0],
        edgecolor='black',
        width=0.6
    )
    axes[0].set_title('Average IMDb Rating')
    axes[0].set_ylim(0, 10)
    axes[0].set_ylabel('IMDb Rating (out of 10)')
    axes[0].set_xlabel('')
    
    # Add values on top of bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    # Right: Rotten Tomatoes Score
    sns.barplot(
        data=df,
        x='Franchise',
        y='Rotten_Tomatoes_Pct',
        palette=palette,
        errorbar=None,
        ax=axes[1],
        edgecolor='black',
        width=0.6
    )
    axes[1].set_title('Average Rotten Tomatoes Score')
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Rotten Tomatoes Score (%)')
    axes[1].set_xlabel('')
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

    plt.suptitle('Marvel vs DC: Audience and Critical Reception Comparison', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ratings_comparison.png', dpi=300)
    plt.close()
    print("- Saved: ratings_comparison.png")

    # 3. ROI Distribution Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='Franchise',
        y='ROI_Pct',
        palette=palette,
        width=0.5,
        linewidth=2,
        fliersize=5
    )
    # Add a horizontal line at 0 ROI (breakeven point)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, label='Breakeven (0% ROI)')
    plt.title('Return on Investment (ROI %) Distribution', pad=15)
    plt.xlabel('Franchise')
    plt.ylabel('ROI %')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roi_distribution.png', dpi=300)
    plt.close()
    print("- Saved: roi_distribution.png")

    # 4. Box Office Trend Over Time
    plt.figure(figsize=(10, 6))
    # Group by franchise and year, computing average box office
    trends = df.groupby(['Release_Year', 'Franchise'])['Box_Office_Million'].mean().reset_index()
    sns.lineplot(
        data=trends,
        x='Release_Year',
        y='Box_Office_Million',
        hue='Franchise',
        palette=palette,
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    plt.title('Average Box Office Revenue Trends (2000 - 2024)', pad=20)
    plt.xlabel('Release Year')
    plt.ylabel('Average Box Office (Millions USD)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxoffice_trends.png', dpi=300)
    plt.close()
    print("- Saved: boxoffice_trends.png")

def main():
    raw_data_file = 'marvel_dc_movies.csv'
    cleaned_data_file = 'marvel_dc_movies_cleaned.csv'
    
    # Complete workflow
    cleaned_df = load_and_clean_data(raw_data_file)
    
    # Save cleaned data
    cleaned_df.to_csv(cleaned_data_file, index=False)
    print(f"\nSaved cleaned dataset to: {cleaned_data_file}")
    
    perform_exploratory_analysis(cleaned_df)
    generate_and_save_visualizations(cleaned_df)
    
    print("\n=== Data Analysis Completed Successfully! ===")

if __name__ == "__main__":
    main()
