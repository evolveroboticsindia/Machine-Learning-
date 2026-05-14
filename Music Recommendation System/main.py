import os
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from flask import Flask, render_template, request

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import umap.umap_ as umap

warnings.filterwarnings("ignore")


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("OUTPUT FOLDER CREATED")
print(OUTPUT_DIR)
print("=" * 60)

DATASET_PATH = "songs_clustered.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")

print(df.columns.tolist())

df["song_lower"] = df["song_name"].astype(str).str.lower()

features_cols = [
    'mfcc_1',
    'mfcc_2',
    'mfcc_3',
    'mfcc_4',
    'mfcc_5',
    'tempo',
    'chroma',
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_flatness'
]


existing_features = [
    col for col in features_cols
    if col in df.columns
]

df[existing_features] = df[existing_features].fillna(
    df[existing_features].median()
)


X = df[existing_features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Data Standardized Successfully!")


z_scores = np.abs(stats.zscore(X_scaled))

filtered_entries = (z_scores < 3).all(axis=1)

X_scaled = X_scaled[filtered_entries]

df = df[filtered_entries]

print("Outliers Removed Successfully!")


umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

X_umap = umap_model.fit_transform(X_scaled)

print("UMAP Applied Successfully!")


dbscan = DBSCAN(
    eps=0.7,
    min_samples=8
)

labels = dbscan.fit_predict(X_umap)

df["cluster"] = labels

print("DBSCAN Clustering Completed!")


valid_mask = df["cluster"] != -1

df = df[valid_mask]

X_umap = X_umap[valid_mask]

labels = labels[valid_mask]

print(f"Remaining Songs After Noise Removal: {len(df)}")


score = silhouette_score(
    X_umap,
    labels
)

print("\n" + "=" * 60)

print(f"SILHOUETTE SCORE : {score:.4f}")

print("=" * 60)


output_csv = os.path.join(
    OUTPUT_DIR,
    "songs_clustered_output.csv"
)

df.to_csv(output_csv, index=False)

print(f"DATASET SAVED: {output_csv}")


plt.figure(figsize=(10, 5))

sns.countplot(x=df['cluster'])

plt.title("Cluster Distribution")

cluster_path = os.path.join(
    OUTPUT_DIR,
    "cluster_distribution.png"
)

plt.savefig(cluster_path)

plt.close()

print(f"IMAGE SAVED: {cluster_path}")

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=labels,
    cmap='viridis'
)

plt.title("UMAP + DBSCAN Song Clusters")

plt.xlabel("UMAP 1")

plt.ylabel("UMAP 2")

plt.colorbar(scatter)

umap_path = os.path.join(
    OUTPUT_DIR,
    "umap_dbscan_clusters.png"
)

plt.savefig(umap_path)

plt.close()

print(f"IMAGE SAVED: {umap_path}")

if 'tempo' in df.columns and 'chroma' in df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x='tempo',
        y='chroma',
        hue='cluster',
        palette='viridis'
    )

    plt.title("Tempo vs Chroma")

    tempo_path = os.path.join(
        OUTPUT_DIR,
        "tempo_vs_chroma.png"
    )

    plt.savefig(tempo_path)

    plt.close()

    print(f"IMAGE SAVED: {tempo_path}")

plt.figure(figsize=(12, 8))

corr = df[existing_features].corr()

sns.heatmap(
    corr,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "feature_correlation_heatmap.png"
)

plt.savefig(heatmap_path)

plt.close()

print(f"IMAGE SAVED: {heatmap_path}")


if 'tempo' in df.columns:

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["tempo"],
        bins=30,
        kde=True
    )

    plt.title("Tempo Distribution")

    hist_path = os.path.join(
        OUTPUT_DIR,
        "tempo_distribution.png"
    )

    plt.savefig(hist_path)

    plt.close()

    print(f"IMAGE SAVED: {hist_path}")


mfcc_columns = [
    col for col in df.columns
    if "mfcc_" in col
]

if len(mfcc_columns) > 0:

    sample_song = df.iloc[0]

    mfcc_values = sample_song[mfcc_columns].values

    plt.figure(figsize=(12, 5))

    plt.plot(mfcc_columns, mfcc_values)

    plt.title("MFCC Audio Fingerprint")

    plt.xticks(rotation=45)

    mfcc_path = os.path.join(
        OUTPUT_DIR,
        "mfcc_audio_fingerprint.png"
    )

    plt.savefig(mfcc_path)

    plt.close()

    print(f"IMAGE SAVED: {mfcc_path}")


pairplot_cols = []

for col in [
    "tempo",
    "chroma",
    "spectral_rolloff",
    "spectral_centroid",
    "cluster"
]:
    if col in df.columns:
        pairplot_cols.append(col)

if len(pairplot_cols) >= 3:

    pairplot = sns.pairplot(
        df[pairplot_cols],
        hue="cluster"
    )

    pairplot_path = os.path.join(
        OUTPUT_DIR,
        "feature_pairplot.png"
    )

    pairplot.savefig(pairplot_path)

    plt.close()

    print(f"IMAGE SAVED: {pairplot_path}")


def recommend(song_name, n_recommendations=10):

    song_name = song_name.lower().strip()

    if song_name not in df["song_lower"].values:

        return []

    cluster = df[
        df["song_lower"] == song_name
    ]["cluster"].values[0]

    rec_df = df[
        (df["cluster"] == cluster) &
        (df["song_lower"] != song_name)
    ][["song_name"]].head(n_recommendations)

    recommendations = []

    for _, row in rec_df.iterrows():

        recommendations.append({
            "name": row["song_name"]
        })

    return recommendations

@app.route('/', methods=['GET', 'POST'])

def home():

    recommendations = []

    message = ""

    if request.method == 'POST':

        song_name = request.form['song'].lower().strip()

        if song_name in df['song_lower'].values:

            recommendations = recommend(song_name)

            message = (
                f"Recommendations for: "
                f"{request.form['song'].strip()}"
            )

        else:

            message = "Song not found in dataset!"

    return render_template(
        'index.html',
        recommendations=recommendations,
        message=message
    )

if __name__ == '__main__':

    print("=" * 60)
    print("ALL OUTPUT FILES SAVED INSIDE OUTPUTS FOLDER")
    print("=" * 60)

    app.run(
        debug=True,
        use_reloader=False
    )