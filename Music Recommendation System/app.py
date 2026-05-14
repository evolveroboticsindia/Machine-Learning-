import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

df = pd.read_csv("songs_clustered.csv")
df["song_lower"] = df["song_name"].astype(str).str.lower()

print("Dataset loaded!")

N_MFCC = 13

features_cols = (
    [f"mfcc_{i}" for i in range(1, N_MFCC + 1)] +
    [
        "tempo",
        "spectral_rolloff",
        "chroma",
        "spectral_centroid",
        "spectral_flatness"
    ]
)

existing_features = [c for c in features_cols if c in df.columns]

df[existing_features] = df[existing_features].fillna(
    df[existing_features].median()
)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[existing_features])

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=20
)

df["cluster"] = kmeans.fit_predict(X_scaled)

def extract_features(file_path):

    y, sr = librosa.load(file_path, duration=10, mono=True)

    # ---------------- MFCC ----------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ---------------- TEMPO ----------------
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    tempo = float(np.atleast_1d(tempo)[0])

    # ---------------- SPECTRAL FEATURES ----------------
    spectral_rolloff = float(np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    ))

    chroma = float(np.mean(
        librosa.feature.chroma_stft(y=y, sr=sr)
    ))

    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    ))

    spectral_flatness = float(np.mean(
        librosa.feature.spectral_flatness(y=y)
    ))

    feature_vector = np.hstack([
        mfcc_mean,
        tempo,
        spectral_rolloff,
        chroma,
        spectral_centroid,
        spectral_flatness
    ])

    return np.array(feature_vector, dtype=np.float64).reshape(1, -1)


def recommend_from_dataset(song_name, n=8):

    song_name = song_name.lower().strip()

    if song_name not in df["song_lower"].values:
        return []

    cluster = df.loc[
        df["song_lower"] == song_name,
        "cluster"
    ].values[0]

    recs = df[
        (df["cluster"] == cluster) &
        (df["song_lower"] != song_name)
    ][["song_name"]].head(n)

    return [{"name": r} for r in recs["song_name"].values]


def recommend_uploaded(file_path, n=8):

    feat = extract_features(file_path)

    feat_scaled = scaler.transform(feat)

    cluster = kmeans.predict(feat_scaled)[0]

    recs = df[df["cluster"] == cluster][["song_name"]].head(n)

    return [{"name": r} for r in recs["song_name"].values]

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    message = ""

    if request.method == "POST":

        song = request.form.get("song")
        file = request.files.get("file")

        if file and file.filename != "":

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(file_path)

            recommendations = recommend_uploaded(file_path)

            message = f"Recommendations for uploaded song: {file.filename}"

        elif song:

            song = song.lower().strip()

            if song in df["song_lower"].values:

                recommendations = recommend_from_dataset(song)

                message = f"Recommendations for: {song}"

            else:

                message = "Song not found in dataset. Try uploading audio."

    return render_template(
        "index.html",
        recommendations=recommendations,
        message=message
    )

if __name__ == "__main__":
    print("App running at http://127.0.0.1:5000")
    app.run(debug=True)