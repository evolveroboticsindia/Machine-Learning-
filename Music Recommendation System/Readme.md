#  Music Recommendation System

A Flask-based music recommendation web app that uses audio feature analysis and machine learning clustering to suggest similar songs. Two model variants are included — one using **KMeans + PCA** and another using the more advanced **UMAP + DBSCAN** pipeline.



##  Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the App

### Option A — Standard model (KMeans + PCA)

```bash
python app.py
```

### Option B — Advanced model (UMAP + DBSCAN)

```bash
python model.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

##  How It Works

1. **Each song is measured** — The system reads audio properties like tempo, pitch, brightness, and texture (MFCCs) from every song in the dataset.
2. **The numbers are balanced** — All measurements are scaled so they're on equal footing before comparison.
3. **The data is simplified** — 25 audio features are reduced to 2 dimensions so the model can find patterns more easily.
4. **Songs are grouped** — Similar-sounding songs get placed into the same cluster automatically, with no manual tagging needed.
5. **You get recommendations** — When you type a song name, the app finds which group it belongs to and suggests other songs from that same group.

---

##  Dataset

**File:** `songs_clustered.csv`

**Columns:**

| Column | Description |
|---|---|
| `song_name` | Song filename / title |
| `mfcc_1` – `mfcc_20` | Mel-frequency cepstral coefficients |
| `tempo` | Beats per minute |
| `spectral_rolloff` | Frequency below which 85% of energy falls |
| `chroma` | Harmonic / tonal content |
| `spectral_centroid` | Brightness of the sound |
| `spectral_flatness` | Noisiness vs tonality |
| `cluster` | Pre-assigned cluster label |

The dataset contains **~78 songs** with pre-computed audio features and cluster assignments.

---

##  Generated Visualizations

All charts are saved to the `OUTPUTS/` folder automatically on startup:

- **Cluster Distribution** — bar chart of songs per cluster
- **PCA / UMAP Scatter** — 2D cluster visualization
- **Tempo vs Chroma** — scatter colored by cluster
- **Feature Correlation Heatmap** — correlation between all audio features
- **Tempo Distribution** — histogram with KDE
- **MFCC Audio Fingerprint** — MFCC line plot for the first song
- **Feature Pairplot** — pairwise plots of key features
- **Silhouette Analysis** *(app.py)* — optimal cluster count sweep

---

## 🛠 Requirements

- Python 3.9+
- Flask
- scikit-learn
- umap-learn
- pandas, numpy, scipy
- matplotlib, seaborn
- See `requirements.txt` for full pinned versions


