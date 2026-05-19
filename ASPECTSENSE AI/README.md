# AspectSense AI

A FastAPI-based service for analyzing customer reviews using NLP and machine learning. It performs sentiment analysis, emotion detection, aspect extraction, fake review detection, and generates actionable insights.

---

## Features

- **Sentiment Analysis** — Classifies review sentiment (positive/negative/neutral) with a confidence score
- **Emotion Detection** — Identifies the dominant emotion expressed in a review
- **Aspect Extraction** — Pulls out key topics and aspects mentioned in the review
- **Fake Review Detection** — Flags potentially inauthentic reviews with a reason
- **Insights Generation** — Produces a human-readable summary of the full analysis

---

## Project Structure

```
.
├── src/
│   └── api/
│       └── api.py          # FastAPI route definitions
├── app.py                  # Dev server entrypoint (uvicorn with --reload)
├── models.py               # Pydantic request/response models
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container build instructions
```

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (FastAPI, Transformers, PyTorch, spaCy, etc.)

---

## Getting Started

### Local Setup

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Start the server
uvicorn src.api.api:app
```

Once the server starts, verify it's running by visiting `http://127.0.0.1:8000` — you should see a JSON response confirming the API is live.

## Using the Web Interface (Swagger UI)

1. Open `http://127.0.0.1:8000/docs` in your browser
2. Click on **POST /analyze** to expand it
3. Click **Try it out**
4. In the request body, replace the default value with your review text:
   ```json
   {
     "text": "The product quality was excellent but delivery took forever."
   }
   ```
5. Click **Execute**
6. Scroll down to see the full analysis under **Response body**

---

## API Reference

### `POST /analyze`

Analyzes a review and returns a full breakdown.

**Request body:**
```json
{
  "text": "The product quality was excellent but delivery took forever."
}
```

**Response:**
```json
{
  "review": "The product quality was excellent but delivery took forever.",
  "sentiment": { "label": "positive", "score": 0.87 },
  "aspects": ["product quality", "delivery"],
  "emotion": { "emotion": "satisfaction", "score": 0.72 },
  "fake_review": { "fake_review": false, "reason": "Natural language patterns detected" },
  "insights": "Customers are happy with product quality but concerned about shipping times."
}
```

---

## Models Used

| Component | Library |
|---|---|
| Sentiment & Emotion | Hugging Face Transformers |
| Aspect Extraction | spaCy (`en_core_web_sm`) |
| Fake Review Detection | scikit-learn / XGBoost |
| Embeddings | sentence-transformers |
| Topic Modeling | BERTopic |

---

## License

MIT
