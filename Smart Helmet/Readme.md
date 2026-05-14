#  Smart Helmet Detection System

A real-time AI-powered helmet detection application built using **YOLOv8** and **Streamlit**. The system detects people in an image and identifies whether they are wearing a safety helmet or not.

The application highlights:
- ✅ **Helmet detected** → Green bounding box
- ❌ **No helmet detected** → Red bounding box with violation alert


## 📌 Use Cases

This project is useful for:
- Construction site monitoring
- Traffic safety enforcement
- Industrial workplace compliance
- Smart surveillance systems



## ✨ Features

- Detects persons using YOLOv8
- Detects helmets separately
- Matches helmets with detected persons
- Displays violations instantly
- User-friendly Streamlit web interface
- Automatic dataset preprocessing and augmentation



## 🔄 Project Workflow

The system works in two stages:

### 1. Person Detection
Detects all people present in the uploaded image using YOLOv8.

### 2. Helmet Verification
Checks the upper region of each detected person and determines whether a helmet overlaps with the head area.

**Results:**
- 🟢 Green box → Helmet detected
- 🔴 Red box → No helmet detected

> This method improves accuracy and reduces false detections.



## 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Object detection model |
| OpenCV | Image processing |
| Streamlit | Web interface |
| NumPy | Numerical operations |
| Pillow | Image handling |
| Scikit-learn | Data splitting utilities |



## ⚙️ Installation

Install the required packages:

```bash
pip install ultralytics streamlit opencv-python numpy pillow scikit-learn
```



## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run APP.PY
```

Upload an image and the system will automatically detect helmet violations.



## 🔧 Automatic Data Processing

The system automatically performs:
- Dataset cleaning
- Duplicate removal
- Label validation
- Image resizing
- Contrast enhancement using CLAHE
- Data augmentation
- Train/validation/test splitting



## 📊 Model Parameters

| Parameter | Purpose | Default |
|---|---|---|
| Person confidence | Person detection threshold | 0.4 |
| Helmet confidence | Helmet detection threshold | 0.10 |
| IoU threshold | Helmet-person matching | 0.08 |



## 📋 Requirements

- Python 3.8+
- ultralytics
- streamlit
- opencv-python
- numpy
- pillow
- scikit-learn


## 🚀 Future Improvements

- Real-time webcam detection
- Video surveillance support
- Mobile deployment
- Cloud-based monitoring dashboard
- Audio warning alerts
- Multi-class PPE detection



## 📄 License

This project is open-source and available for educational and research purposes.
