# 🧠 DeepFake Image Detection System using Deep Learning

## 📌 Project Overview

The rapid growth of Artificial Intelligence and Deep Learning technologies has led to the creation of highly realistic manipulated media known as **DeepFakes**. DeepFake images are artificially generated or modified images that can closely resemble real human faces, making it difficult to identify whether an image is authentic or fake.

This project presents a **DeepFake Image Detection System** developed using **Deep Learning**, **TensorFlow**, **Keras**, **MobileNetV2**, and **Streamlit**. The system is designed to analyze uploaded facial images and classify them as either:

- ✅ Real Image
- ❌ Fake / DeepFake Image

The project utilizes **Transfer Learning** with the pretrained **MobileNetV2** Convolutional Neural Network (CNN) architecture to improve prediction accuracy while reducing training time.

The application is deployed using **Streamlit**, providing an interactive and user-friendly web interface where users can upload images and instantly receive prediction results along with confidence scores.

This project demonstrates a complete end-to-end Deep Learning workflow including:

- Data preprocessing
- Image normalization
- CNN-based image classification
- Transfer learning
- Model evaluation
- Real-time prediction
- Streamlit deployment
- VS Code development environment


# 🎯 Objectives of the Project

The main objectives of this project are:

- To build an AI-powered DeepFake detection system
- To classify facial images into real and fake categories
- To implement transfer learning using MobileNetV2
- To understand image preprocessing and model training techniques
- To create an interactive web application using Streamlit
- To improve awareness about manipulated media and AI-generated content


# 🚀 Features

- ✅ DeepFake image classification
- ✅ Real-time image prediction
- ✅ Transfer Learning using MobileNetV2
- ✅ Interactive Streamlit web interface
- ✅ Upload support for JPG, JPEG, and PNG images
- ✅ Confidence score display
- ✅ Image preview functionality
- ✅ Image preprocessing pipeline
- ✅ Lightweight and fast prediction model
- ✅ User-friendly interface
- ✅ Developed completely in VS Code


# 🧰 Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow |
| Neural Network Library | Keras |
| CNN Architecture | MobileNetV2 |
| Image Processing | Pillow (PIL) |
| Data Processing | NumPy |
| Web Framework | Streamlit |
| Development Environment | Visual Studio Code (VS Code) |


# 💻 Developed Using VS Code

This project was fully developed and executed using **Visual Studio Code (VS Code)**.

VS Code was used for:

- Writing and editing Python code
- Running Streamlit applications
- Managing project folders and files
- Installing required Python libraries
- Debugging and testing the project
- Training and evaluating the model

The integrated terminal in VS Code made it easy to execute commands and manage dependencies efficiently.


# 📊 Dataset

The dataset used in this project contains two categories of facial images:

- Real Images
- Fake / DeepFake Images

The images are used to train the CNN model for binary image classification.

The dataset is divided into:

- Training dataset
- Validation dataset
- Testing dataset

The model learns patterns and facial inconsistencies from fake images and differentiates them from real images.


# ⚙️ Installation and Setup

## 1️⃣ Install Required Software

Before running the project, install the following software:

- Python
- Visual Studio Code (VS Code)
- Python Extension for VS Code


## 2️⃣ Open Project in VS Code

1. Open Visual Studio Code
2. Click on **File → Open Folder**
3. Select the project folder


## 3️⃣ Create Virtual Environment

Open the VS Code terminal and run the following command:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


## 4️⃣ Install Required Libraries

Install all required libraries using:

```bash
pip install tensorflow streamlit pillow numpy
```

Or install directly from requirements file:

```bash
pip install -r requirements.txt
```



# ▶️ Running the Application

Run the following command in the VS Code terminal:

```bash
streamlit run app.py
```


# 🌐 Application Workflow

The application works in the following steps:

1. User uploads an image
2. Uploaded image is displayed
3. Image is resized to `224x224`
4. Image is converted into NumPy array format
5. Image normalization is applied
6. The trained MobileNetV2 model analyzes the image
7. Prediction result is generated
8. Output is displayed:
   - Real Image
   - Fake Image
9. Confidence score is shown to the user

# 🧹 Data Preprocessing

Data preprocessing is one of the most important steps in Deep Learning.

The preprocessing pipeline includes:

- Image resizing (`224x224`)
- Image normalization
- Conversion to NumPy arrays
- Batch processing
- Data augmentation techniques

These preprocessing techniques improve:

- Model performance
- Prediction accuracy
- Generalization capability


# 🧠 Model Architecture

This project uses **MobileNetV2**, a pretrained Convolutional Neural Network architecture optimized for lightweight and efficient image classification tasks.

### Why MobileNetV2?

- Lightweight architecture
- Faster training
- High accuracy
- Efficient for real-time applications
- Suitable for transfer learning

## 📌 Model Pipeline

```text
Input Image (224x224)
        ↓
Image Preprocessing
        ↓
MobileNetV2 Backbone
        ↓
Global Average Pooling
        ↓
Dense Layer
        ↓
Dropout Layer
        ↓
Sigmoid Output Layer
        ↓
Prediction Result
```

# 🔍 Prediction System

The trained model predicts whether the uploaded image is:

- ✅ Real Image
- ❌ Fake Image

The prediction is displayed along with:

- Confidence score
- Uploaded image preview
- Image details


# 📈 Evaluation Metrics

The model performance can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help measure the effectiveness of the DeepFake detection system.

# 📸 Application Features

| Feature | Description |
|----------|-------------|
| 📤 Upload Image | Upload JPG, JPEG, PNG files |
| 🖼️ Image Preview | Displays uploaded image |
| 🔍 Prediction | Detects Fake or Real |
| 📊 Confidence Score | Shows prediction confidence |
| ℹ️ Image Information | Displays image size and mode |
| ⚡ Fast Processing | Lightweight prediction model |


# ❗ Common Errors and Solutions

| Error | Solution |
|-------|----------|
| Python not recognized | Add Python to PATH |
| ModuleNotFoundError | Install missing library |
| Streamlit not recognized | Install Streamlit using pip |
| Model file not found | Ensure `model.keras` exists |
| Invalid image format | Upload JPG, JPEG, or PNG images |


# 📉 Future Improvements

The following enhancements can be added in the future:

- Video DeepFake Detection
- Real-time Webcam Detection
- Explainable AI (Grad-CAM)
- Better CNN architectures
- Cloud deployment
- Mobile application support
- Improved UI/UX design

# 🎯 Learning Outcomes

This project helps in understanding:

- Deep Learning workflows
- CNN architectures
- Transfer Learning
- Image preprocessing
- Streamlit deployment
- AI-based image classification
- Real-world AI applications

# 🤝 Contributing

Contributions are welcome.

Steps to contribute:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request


