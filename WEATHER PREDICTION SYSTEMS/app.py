# =========================================================
# 🌦️ WEATHER PREDICTION STREAMLIT APP
# =========================================================

import streamlit as st
import numpy as np
import joblib

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/best_weather_model.pkl"
)

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Weather Prediction",
    page_icon="🌦️",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🌦️ AI Weather Prediction System")

st.markdown(
    """
    Predict rainfall using machine learning
    based on atmospheric conditions.
    """
)

# =========================================================
# INPUTS
# =========================================================

temperature = st.slider(
    "Temperature",
    0.0,
    50.0,
    25.0
)

humidity = st.slider(
    "Humidity",
    0.0,
    100.0,
    50.0
)

wind_speed = st.slider(
    "Wind Speed",
    0.0,
    30.0,
    5.0
)

cloud_cover = st.slider(
    "Cloud Cover",
    0.0,
    100.0,
    40.0
)

pressure = st.slider(
    "Pressure",
    950.0,
    1050.0,
    1013.0
)

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Weather"):

    features = np.array([[
        temperature,
        humidity,
        wind_speed,
        cloud_cover,
        pressure
    ]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    result = encoder.inverse_transform(
        prediction
    )[0]

    confidence = np.max(probability) * 100

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    if result.lower() == "rain":

        st.error(
            f"🌧️ Prediction: {result}"
        )

    else:

        st.success(
            f"☀️ Prediction: {result}"
        )

    st.info(
        f"Confidence Score: {confidence:.2f}%"
    )

    # =====================================================
    # PROBABILITY DISPLAY
    # =====================================================

    st.subheader("Prediction Probabilities")

    st.write(
        {
            class_name: float(prob)
            for class_name, prob in zip(
                encoder.classes_,
                probability[0]
            )
        }
    )