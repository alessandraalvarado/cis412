import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Load trained models
# ------------------------------
log_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write("Predict whether a passenger will be satisfied based on 5 service factors.")

# ------------------------------
# User Inputs
# ------------------------------
st.header("Passenger Ratings (1–5 scale)")

seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
inflight_service = st.slider("Inflight Service", 1, 5, 3)
cleanliness = st.slider("Cleanliness", 1, 5, 3)
online_boarding = st.slider("Online Boarding", 1, 5, 3)
onboard_service = st.slider("On-board Service", 1, 5, 3)

features = np.array([[seat_comfort, inflight_service, cleanliness,
                      online_boarding, onboard_service]])

# ------------------------------
# Model Selection
# ------------------------------
model_choice = st.selectbox(
    "Choose Model",
    ("Logistic Regression", "Random Forest")
)

if st.button("Predict Satisfaction"):
    if model_choice == "Logistic Regression":
        prediction = log_model.predict(features)[0]
    else:
        prediction = rf_model.predict(features)[0]

    st.subheader("Prediction:")
    if prediction == 1:
        st.success("😊 The passenger is **SATISFIED**.")
    else:
        st.error("😕 The passenger is **NOT satisfied**.")
