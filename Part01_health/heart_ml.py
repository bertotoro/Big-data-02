import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Set title for the app
st.title("Heart Attack Prediction Model")

# File uploader for dataset
# Model upload for prediction
st.subheader("Upload a Saved Model for Prediction")
uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

if uploaded_model is not None:
    model = joblib.load(uploaded_model)  # Load the uploaded model

    # Dataset attributes description
    st.subheader("Dataset Attributes")
    st.write("""
    The dataset contains 14 attributes related to heart health. Please fill in the values based on the following attributes:
    1. **Age**: Age (in years)
    2. **Sex**: Gender (1 = male; 0 = female)
    3. **Chest Pain**: Chest pain type:
       - 0: Typical angina (all criteria present)
       - 1: Atypical angina (two of three criteria satisfied)
       - 2: Non-anginal pain (less than one criteria satisfied)
       - 3: Asymptomatic (none of the criteria are satisfied)
    4. **Resting Blood Pressure**: (in mmHg, upon admission to the hospital)
    5. **Cholesterol**: Serum cholesterol in mg/dL
    6. **Fasting Blood Sugar**: > 120 mg/dL (likely to be diabetic; 1 = true; 0 = false)
    7. **Resting Electrocardiogram Results**:
       - 0: Normal
       - 1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
       - 2: Left ventricular hypertrophy by Estes' criteria
    8. **Maximum Heart Rate Achieved**: Greatest number of beats per minute during all-out strenuous exercise.
    9. **Exercise Induced Angina**: (1 = yes; 0 = no)
    10. **Oldpeak**: ST depression induced by exercise relative to rest (in mm).
    11. **Slope**: The slope of the peak exercise ST segment:
        - 0: Upsloping
        - 1: Flat
        - 2: Downsloping
    12. **Number of Major Vessels**: (0-4) colored by fluoroscopy.
    13. **Thalassemia**: 
        - 0: Normal
        - 1: Fixed defect
        - 2: Reversible defect
    """)

    # Sample input data for prediction
    st.subheader("Input Sample Data for Prediction")

    # Input fields corresponding to the dataset
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
    trtbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=0)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=500, value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", options=[0, 1])
    rest_ecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=0)
    exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", options=[0, 1])
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, value=0)

    # Prepare input data for prediction (ensure all 14 features are included)
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, rest_ecg, thalach, exang, ca, oldpeak, slope, thal]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.subheader("Prediction Result")

        # Check the prediction and display the result
        if prediction[0] == 0:
            st.write("The predicted class is: 0 (Less Chance of Heart Attack)")
        else:
            st.write("The predicted class is: 1 (More Chance of Heart Attack)")
