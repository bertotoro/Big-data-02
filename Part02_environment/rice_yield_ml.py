import streamlit as st
import pandas as pd
import joblib  # Import joblib to load the model

# Function to predict based on the input features
def predict_rice_yield(features):
    return model.predict(features)

st.title("Rice Yield Prediction App")

# File uploader for the model
uploaded_file = st.file_uploader("Upload your trained model (joblib format)", type=["pkl", "joblib"])

# Load the model if a file is uploaded
if uploaded_file is not None:
    model = joblib.load(uploaded_file)

    # Display expected feature names
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        st.write("Feature names cannot be retrieved from this model.")

    st.header("Input Features for Rice Yield Prediction")

    # User inputs for rice yield features
    annual = st.number_input("Annual Rainfall (in mm)", min_value=0.0, value=1103.9, format="%.2f")
    avg_rain = st.number_input("Average Rainfall (in mm)", min_value=0.0, value=62.53, format="%.2f")
    nitrogen = st.number_input("Nitrogen (in kg/ha)", min_value=0.0, value=52888.0, format="%.1f")
    potash = st.number_input("Potash (in kg/ha)", min_value=0.0, value=10466.0, format="%.1f")
    phosphate = st.number_input("Phosphate (in kg/ha)", min_value=0.0, value=23912.0, format="%.1f")
    dystropepts = st.number_input("Dystropepts (in %)", min_value=0.0, value=0.0, format="%.2f")
    fluvents = st.number_input("Fluvents (in %)", min_value=0.0, value=0.0, format="%.2f")
    inceptisols = st.number_input("Inceptisols (in %)", min_value=0.0, value=0.0, format="%.2f")
    loamy_alfisol = st.number_input("Loamy Alfisol (in %)", min_value=0.0, value=0.6, format="%.2f")
    orthents = st.number_input("Orthents (in %)", min_value=0.0, value=0.0, format="%.2f")
    orthids = st.number_input("Orthids (in %)", min_value=0.0, value=0.0, format="%.2f")
    psamments = st.number_input("Psamments (in %)", min_value=0.0, value=0.0, format="%.2f")
    sandy_alfisol = st.number_input("Sandy Alfisol (in %)", min_value=0.0, value=0.0, format="%.2f")
    udalfs = st.number_input("Udalf (in %)", min_value=0.0, value=0.0, format="%.2f")
    udolls_udalfs = st.number_input("Udolls Udalfs (in %)", min_value=0.0, value=0.0, format="%.2f")
    udupts_udalfs = st.number_input("Udupts Udalfs (in %)", min_value=0.0, value=0.0, format="%.2f")
    ustalf_ustolls = st.number_input("Ustalf Ustolls (in %)", min_value=0.0, value=0.4, format="%.2f")
    ustalfs = st.number_input("Ustalfs (in %)", min_value=0.0, value=0.0, format="%.2f")
    vertic_soils = st.number_input("Vertic Soils (in %)", min_value=0.0, value=0.0, format="%.2f")
    vertisols = st.number_input("Vertisols (in %)", min_value=0.0, value=0.0, format="%.2f")
    rice_production = st.number_input("Rice Production (in tons)", min_value=0.0, value=984.31, format="%.2f")
    # Removed rice_yield from input since we are predicting it

    # Prepare the input for prediction
    input_features = pd.DataFrame({
        'ANNUAL': [annual],
        'avg_rain': [avg_rain],
        'Nitrogen': [nitrogen],
        'POTASH': [potash],
        'PHOSPHATE': [phosphate],
        'DYSTROPEPTS': [dystropepts],
        'FLUVENTS': [fluvents],
        'INCEPTISOLS': [inceptisols],
        'LOAMY_ALFISOL': [loamy_alfisol],
        'ORTHENTS': [orthents],
        'ORTHIDS': [orthids],
        'PSAMMENTS': [psamments],
        'SANDY_ALFISOL': [sandy_alfisol],
        'UDALFS': [udalfs],
        'UDOLLS_UDALFS': [udolls_udalfs],
        'UDUPTS_UDALFS': [udupts_udalfs],
        'USTALF_USTOLLS': [ustalf_ustolls],
        'USTALFS': [ustalfs],
        'VERTIC_SOILS': [vertic_soils],
        'VERTISOLS': [vertisols],
        'RICE_PRODUCTION': [rice_production]
        # Removed RICE_YIELD as it's the output we are predicting
    })

   

    if st.button("Predict"):
        try:
            prediction = predict_rice_yield(input_features)
            st.success(f"Predicted Rice Yield: {prediction[0]:.2f} tons/ha")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
else:
    st.warning("Please upload a trained model file.")
