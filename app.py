import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="House Price Prediction")

# Load artifacts
model = joblib.load("rf_regressor.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("House Price Prediction")
st.write("Random Forest Regressor")

# Dynamic input UI
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted House Price: {prediction:,.2f}")
