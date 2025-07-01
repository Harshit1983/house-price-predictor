import streamlit as st
import pandas as pd
import joblib

# Load the trained model and metrics
model = joblib.load("house_price_model.pkl")
metrics = joblib.load("metrics.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("Enter house features to estimate the selling price.")

# Sidebar inputs
sqft = st.slider("Living Area (sq ft)", min_value=500, max_value=5000, value=1500, step=50)
bedrooms = st.selectbox("Number of Bedrooms", list(range(1, 11)), index=2)
bathrooms = st.selectbox("Number of Full Bathrooms", list(range(1, 6)), index=1)

# Predict button
if st.button("🔍 Predict Price"):
    input_df = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
    prediction = model.predict(input_df)[0]
    
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
    st.info(f"📊 R² Score: {metrics['r2']:.4f} | 🧮 MSE: {metrics['mse']:,.2f}")

# Optional: Display model info
with st.expander("ℹ️ Model Details"):
    st.write("This prediction is based on a Linear Regression model trained using:")
    st.write("• `GrLivArea` (sq ft)\n• `BedroomAbvGr` (bedrooms)\n• `FullBath` (bathrooms)")
    st.write("The model was trained using 80% of the dataset and evaluated on the remaining 20%.")
