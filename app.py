import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from utils import load_image
from feature_engineering import preprocess_input

# === Page Config ===
st.set_page_config(page_title="Logistics Delay Predictor", layout="wide")

# === Load Model ===
model = joblib.load("models/best_model.pkl")

# === Title ===
st.title("📦 Logistics Delay Prediction Dashboard")
st.markdown("Upload your logistics data below to predict delivery delays.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocess
    X = preprocess_input(df)

    # Predict
    y_pred = model.predict(X)
    df['Predicted_Delay'] = y_pred

    st.subheader("Prediction Results")
    st.dataframe(df[['Predicted_Delay']])

    st.download_button("📥 Download Predictions", data=df.to_csv(index=False), file_name="predictions.csv")

# === EDA Dashboard ===
st.sidebar.markdown("### 📊 View EDA Visualizations")
graphs_dir = "output_graphs"

if st.sidebar.checkbox("Show Correlation Heatmap"):
    image = load_image(os.path.join(graphs_dir, "correlation", "correlation_heatmap.png"))
    if image:
        st.image(image, caption="Correlation Heatmap", use_column_width=True)

if st.sidebar.checkbox("Show Feature Distribution"):
    image = load_image(os.path.join(graphs_dir, "bivariate", "feature_distribution.png"))
    if image:
        st.image(image, caption="Feature Distribution", use_column_width=True)

if st.sidebar.checkbox("Show Delay by Category"):
    image = load_image(os.path.join(graphs_dir, "bivariate", "delay_by_category.png"))
    if image:
        st.image(image, caption="Delay by Category", use_column_width=True)
