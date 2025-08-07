import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# === Page Configuration ===
st.set_page_config(page_title="SwiftChain Delay Predictor", layout="wide")

# === Load Model and Preprocessor ===
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    return model

model = load_model()
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# === Load Sample Input Template ===
@st.cache_data
def load_sample():
    return pd.read_csv("logistics_featurized.csv").drop(columns=['label'], errors='ignore')

sample_df = load_sample()
input_columns = sample_df.columns

# === Sidebar ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2649/2649222.png", width=100)
st.sidebar.title("📦 SwiftChain Predictor")
st.sidebar.markdown("Predict whether an order will be **early**, **on-time**, or **delayed** based on logistics and order features.")

# === Main Interface ===
st.title("🚚 Delivery Delay Prediction App")
st.markdown("This app predicts whether a delivery will arrive early, on time, or be delayed using historical logistics data and machine learning.")

# === Input Options ===
mode = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with input data:", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    st.subheader("Enter Order Details")
    input_data = {}
    for col in input_columns:
        dtype = sample_df[col].dtype
        if "date" in col:
            input_data[col] = st.date_input(f"{col.replace('_', ' ').title()}", datetime.today())
        elif dtype == 'float64' or dtype == 'int64':
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)
        elif sample_df[col].nunique() <= 15:
            options = sample_df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options)
        else:
            input_data[col] = st.text_input(f"{col.replace('_', ' ').title()}", "")

    input_df = pd.DataFrame([input_data])

# === Prediction Button ===
if st.button("🔍 Predict Delivery Status"):
    try:
        # Apply same preprocessing
        processed_input = preprocessor.transform(input_df)
        prediction = classifier.predict(processed_input)

        # Map prediction back to label
        if isinstance(classifier, joblib.load("models/best_model.pkl").named_steps['classifier'].__class__):
            label_map = {-1: "🟢 Early", 0: "🟡 On-Time", 1: "🔴 Delayed"}
        else:
            label_map = {0: "🟢 Early", 1: "🟡 On-Time", 2: "🔴 Delayed"}
            prediction = pd.Series(prediction).replace({0: -1, 1: 0, 2: 1})

        st.success(f"Predicted Delivery Status: **{label_map[int(prediction[0])]}**")

    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

# === Feature Importance Section ===
st.markdown("---")
st.header("📊 Top Predictive Features")
image = Image.open("output/phase5/feature_importance.png")
st.image(image, caption="Top 15 Features Influencing Delivery Status", use_column_width=True)

# === Insights and Recommendations ===
st.markdown("---")
st.header("💡 Insights & Recommendations")

with open("output/phase5/insights_and_recommendations.txt", "r", encoding="utf-8") as f:
    content = f.read()

st.code(content)
