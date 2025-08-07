import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from PIL import Image

# === CONFIGURATION ===
st.set_page_config(page_title="SwiftChain Delivery Delay Predictor", layout="wide")
st.title("🚚 SwiftChain Delivery Delay Prediction")
st.markdown("""
SwiftChain Analytics specializes in transforming logistics data into operational efficiency.  
This app predicts whether a customer order will be delivered **early**, **on time**, or **late** based on historical order data.

📦 Built by a contracted Data Scientist for SwiftChain's 2024 supply chain intelligence initiative.
""")

# === Load model and sample data ===
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_sample():
    df = pd.read_csv("logistics_featurized.csv").drop(columns=["label"], errors='ignore')
    return df

model = load_model()
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']
sample_df = load_sample()

# === Sidebar ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5972/5972511.png", width=80)
st.sidebar.title("📊 Project Details")
st.sidebar.markdown("""
**Company**: SwiftChain Analytics  
**Founded**: 2010 (HQ: Chicago)  
**Use Case**: Predicting delivery delays in real-time  
**Label**:  
- `-1`: Late  
- `0`: On Time  
- `1`: Early  
""")

# === Input Options ===
st.header("🧾 Provide Order Data")
input_mode = st.radio("Choose Input Mode", ["Manual Entry", "Upload CSV"])

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file (logistics_featurized format)", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    st.markdown("Fill in the order details below to get a prediction.")
    input_data = {}
    for col in sample_df.columns:
        dtype = sample_df[col].dtype
        if "date" in col:
            input_data[col] = st.date_input(f"{col.replace('_', ' ').title()}", datetime.today())
        elif dtype in ['float64', 'int64']:
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)
        elif sample_df[col].nunique() <= 15:
            options = sample_df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options)
        else:
            input_data[col] = st.text_input(f"{col.replace('_', ' ').title()}", "")

    input_df = pd.DataFrame([input_data])

# === Prediction ===
st.subheader("🔍 Predict Delivery Status")
if st.button("Run Prediction"):
    try:
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        prediction = classifier.predict(processed_input)

        # Decode label
        label_map = {-1: "🔴 Late", 0: "🟡 On Time", 1: "🟢 Early"}
        if isinstance(prediction[0], np.int64) or isinstance(prediction[0], int):
            result = label_map[int(prediction[0])]
        else:
            result = "⚠️ Unknown prediction"

        st.success(f"**Predicted Delivery Status**: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# === Feature Importance ===
st.markdown("---")
st.header("📌 Top Predictive Features")
try:
    st.image("output/phase5/feature_importance.png", use_column_width=True, caption="Top 15 Feature Importances")
except:
    st.warning("Feature importance image not found.")

# === Insights & Recommendations ===
st.markdown("---")
st.header("💡 Key Insights & Recommendations")

try:
    with open("output/phase5/insights_and_recommendations.txt", "r", encoding="utf-8") as f:
        content = f.read()
    st.code(content, language='markdown')
except:
    st.warning("Insights file not found.")

# === Footer ===
st.markdown("---")
st.markdown("""
© 2024 SwiftChain Analytics  
Empowering global logistics through predictive intelligence.
""")
