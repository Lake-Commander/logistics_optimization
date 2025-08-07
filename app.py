import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
expected_features = sample_df.columns.tolist()

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

# === Input File Upload Only ===
st.header("🧾 Upload Order Data")
uploaded_file = st.file_uploader("Upload a CSV file (must match featurized format)", type=["csv"])

if uploaded_file is not None:
    input_df_raw = pd.read_csv(uploaded_file)

    # Intersect columns with what the model expects
    matching_cols = [col for col in expected_features if col in input_df_raw.columns]
    if not matching_cols:
        st.error("❌ None of the required features were found in your uploaded file.")
        st.stop()

    input_df = input_df_raw[matching_cols]

    st.markdown(f"✅ Using **{len(matching_cols)}** matched features out of {len(expected_features)} expected.")
    st.dataframe(input_df.head())

    # === Prediction ===
    st.subheader("🔍 Predict Delivery Status")
    if st.button("Run Prediction"):
        try:
            processed_input = preprocessor.transform(input_df)
            predictions = classifier.predict(processed_input)

            label_map = {-1: "🔴 Late", 0: "🟡 On Time", 1: "🟢 Early"}
            decoded = [label_map.get(int(pred), "⚠️ Unknown") for pred in predictions]

            st.success("✅ Prediction Results:")
            result_df = input_df.copy()
            result_df["Predicted Status"] = decoded
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("📂 Please upload a CSV file to proceed.")

# === Feature Importance ===
st.markdown("---")
st.header("📌 Top Predictive Features")
try:
    st.image("output/phase5/feature_importance.png", use_column_width=True, caption="Top 15 Feature Importances")
except:
    st.warning("⚠️ Feature importance image not found.")

# === Insights & Recommendations ===
st.markdown("---")
st.header("💡 Key Insights & Recommendations")

try:
    with open("output/phase5/insights_and_recommendations.txt", "r", encoding="utf-8") as f:
        content = f.read()
    st.code(content, language='markdown')
except:
    st.warning("⚠️ Insights file not found.")

# === Footer ===
st.markdown("---")
st.markdown("""
© 2024 SwiftChain Analytics  
Empowering global logistics through predictive intelligence.
""")
