
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
expected_cols = sample_df.columns.tolist()

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

# === Upload CSV Input Only ===
st.header("🧾 Upload Order Data")
uploaded_file = st.file_uploader("Upload a CSV file (must match featurized columns)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Find mismatched columns
    uploaded_cols = set(input_df.columns)
    required_cols = set(expected_cols)
    missing = required_cols - uploaded_cols
    used_cols = uploaded_cols & required_cols

    if missing:
        st.warning(f"⚠️ Missing columns in uploaded file: {', '.join(sorted(missing))}")
        if len(used_cols) < 5:
            st.error("🚫 Not enough valid columns to proceed with prediction.")
            st.stop()

    input_df = input_df[list(used_cols)]  # keep only matching columns

    # Align column order to model expectation
    aligned_df = pd.DataFrame(columns=expected_cols)
    for col in expected_cols:
        aligned_df[col] = input_df[col] if col in input_df.columns else np.nan

    st.subheader("🔍 Predict Delivery Status")
    if st.button("Run Prediction"):
        try:
            processed_input = preprocessor.transform(aligned_df)
            prediction = classifier.predict(processed_input)

            # Decode label
            label_map = {-1: "🔴 Late", 0: "🟡 On Time", 1: "🟢 Early"}
            result = [label_map.get(int(p), "⚠️ Unknown") for p in prediction]

            input_df["Predicted Status"] = result
            st.success("✅ Prediction Complete")
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("📤 Please upload a CSV file to begin prediction.")

# === Visual Analytics ===
st.markdown("---")
st.header("📊 Visual Analytics Dashboard")

viz_files = [
    ("delivery_status_distribution.png", "Delivery Status Distribution"),
    ("discount_vs_label.png", "Discount vs Delivery Status"),
    ("missing_values_matrix.png", "Missing Value Matrix"),
    ("shipping_mode_vs_label.png", "Shipping Mode vs Delivery Status"),
    ("top10_regions_vs_label.png", "Top 10 Regions vs Delivery Status"),
]

for file, caption in viz_files:
    try:
        st.image(f"output/phase5/{file}", use_column_width=True, caption=caption)
    except:
        st.warning(f"{file} not found.")

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
