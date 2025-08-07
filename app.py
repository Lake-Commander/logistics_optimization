import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# === CONFIGURATION ===
st.set_page_config(page_title="SwiftChain Delivery Delay Predictor", layout="wide")

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

# === Tabs ===
tabs = st.tabs(["📦 Predict Delay", "📑 Expected Schema", "📊 Dashboard"])

# === Tab 1: Predict Delay ===
with tabs[0]:
    st.title("🚚 SwiftChain Delivery Delay Prediction")
    st.markdown("""
    SwiftChain Analytics specializes in transforming logistics data into operational efficiency.  
    This app predicts whether a customer order will be delivered **early**, **on time**, or **late** based on historical order data.

    📦 Built by a contracted Data Scientist for SwiftChain's 2024 supply chain intelligence initiative.
    """)

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

    st.header("🧾 Upload Order Data")
    uploaded_file = st.file_uploader("Upload a CSV file (must match featurized columns)", type=["csv"])

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        else:
        input_df = sample_df.copy()
        source_label = "📁 Default Sample Data: `logistics_cleaned.csv`"

    st.caption(f"Data Source: {source_label}")

        uploaded_cols = set(input_df.columns)
        required_cols = set(expected_cols)
        missing = required_cols - uploaded_cols
        used_cols = uploaded_cols & required_cols

        if missing:
            st.warning(f"⚠️ Missing columns in uploaded file: {', '.join(sorted(missing))}")
            if len(used_cols) < 5:
                st.error("🚫 Not enough valid columns to proceed with prediction.")
                st.stop()

        input_df = input_df[list(used_cols)]
        aligned_df = pd.DataFrame(columns=expected_cols)
        for col in expected_cols:
            aligned_df[col] = input_df[col] if col in input_df.columns else np.nan

        st.subheader("🔍 Predict Delivery Status")
        if st.button("Run Prediction"):
            try:
                processed_input = preprocessor.transform(aligned_df)
                prediction = classifier.predict(processed_input)

                label_map = {-1: "🔴 Late", 0: "🟡 On Time", 1: "🟢 Early"}
                result = [label_map.get(int(p), "⚠️ Unknown") for p in prediction]

                input_df["Predicted Status"] = result
                st.success("✅ Prediction Complete")
                st.dataframe(input_df, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("📤 Please upload a CSV file to begin prediction.")

# === Tab 2: Expected Schema ===
with tabs[1]:
    st.subheader("📑 Expected Input Schema")
    st.info("All fields below are required for prediction. Ensure correct formatting (especially for dates and categorical fields).")

    schema_data = [
        {"Column Name": "payment_type", "Description": "Mode of payment (e.g., Credit Card, Cash)"},
        {"Column Name": "profit_per_order", "Description": "Profit made per order"},
        {"Column Name": "sales_per_customer", "Description": "Total sales value per customer"},
        {"Column Name": "category_id", "Description": "ID of the product category"},
        {"Column Name": "category_name", "Description": "Name of the product category"},
        {"Column Name": "customer_city", "Description": "Customer's city"},
        {"Column Name": "customer_country", "Description": "Customer's country"},
        {"Column Name": "customer_id", "Description": "Unique customer ID"},
        {"Column Name": "customer_segment", "Description": "Customer segmentation (e.g., Consumer, Corporate)"},
        {"Column Name": "customer_state", "Description": "Customer's state"},
        {"Column Name": "customer_zipcode", "Description": "Customer's ZIP code"},
        {"Column Name": "department_id", "Description": "ID of the department selling the product"},
        {"Column Name": "department_name", "Description": "Name of the department"},
        {"Column Name": "latitude", "Description": "Shipping location latitude"},
        {"Column Name": "longitude", "Description": "Shipping location longitude"},
        {"Column Name": "market", "Description": "Geographic market segment (e.g., Africa, EU)"},
        {"Column Name": "order_city", "Description": "City where order was placed"},
        {"Column Name": "order_country", "Description": "Country where order was placed"},
        {"Column Name": "order_customer_id", "Description": "Customer ID used in the order"},
        {"Column Name": "order_date", "Description": "Date when the order was placed (YYYY-MM-DD)"},
        {"Column Name": "order_id", "Description": "Unique identifier of the order"},
        {"Column Name": "order_item_cardprod_id", "Description": "Product ID in the order"},
        {"Column Name": "order_item_discount", "Description": "Discount amount on the item"},
        {"Column Name": "order_item_discount_rate", "Description": "Discount as a percentage"},
        {"Column Name": "order_item_id", "Description": "ID of the order item"},
        {"Column Name": "order_item_product_price", "Description": "Price of the product"},
        {"Column Name": "order_item_profit_ratio", "Description": "Profit ratio on the product"},
        {"Column Name": "order_item_quantity", "Description": "Quantity of the item ordered"},
        {"Column Name": "sales", "Description": "Total sales amount"},
        {"Column Name": "order_item_total_amount", "Description": "Total amount for the item"},
        {"Column Name": "order_profit_per_order", "Description": "Profit for this entire order"},
        {"Column Name": "order_region", "Description": "Region of the order"},
        {"Column Name": "order_state", "Description": "State of the order"},
        {"Column Name": "order_status", "Description": "Status (e.g., Completed, Cancelled)"},
        {"Column Name": "product_card_id", "Description": "ID of the product card"},
        {"Column Name": "product_category_id", "Description": "Category ID of the product"},
        {"Column Name": "product_name", "Description": "Name of the product"},
        {"Column Name": "product_price", "Description": "Listed price of the product"},
        {"Column Name": "shipping_date", "Description": "Date when the item was shipped (YYYY-MM-DD)"},
        {"Column Name": "shipping_mode", "Description": "Mode of shipping (e.g., First Class, Standard)"},
        {"Column Name": "label", "Description": "(Optional for prediction) Whether order was delayed (for training only)"},
        {"Column Name": "is_late_shipping", "Description": "Boolean: Was the shipping late? (can be auto-generated)"},
        {"Column Name": "is_weekend_shipping", "Description": "Boolean: Did shipping occur on a weekend?"},
        {"Column Name": "order_weekday", "Description": "Day of week order was placed (0=Monday, 6=Sunday)"},
        {"Column Name": "shipping_delay_days", "Description": "Number of days between order and shipping"},
    ]
    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True)

# === Tab 3: Dashboard ===
with tabs[2]:
    st.subheader("📊 Visual Analytics Dashboard")

    viz_files = [
        ("delivery_status_distribution.png", "Delivery Status Distribution"),
        ("discount_vs_label.png", "Discount vs Delivery Status"),
        ("missing_values_matrix.png", "Missing Value Matrix"),
        ("shipping_mode_vs_label.png", "Shipping Mode vs Delivery Status"),
        ("top10_regions_vs_label.png", "Top 10 Regions vs Delivery Status"),
    ]

    for file, caption in viz_files:
        try:
            st.image(f"eda/plots/{file}", use_container_width=True, caption=caption)
        except:
            st.warning(f"{file} not found.")

    st.markdown("---")
    st.subheader("📌 Top Predictive Features")
    try:
        st.image("output/phase5/feature_importance.png", use_container_width=True, caption="Top 15 Feature Importances")
    except:
        st.warning("Feature importance image not found.")

    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    try:
        with open("output/phase5/insights_and_recommendations.txt", "r", encoding="utf-8") as f:
            content = f.read()
        st.code(content, language='markdown')
    except:
        st.warning("Insights file not found.")

# === Footer ===
st.markdown("---")
st.markdown("© 2024 SwiftChain Analytics | Empowering global logistics through predictive intelligence.")
