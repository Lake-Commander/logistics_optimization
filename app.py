import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# === Page Config ===
st.set_page_config(page_title="Delivery Delay Predictor", layout="wide")

# === Load Model & Scaler ===
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# === Helper: Expected Columns ===
expected_columns = [
    "payment_type", "profit_per_order", "sales_per_customer", "category_id", "category_name",
    "customer_city", "customer_country", "customer_id", "customer_segment", "customer_state", "customer_zipcode",
    "department_id", "department_name", "latitude", "longitude", "market", "order_city", "order_country",
    "order_customer_id", "order_date", "order_id", "order_item_cardprod_id", "order_item_discount",
    "order_item_discount_rate", "order_item_id", "order_item_product_price", "order_item_profit_ratio",
    "order_item_quantity", "sales", "order_item_total_amount", "order_profit_per_order", "order_region",
    "order_state", "order_status", "product_card_id", "product_category_id", "product_name", "product_price",
    "shipping_date", "shipping_mode", "label", "is_late_shipping", "is_weekend_shipping", "order_weekday",
    "shipping_delay_days"
]

# === Helper: Generate Predictions ===
def preprocess_and_predict(data):
    missing = set(expected_columns) - set(data.columns)
    if missing:
        st.error(f"Prediction failed: columns are missing: {missing}")
        return None

    X = data[expected_columns].copy()
    if scaler:
        X_scaled = scaler.transform(X.select_dtypes(include=np.number))
        X[X.select_dtypes(include=np.number).columns] = X_scaled

    predictions = model.predict(X)
    return predictions

# === Tabs ===
tabs = st.sidebar.radio("Navigation", ["📊 Dashboard", "📁 Predict", "📋 Input Schema"])

# === Tab 1: Dashboard ===
if tabs == "📊 Dashboard":
    st.title("📊 Exploratory Dashboard")
    st.markdown("Visual insights from training data")

    viz_files = [
        ("delivery_status_distribution.png", "Delivery Status Distribution"),
        ("discount_vs_label.png", "Discount vs Delivery Status"),
        ("missing_values_matrix.png", "Missing Value Matrix"),
        ("shipping_mode_vs_label.png", "Shipping Mode vs Delivery Status"),
        ("top10_regions_vs_label.png", "Top 10 Regions vs Delivery Status"),
    ]

    for file_name, title in viz_files:
        path = os.path.join("eda/plots", file_name)
        if os.path.exists(path):
            st.subheader(title)
            st.image(Image.open(path), use_column_width=True)
        else:
            st.warning(f"⚠️ {file_name} not found.")

# === Tab 2: Predict ===
elif tabs == "📁 Predict":
    st.title("📁 Predict Delivery Delay")
    uploaded_file = st.file_uploader("Upload a CSV file with required columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", df.head())

        with st.spinner("Predicting..."):
            preds = preprocess_and_predict(df)
            if preds is not None:
                df['Predicted_Label'] = preds
                st.success("✅ Prediction complete!")
                st.write(df[['order_id', 'Predicted_Label']].head())
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

# === Tab 3: Input Schema ===
elif tabs == "📋 Input Schema":
    st.title("📋 Expected Input Schema")
    st.markdown("All fields below are required. Ensure correct formatting (especially for dates and categorical fields).")

    schema_dict = {
        "payment_type": "Mode of payment (e.g., Credit Card, Cash)",
        "profit_per_order": "Profit made per order",
        "sales_per_customer": "Total sales value per customer",
        "category_id": "ID of the product category",
        "category_name": "Name of the product category",
        "customer_city": "Customer's city",
        "customer_country": "Customer's country",
        "customer_id": "Unique customer ID",
        "customer_segment": "Customer segmentation (e.g., Consumer, Corporate)",
        "customer_state": "Customer's state",
        "customer_zipcode": "Customer's ZIP code",
        "department_id": "ID of the department selling the product",
        "department_name": "Name of the department",
        "latitude": "Shipping location latitude",
        "longitude": "Shipping location longitude",
        "market": "Geographic market segment (e.g., Africa, EU)",
        "order_city": "City where order was placed",
        "order_country": "Country where order was placed",
        "order_customer_id": "Customer ID used in the order",
        "order_date": "Date when the order was placed (YYYY-MM-DD)",
        "order_id": "Unique identifier of the order",
        "order_item_cardprod_id": "Product ID in the order",
        "order_item_discount": "Discount amount on the item",
        "order_item_discount_rate": "Discount as a percentage",
        "order_item_id": "ID of the order item",
        "order_item_product_price": "Price of the product",
        "order_item_profit_ratio": "Profit ratio on the product",
        "order_item_quantity": "Quantity of the item ordered",
        "sales": "Total sales amount",
        "order_item_total_amount": "Total amount for the item",
        "order_profit_per_order": "Profit for this entire order",
        "order_region": "Region of the order",
        "order_state": "State of the order",
        "order_status": "Status (e.g., Completed, Cancelled)",
        "product_card_id": "ID of the product card",
        "product_category_id": "Category ID of the product",
        "product_name": "Name of the product",
        "product_price": "Listed price of the product",
        "shipping_date": "Date when the item was shipped (YYYY-MM-DD)",
        "shipping_mode": "Mode of shipping (e.g., First Class, Standard)",
        "label": "(Optional for prediction) Whether order was delayed (for training only)",
        "is_late_shipping": "Boolean: Was the shipping late? (can be auto-generated)",
        "is_weekend_shipping": "Boolean: Did shipping occur on a weekend?",
        "order_weekday": "Day of week order was placed (0=Monday, 6=Sunday)",
        "shipping_delay_days": "Number of days between order and shipping"
    }

    schema_df = pd.DataFrame(list(schema_dict.items()), columns=["Column Name", "Description"])
    st.dataframe(schema_df, use_container_width=True)
