import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# === Load Trained Model ===
model = joblib.load("models/best_model.pkl")

# === App Config ===
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")
st.title("🚚 Delivery Delay Prediction App")
st.write("Estimate if a shipment is likely to be **delayed** based on order and shipping details.")

# === Sidebar Input Form ===
st.sidebar.header("Enter Shipment Details")

with st.sidebar.form(key="input_form"):
    mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
    product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])
    cost_of_the_product = st.number_input("Cost of Product ($)", min_value=1, max_value=5000, value=100)
    weight_in_grams = st.number_input("Weight (grams)", min_value=10, max_value=100000, value=1000)
    discount_offered = st.slider("Discount Offered (%)", min_value=0, max_value=100, value=10)
    
    # Dates
    order_date = st.date_input("Order Date", value=datetime.today())
    shipping_date = st.date_input("Shipping Date", value=datetime.today())
    
    submit = st.form_submit_button("Predict Delay")

# === Process & Predict ===
if submit:
    # Derived features
    shipping_delay_days = (shipping_date - order_date).days
    order_weekday = order_date.weekday()
    is_weekend_shipping = 1 if shipping_date.weekday() >= 5 else 0
    is_late_shipping = 1 if shipping_delay_days > 3 else 0  # Or use the threshold from your EDA

    # Prepare input
    input_data = pd.DataFrame({
        "mode_of_shipment": [mode_of_shipment],
        "product_importance": [product_importance],
        "cost_of_the_product": [cost_of_the_product],
        "weight_in_grams": [weight_in_grams],
        "discount_rate": [discount_offered / 100],
        "shipping_delay_days": [shipping_delay_days],
        "is_late_shipping": [is_late_shipping],
        "order_weekday": [order_weekday],
        "is_weekend_shipping": [is_weekend_shipping],
    })

    # Encode categorical values
    mapping_mode = {"Flight": 0, "Road": 1, "Ship": 2}
    mapping_importance = {"low": 0, "medium": 1, "high": 2}

    input_data["mode_of_shipment"] = input_data["mode_of_shipment"].map(mapping_mode)
    input_data["product_importance"] = input_data["product_importance"].map(mapping_importance)

    # Predict
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]

    st.subheader("📦 Prediction Result")
    if prediction == 1:
        st.error(f"❌ Shipment is likely to be **Delayed** with {probas[1]*100:.1f}% probability.")
    else:
        st.success(f"✅ Shipment is likely to be **On Time** with {probas[0]*100:.1f}% probability.")

    # Show table
    st.markdown("#### 🔍 Input Summary")
    st.dataframe(input_data)

