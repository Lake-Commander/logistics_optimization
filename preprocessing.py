# Phase 3: Data Preprocessing for SwiftChain Delay Prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv('logistics.csv')

# ================================
# Step 1: Handle Missing Values
# ================================

# Check for missing values
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Example strategy: fill missing shipping_date with forward fill
if 'shipping_date' in df.columns:
    df['shipping_date'] = pd.to_datetime(df['shipping_date'], errors='coerce')
    df['shipping_date'].fillna(method='ffill', inplace=True)

# Optional: Drop columns with too many missing values (e.g., >40%)
threshold = 0.4
to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
df.drop(columns=to_drop, inplace=True)
print(f"\nDropped columns with >{int(threshold*100)}% missing values: {to_drop}")

# ================================
# Step 2: Encode Categorical Variables
# ================================

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Label encode simple categorical variables
le = LabelEncoder()
for col in cat_cols:
    if df[col].nunique() <= 20:  # threshold for label encoding
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        df[col] = df[col].astype(str)

# One-hot encode high-cardinality features later if needed

# ================================
# Step 3: Handle Outliers (Optional)
# ================================

# Example: Clip extreme outliers in numeric columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [col for col in num_cols if col != 'label']

for col in num_cols:
    q_low = df[col].quantile(0.01)
    q_hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=q_low, upper=q_hi)

# ================================
# Step 4: Scale Numerical Features
# ================================

scaler = StandardScaler()

# Features to scale
scale_cols = ['profit_per_order', 'sales_per_customer', 'order_item_discount',
              'order_item_discount_rate', 'order_item_product_price',
              'order_item_quantity', 'sales', 'order_item_total_amount',
              'order_profit_per_order', 'product_price']

for col in scale_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# ================================
# Step 5: Save Cleaned Dataset
# ================================

# Save to new file
df.to_csv("logistics_cleaned.csv", index=False)
print("\nPreprocessing complete. Cleaned dataset saved as 'logistics_cleaned.csv'")
