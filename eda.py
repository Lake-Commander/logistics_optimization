# Phase 2: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import missingno as msno

# Set styles for plots
sns.set(style='whitegrid')

# === 1. Load Data ===
df = pd.read_csv("logistics.csv")

# === 2. Create output folder for plots ===
os.makedirs("eda/plots", exist_ok=True)

# === 3. Check Target Distribution ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title("Delivery Status Distribution")
plt.xlabel("Label (-1 = Early, 0 = On-time, 1 = Late)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda/plots/delivery_status_distribution.png")
plt.close()

# === 4. Shipping Mode vs Delivery Status ===
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='shipping_mode', hue='label')
plt.title("Shipping Mode vs Delivery Status")
plt.xlabel("Shipping Mode")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda/plots/shipping_mode_vs_label.png")
plt.close()

# === 5. Top 10 Order Regions vs Delivery Status ===
top_regions = df['order_region'].value_counts().nlargest(10).index
subset = df[df['order_region'].isin(top_regions)]

plt.figure(figsize=(10, 6))
sns.countplot(data=subset, x='order_region', hue='label')
plt.title("Top 10 Order Regions vs Delivery Status")
plt.xlabel("Order Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda/plots/top10_regions_vs_label.png")
plt.close()

# === 6. Discount Rate vs Delivery Outcome ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='label', y='order_item_discount_rate')
plt.title("Discount Rate vs Delivery Outcome")
plt.xlabel("Delivery Status")
plt.ylabel("Discount Rate")
plt.tight_layout()
plt.savefig("eda/plots/discount_vs_label.png")
plt.close()

# === 7. Missing Values Matrix ===
msno.matrix(df)
plt.title("Missing Values Matrix")
plt.tight_layout()
plt.savefig("eda/plots/missing_values_matrix.png")
plt.close()

print("✅ EDA completed. All plots saved to 'eda/plots'")
