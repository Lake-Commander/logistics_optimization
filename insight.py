# phase5_insights_recommendations.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load cleaned and engineered data
df = pd.read_csv("logistics_featurized.csv")

# Load trained model (e.g., Random Forest)
model = joblib.load("models/best_model.pkl")

# Ensure output directory
os.makedirs("output/phase5", exist_ok=True)

# ========================
# 1. Feature Importance
# ========================
def plot_feature_importance(model, feature_names):
    importances = model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("output/phase5/feature_importance.png")
    plt.close()
    return importance_df

# Plot and save
preprocessor = model.named_steps['preprocessor']
# Get feature names after preprocessing
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
num_features = preprocessor.transformers_[0][2]
feature_names = np.concatenate([num_features, cat_features])
importance_df = plot_feature_importance(model, feature_names)

# ========================
# 2. Key Insights
# ========================
insights = []

top_features = importance_df.head(5)['Feature'].tolist()

insights.append(f"Top predictive features include: {', '.join(top_features)}.")
if 'shipping_duration' in top_features:
    insights.append("Shipping duration plays a significant role in predicting delays.")
if 'product_category_id' in top_features:
    insights.append("Product category is influential, suggesting some categories are more delay-prone.")
if 'customer_country' in top_features:
    insights.append("Customer location also contributes heavily, indicating geographic logistics challenges.")

# ========================
# 3. Recommendations
# ========================
recommendations = [
    "✅ Optimize operations for product categories that are frequently delayed.",
    "✅ Investigate and improve shipping processes in high-delay regions.",
    "✅ Use predicted delay probabilities to notify customers ahead of time.",
    "✅ Prioritize orders with short delivery windows to improve reliability ratings.",
    "✅ Consider reinforcing customer support in delay-prone areas to manage expectations."
]

# Save to file
with open("output/phase5/insights_and_recommendations.txt", "w", encoding="utf-8") as f:
    f.write("=== KEY INSIGHTS ===\n")
    for insight in insights:
        f.write(f"- {insight}\n")
    f.write("\n=== RECOMMENDATIONS ===\n")
    for rec in recommendations:
        f.write(f"- {rec}\n")

print("✅ Phase 5 complete: Feature importances, insights, and recommendations saved.")
