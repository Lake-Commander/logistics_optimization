import pandas as pd

# === Load original and cleaned datasets ===
original_df = pd.read_csv('logistics.csv')
cleaned_df = pd.read_csv('logistics_cleaned.csv')

# === Normalize column names for consistency ===
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

original_df = clean_column_names(original_df)
cleaned_df = clean_column_names(cleaned_df)

# === Compare original vs cleaned dataset columns ===
def compare_columns(df1, df2):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    removed = cols1 - cols2
    added = cols2 - cols1
    print("Removed columns:", removed)
    print("Added columns:", added)

compare_columns(original_df, cleaned_df)

# === Proceed with feature engineering ===
df = cleaned_df.copy()

# Ensure datetime columns exist
date_cols = ['scheduled_date', 'actual_delivery_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in dataset.")

# Feature: delivery_delay_days
if 'scheduled_date' in df.columns and 'actual_delivery_date' in df.columns:
    df['delivery_delay_days'] = (df['actual_delivery_date'] - df['scheduled_date']).dt.days
else:
    df['delivery_delay_days'] = None

# Feature: is_delayed
df['is_delayed'] = df['delivery_delay_days'].apply(lambda x: 1 if x is not None and x > 0 else 0)

# Feature: scheduled_weekday
if 'scheduled_date' in df.columns:
    df['scheduled_weekday'] = df['scheduled_date'].dt.day_name()

# Feature: is_weekend_delivery
if 'actual_delivery_date' in df.columns:
    df['is_weekend_delivery'] = df['actual_delivery_date'].dt.weekday >= 5  # Saturday/Sunday = True

# === Save engineered features ===
df.to_csv('logistics_featurized.csv', index=False)
print("Feature engineering complete. Output saved as 'logistics_featurized.csv'")
