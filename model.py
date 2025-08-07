import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('logistics_featurized.csv')

# Drop unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Check if 'label' column exists
if 'label' not in df.columns:
    raise ValueError("Missing 'label' column in dataset")

X = df.drop(columns=['label'])
y = df['label']

# Print full dataset label distribution
print("Full dataset label distribution:\n", y.value_counts())

# Stratified train-test split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print training label distribution
print("Training label distribution:\n", y_train.value_counts())

# Ensure we have at least two classes in y_train
if len(y_train.unique()) < 2:
    raise ValueError(f"y_train has only one class: {y_train.unique()[0]}. Cannot train classifiers.")

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle column with all missing values (e.g., 'delivery_delay_days')
for col in numerical_cols:
    if X[col].isna().all():
        print(f"Dropping column with all missing values: {col}")
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
        numerical_cols.remove(col)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Remap labels for XGBoost
    if name == 'XGBoost':
        y_train_xgb = y_train.replace({-1: 0, 0: 1, 1: 2})
        y_test_xgb = y_test.replace({-1: 0, 0: 1, 1: 2})
        pipeline.fit(X_train, y_train_xgb)
        y_pred = pipeline.predict(X_test)
        # Map predictions back for reporting
        y_pred_report = pd.Series(y_pred).replace({0: -1, 1: 0, 2: 1})
        print(f"\n{name} Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred_report))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_report))
        print("Classification Report:\n", classification_report(y_test, y_pred_report))
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n{name} Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Hyperparameter tuning for best model (Random Forest shown)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

print("\nTuning Random Forest with GridSearchCV...")

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Final evaluation
final_pred = best_model.predict(X_test)
print("\nTuned Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))
print("Classification Report:\n", classification_report(y_test, final_pred))

# Save final model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print("\nModel saved to models/best_model.pkl")
