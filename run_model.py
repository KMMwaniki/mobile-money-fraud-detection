"""
Run model training with smaller model size for deployment
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from features import FeatureEngineer
import os

print("="*60)
print("MOBILE MONEY FRAUD DETECTION - TRAINING SMALL MODEL")
print("="*60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print(f"   Dataset loaded: {df.shape[0]:,} transactions")

# Feature engineering
print("\n2. Feature engineering...")
fe = FeatureEngineer(df)
X, y = fe.run_full_pipeline(apply_smote=True)
print(f"   Features created: {X.shape[1]}")
print(f"   Dataset size after SMOTE: {X.shape[0]:,}")

# Take 20% sample for smaller model
print("\n3. Sampling 20% of data for smaller model...")
X_sample = X.sample(frac=0.2, random_state=42)
y_sample = y.loc[X_sample.index]
print(f"   Reduced dataset: {X_sample.shape[0]:,} samples")

# Train-test split
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Testing: {X_test.shape[0]:,} samples")

# Train smaller XGBoost model
print("\n5. Training smaller XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model
print("\n6. Saving model...")
joblib.dump(xgb_model, 'fraud_model.pkl')
print("   ✅ Model saved as fraud_model.pkl (smaller version)")

# Get file size
size_mb = os.path.getsize('fraud_model.pkl') / (1024 * 1024)
print(f"   File size: {size_mb:.2f} MB")

print("\n✅ Small model training complete!")