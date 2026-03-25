import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from features import FeatureEngineer
from model import FraudDetector

print("Loading data...")
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

print("Feature engineering...")
fe = FeatureEngineer(df)
X, y = fe.run_full_pipeline(apply_smote=True)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost...")
detector = FraudDetector(X_train, X_test, y_train, y_test)
detector.train_xgboost()

print("Saving model...")
joblib.dump(detector.results['XGBoost']['model'], 'fraud_model.pkl')
print("✅ Model saved as fraud_model.pkl")