"""
Feature Engineering for Mobile Money Fraud Detection
Context: East African fintech ecosystem (M-Pesa inspired)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for mobile money fraud detection
    Creates features that capture suspicious patterns in transactions
    """
    
    def __init__(self, df):
        """
        Initialize with dataframe
        
        Parameters:
        -----------
        df : pandas DataFrame
            Raw transaction data
        """
        self.df = df.copy()
        self.label_encoder = LabelEncoder()
        
    def create_time_features(self):
        """
        Create time-based features from the 'step' column
        Step represents hour of transaction (1 step = 1 hour)
        """
        # Transaction hour (1-744 for 31 days)
        self.df['transaction_hour'] = self.df['step'] % 24
        self.df['transaction_day'] = (self.df['step'] // 24) + 1
        
        # Time of day categories (numeric encoding to avoid categorical issues)
        self.df['time_category'] = pd.cut(
            self.df['transaction_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3]  # Numeric labels: 0=Night, 1=Morning, 2=Afternoon, 3=Evening
        ).astype(float)
        
        # Handle NaN values
        self.df['time_category'] = self.df['time_category'].fillna(0)
        
        print("✅ Created time features:")
        print(f"   - transaction_hour: {self.df['transaction_hour'].min()}-{self.df['transaction_hour'].max()}")
        print(f"   - transaction_day: {self.df['transaction_day'].min()}-{self.df['transaction_day'].max()}")
        
        return self
    
    def create_balance_features(self):
        """
        Create balance-related features that capture suspicious patterns
        """
        # Balance difference for origin account
        self.df['balance_diff_org'] = self.df['oldbalanceOrg'] - self.df['newbalanceOrig']
        
        # Balance difference for destination account
        self.df['balance_diff_dest'] = self.df['oldbalanceDest'] - self.df['newbalanceDest']
        
        # Amount to balance ratio (normalized)
        self.df['amount_to_balance_ratio'] = self.df['amount'] / (self.df['oldbalanceOrg'] + 1)
        
        # Flag: Zero balance after transaction
        self.df['zero_balance_after'] = (self.df['newbalanceOrig'] == 0).astype(int)
        
        # Flag: Large round number amounts (suspicious)
        self.df['is_round_amount'] = ((self.df['amount'] % 1000 == 0) & (self.df['amount'] > 0)).astype(int)
        
        # Flag: Amount exactly matches balance (draining account)
        self.df['exact_balance_drain'] = (self.df['amount'] == self.df['oldbalanceOrg']).astype(int)
        
        print("✅ Created balance features:")
        print(f"   - balance_diff_org, balance_diff_dest")
        print(f"   - amount_to_balance_ratio")
        print(f"   - zero_balance_after: {(self.df['zero_balance_after'].sum()):,} occurrences")
        print(f"   - is_round_amount: {(self.df['is_round_amount'].sum()):,} occurrences")
        print(f"   - exact_balance_drain: {(self.df['exact_balance_drain'].sum()):,} occurrences")
        
        return self
    
    def create_transaction_patterns(self):
        """
        Create features that capture transaction patterns
        """
        # Flag for large transactions (top 1%)
        amount_threshold = self.df['amount'].quantile(0.99)
        self.df['is_large_transaction'] = (self.df['amount'] > amount_threshold).astype(int)
        
        # Flag for unusual balance changes
        self.df['unusual_balance_change'] = (
            (abs(self.df['balance_diff_org']) > self.df['oldbalanceOrg'] * 0.5) &
            (self.df['oldbalanceOrg'] > 0)
        ).astype(int)
        
        # Create interaction feature: amount * is_round_amount
        self.df['round_amount_impact'] = self.df['amount'] * self.df['is_round_amount']
        
        print("✅ Created transaction pattern features:")
        print(f"   - is_large_transaction: {(self.df['is_large_transaction'].sum()):,} occurrences")
        print(f"   - unusual_balance_change: {(self.df['unusual_balance_change'].sum()):,} occurrences")
        
        return self
    
    def encode_categorical(self):
        """
        Encode categorical variables using label encoding
        """
        # Encode transaction type
        self.df['type_encoded'] = self.label_encoder.fit_transform(self.df['type'])
        
        # Map for interpretation
        type_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        print("✅ Encoded categorical variables:")
        print(f"   - Transaction type mapping: {type_mapping}")
        
        return self
    
    def select_features(self):
        """
        Select features for modeling
        """
        # List of all engineered features
        engineered_features = [
            'transaction_hour', 'transaction_day', 'time_category',
            'balance_diff_org', 'balance_diff_dest', 'amount_to_balance_ratio',
            'zero_balance_after', 'is_round_amount', 'exact_balance_drain',
            'is_large_transaction', 'unusual_balance_change', 'round_amount_impact',
            'type_encoded'
        ]
        
        # Original features to keep
        original_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest']
        
        # All features for modeling
        self.feature_columns = original_features + engineered_features
        
        # Target column
        self.target_column = 'isFraud'
        
        print("✅ Selected features for modeling:")
        print(f"   - Total features: {len(self.feature_columns)}")
        
        return self
    
    def get_features_and_target(self):
        """
        Return X (features) and y (target) for modeling
        """
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        # Convert all columns to numeric, replace non-numeric with 0
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        print(f"✅ Fraud cases in target: {y.sum():,}")
        
        return X, y
    
    def apply_smote(self, X, y):
        """
        Apply SMOTE to handle class imbalance
        SMOTE: Synthetic Minority Over-sampling Technique
        """
        print("\n" + "="*60)
        print("APPLYING SMOTE FOR CLASS IMBALANCE")
        print("="*60)
        print(f"Before SMOTE:")
        print(f"  - Non-fraud: {(y == 0).sum():,}")
        print(f"  - Fraud: {(y == 1).sum():,}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"\nAfter SMOTE:")
        print(f"  - Non-fraud: {(y_resampled == 0).sum():,}")
        print(f"  - Fraud: {(y_resampled == 1).sum():,}")
        print(f"  - New dataset size: {len(X_resampled):,} samples")
        
        return X_resampled, y_resampled
    
    def run_full_pipeline(self, apply_smote=True):
        """
        Run all feature engineering steps
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Create all features
        self.create_time_features()
        print()
        self.create_balance_features()
        print()
        self.create_transaction_patterns()
        print()
        self.encode_categorical()
        print()
        self.select_features()
        print()
        
        # Get features and target
        X, y = self.get_features_and_target()
        
        # Apply SMOTE if requested
        if apply_smote:
            X, y = self.apply_smote(X, y)
        
        return X, y
    
    def get_explanation(self, transaction):
        """
        Generate explanation for why a transaction is suspicious
        Used in the Streamlit dashboard
        
        Parameters:
        -----------
        transaction : dict or Series
            Single transaction to explain
        """
        explanations = []
        
        # Check transaction type
        if transaction['type'] in ['TRANSFER', 'CASH_OUT']:
            explanations.append("⚠️ Transaction type (TRANSFER/CASH_OUT) is commonly used for fraud")
        
        # Check amount
        if transaction['amount'] > 1_000_000:
            explanations.append("💰 Unusually large transaction amount")
        
        # Check balance patterns
        if transaction.get('exact_balance_drain', False):
            explanations.append("💸 Transaction drains entire account balance - suspicious pattern")
        
        if transaction.get('zero_balance_after', False):
            explanations.append("⚖️ Account balance becomes zero after transaction")
        
        # Check round amount
        if transaction.get('is_round_amount', False):
            explanations.append("🔢 Transaction amount is a round number - common in fraud")
        
        # Check large transaction relative to balance
        if transaction.get('amount_to_balance_ratio', 0) > 0.8:
            explanations.append("📊 Transaction uses >80% of available balance")
        
        # Time-based suspicion
        hour = transaction.get('transaction_hour', 0)
        if hour < 4 or hour > 23:
            explanations.append("🌙 Transaction occurs during unusual hours (late night)")
        
        if not explanations:
            explanations.append("✅ No obvious fraud patterns detected")
        
        return explanations


# Example usage
if __name__ == "__main__":
    # Load the data
    print("Loading dataset...")
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Run the full pipeline
    X, y = fe.run_full_pipeline(apply_smote=True)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"Final feature matrix: {X.shape}")
    print(f"Features: {X.columns.tolist()}")