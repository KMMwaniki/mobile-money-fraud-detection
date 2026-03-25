"""
Fraud Detection Models for Mobile Money Transactions
Trains and evaluates multiple models for fraud detection
Context: East African fintech ecosystem (M-Pesa inspired)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, precision_score, recall_score,
                             f1_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineer
from features import FeatureEngineer

class FraudDetector:
    """
    Fraud detection model trainer for mobile money transactions
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with train/test data
        
        Parameters:
        -----------
        X_train, X_test : Features for training and testing
        y_train, y_test : Targets for training and testing
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self):
        """
        Train baseline Logistic Regression model
        """
        print("\n" + "="*60)
        print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
        print("="*60)
        
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = lr
        self._evaluate_model('Logistic Regression', lr)
        
        return lr
    
    def train_random_forest(self):
        """
        Train Random Forest model
        """
        print("\n" + "="*60)
        print("MODEL 2: RANDOM FOREST")
        print("="*60)
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = rf
        self._evaluate_model('Random Forest', rf)
        
        return rf
    
    def train_xgboost(self):
        """
        Train XGBoost model (best for imbalanced data)
        """
        print("\n" + "="*60)
        print("MODEL 3: XGBOOST")
        print("="*60)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1]),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = xgb_model
        self._evaluate_model('XGBoost', xgb_model)
        
        return xgb_model
    
    def _evaluate_model(self, name, model):
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        # Store results
        self.results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print metrics
        print(f"\nResults for {name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]:,}")
        print(f"  False Positives: {cm[0,1]:,}")
        print(f"  False Negatives: {cm[1,0]:,}")
        print(f"  True Positives: {cm[1,1]:,}")
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, model_name):
        """
        Plot confusion matrix for a specific model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found!")
            return
        
        y_pred = self.results[model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self):
        """
        Plot Precision-Recall curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            plt.plot(recall, precision, label=f'{name} (AP = {result["avg_precision"]:.4f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name='XGBoost'):
        """
        Plot feature importance for tree-based models
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found!")
            return
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.show()
            
            # Print top 10 features
            print(f"\nTop 10 Most Important Features ({model_name}):")
            for i in range(min(10, len(indices))):
                print(f"  {i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
        else:
            print(f"Model {model_name} does not have feature importance")
    
    def get_best_model(self):
        """
        Select best model based on F1-score (balanced precision and recall)
        """
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        print("\n" + "="*60)
        print("BEST MODEL SELECTION")
        print("="*60)
        print(f"🏆 Best Model: {best_model[0]}")
        print(f"   F1-Score: {best_model[1]['f1']:.4f}")
        print(f"   Precision: {best_model[1]['precision']:.4f}")
        print(f"   Recall: {best_model[1]['recall']:.4f}")
        print(f"   ROC-AUC: {best_model[1]['roc_auc']:.4f}")
        
        return best_model
    
    def explain_why_precision_recall_matters(self):
        """
        Explanation for README about why precision-recall is better than accuracy
        """
        print("\n" + "="*60)
        print("WHY PRECISION-RECALL MATTERS MORE THAN ACCURACY")
        print("="*60)
        
        # Calculate accuracy for best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1'])[0]
        best_result = self.results[best_model_name]
        accuracy = (best_result['y_pred'] == self.y_test).mean()
        
        print(f"""
In our dataset:
- Fraud transactions: {self.y_test.sum():,} out of {len(self.y_test):,} ({self.y_test.mean()*100:.4f}%)
- Non-fraud transactions: {(self.y_test == 0).sum():,} ({100 - self.y_test.mean()*100:.4f}%)

If we used a dummy model that always predicts "Non-Fraud":
- Accuracy would be {100 - self.y_test.mean()*100:.2f}% (looks good!)
- But it would catch ZERO fraud transactions! ❌

This is why we use Precision and Recall instead:

📊 PRECISION: Of transactions flagged as fraud, how many are actually fraud?
   - High precision = Fewer false alarms
   - Formula: True Positives / (True Positives + False Positives)

📊 RECALL: Of actual fraud transactions, how many did we catch?
   - High recall = Catching more fraud
   - Formula: True Positives / (True Positives + False Negatives)

🎯 F1-SCORE: Harmonic mean of Precision and Recall
   - Balances both metrics
   - Our best model ({best_model_name}) achieved F1 = {best_result['f1']:.4f}
   - While accuracy is {accuracy:.4f}, F1 tells the real story!

For fraud detection, catching fraud (Recall) and minimizing false alarms (Precision)
are equally important - this is why we optimize for F1-Score! 🚀
""")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("MOBILE MONEY FRAUD DETECTION - MODEL BUILDING")
    print("East African Fintech Context")
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
    
    # Train-test split
    print("\n3. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    print(f"   Fraud in test: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    
    # Initialize detector
    detector = FraudDetector(X_train, X_test, y_train, y_test)
    
    # Train models
    print("\n4. Training models...")
    detector.train_logistic_regression()
    detector.train_random_forest()
    detector.train_xgboost()
    
    # Visualizations
    print("\n5. Generating visualizations...")
    
    print("\n   - Confusion Matrices:")
    for model_name in detector.results.keys():
        detector.plot_confusion_matrix(model_name)
    
    print("\n   - ROC Curves:")
    detector.plot_roc_curves()
    
    print("\n   - Precision-Recall Curves:")
    detector.plot_precision_recall_curves()
    
    print("\n   - Feature Importance (XGBoost):")
    detector.plot_feature_importance('XGBoost')
    
    # Best model selection
    detector.get_best_model()
    
    # Explanation
    detector.explain_why_precision_recall_matters()
    
    print("\n" + "="*60)
    print("MODEL BUILDING COMPLETE!")
    print("="*60)