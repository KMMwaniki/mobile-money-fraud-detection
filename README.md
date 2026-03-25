# Mobile Money Fraud Detection System

## Project Overview
A machine learning system to detect fraudulent mobile money transactions in East Africa's M-Pesa ecosystem.

## Key Results
- **99.99% Precision** - Only 118 false positives
- **99.98% Recall** - Catches 99.98% of fraud
- **XGBoost** achieved perfect ROC-AUC of 1.0000

## Dataset
- 6.36 million transactions
- 0.13% fraud rate (8,213 fraud cases)
- PaySim synthetic mobile money data

## Key Findings
- Fraud occurs **only in CASH_OUT and TRANSFER** transactions
- **Exact balance drain** is the strongest predictor (97.7% importance)
- Rule-based systems miss sophisticated fraud patterns

## Model Performance
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | 98.77% | 99.22% | 98.99% |
| Random Forest | 99.99% | 99.93% | 99.96% |
| **XGBoost** | **99.99%** | **99.98%** | **99.99%** |

## Installation
```bash
pip install -r requirements.txt
python run_model.py
streamlit run app.py