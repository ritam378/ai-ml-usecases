# Problem Statement: Credit Card Fraud Detection

## Business Context

Financial institutions lose billions of dollars annually to credit card fraud. For every legitimate transaction, fraud detection systems must make split-second decisions to either approve or flag the transaction. The challenge is balancing two competing objectives:

1. **Catch fraudulent transactions** - Minimize financial losses
2. **Avoid false positives** - Don't block legitimate customers (causes frustration and lost revenue)

## The Core Challenge: Class Imbalance

This is the **key learning point** for this use case:

In real-world fraud detection, fraudulent transactions typically represent **less than 1-5%** of all transactions. This creates a severely imbalanced dataset where:

- **Majority class (legitimate)**: 95-99% of transactions
- **Minority class (fraud)**: 1-5% of transactions

### Why Class Imbalance Matters

A naive model could achieve **95%+ accuracy** by simply predicting "legitimate" for every transaction. But this would:
- Catch **0% of fraud** (complete failure)
- Cost the company millions in undetected fraud

**Key Interview Insight**: Accuracy is a misleading metric for imbalanced data. We need different evaluation approaches.

## Business Requirements

### Primary Goal
Detect as many fraudulent transactions as possible while maintaining acceptable false positive rates.

### Success Metrics
1. **High Recall (Sensitivity)**: Catch 80%+ of actual fraud
2. **Acceptable Precision**: When we flag fraud, be right at least 20-30% of the time
3. **Low False Positive Rate**: Don't block more than 1-2% of legitimate transactions
4. **Real-time Performance**: Make decisions in under 100ms

### Cost Matrix
- **False Negative (missed fraud)**: Lose $50-500 per transaction
- **False Positive (blocked legitimate)**: Customer frustration, lost revenue, potential customer churn
- **True Positive (caught fraud)**: Save $50-500 per transaction
- **True Negative (approved legitimate)**: Normal business operation

## Dataset Characteristics

### Features (Typical)
For learning purposes, our simplified dataset includes:

1. **Transaction Amount**: Dollar value of transaction
2. **Time**: Seconds elapsed from first transaction (detects unusual timing)
3. **Anonymized Features (V1-V28)**: PCA-transformed features (mimics real-world privacy protection)
   - In real systems, these might represent:
     - Geographic distance from previous transaction
     - Merchant category
     - Transaction velocity (transactions per hour)
     - Device fingerprinting

### Target Variable
- `0` = Legitimate transaction (majority class)
- `1` = Fraudulent transaction (minority class)

## Interview-Relevant Scenarios

This use case prepares you to discuss:

1. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Class weights
   - Anomaly detection approaches

2. **Metric Selection**
   - Why accuracy fails
   - Precision vs Recall trade-offs
   - F1 Score, PR-AUC, ROC-AUC

3. **Model Choice**
   - Why tree-based models (Random Forest, XGBoost) work well
   - Interpretability requirements (explain why transaction was flagged)
   - Real-time inference constraints

4. **System Design**
   - Online vs batch prediction
   - Threshold tuning for business requirements
   - A/B testing fraud models
   - Feedback loops and model retraining

## Learning Objectives

By completing this case study, you will understand:

1. How to recognize and handle severely imbalanced datasets
2. Why traditional metrics fail and what to use instead
3. Techniques to oversample/undersample for better model training
4. How to tune decision thresholds based on business costs
5. How to evaluate models using precision-recall curves
6. Common interview questions about fraud detection systems

## Simplifications for Learning

To focus on core concepts, our implementation simplifies:
- **Dataset size**: 1,000 transactions (vs millions in production)
- **Features**: Limited set of features (vs hundreds in production)
- **Real-time requirements**: Offline model (vs sub-100ms inference)
- **Model complexity**: Single model (vs ensemble of models)
- **Data drift**: Static dataset (vs continuous retraining)

The principles learned here apply directly to production systems at scale.
