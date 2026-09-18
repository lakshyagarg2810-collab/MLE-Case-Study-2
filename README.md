# Credit Card Fraud Detection

## Overview

This project uses machine learning to detect fraudulent credit card transactions.

The model predicts whether a transaction is **Normal** or **Fraudulent** using transaction and customer-related features.

## Objective

To develop a machine learning classification model that can identify fraudulent transactions while handling the imbalance between normal and fraudulent transactions.

## Dataset

The project uses a credit card transaction dataset stored as:

`credit_card_transactions.csv`

The dataset contains information such as:

- Transaction ID
- Transaction Amount
- Transaction Hour
- Customer Age
- Account Age in Days
- Number of Previous Transactions
- Distance From Home
- Fraud Status (`IsFraud`)

The dataset itself is not included in this repository.

## Features Used

The following features are used for prediction:

- `TransactionAmount`
- `TransactionHour`
- `CustomerAge`
- `AccountAgeDays`
- `NumPrevTransactions`
- `DistanceFromHomeKM`

The target variable is:

`IsFraud`

where:

- `0` = Normal transaction
- `1` = Fraudulent transaction

## Machine Learning Approach

The project follows these steps:

1. Load and explore the transaction dataset.
2. Analyze the distribution of normal and fraudulent transactions.
3. Select relevant features.
4. Split the data into training and testing sets.
5. Handle class imbalance using **SMOTE**.
6. Train an **XGBoost Classifier**.
7. Evaluate the model using classification metrics and ROC-AUC.
8. Test different probability thresholds.
9. Use a tuned threshold of `0.25` for fraud prediction.
10. Analyze feature importance.
11. Generate a CSV file containing transaction IDs, fraud probabilities, and predictions.

## Handling Class Imbalance

Fraudulent transactions are generally much fewer than normal transactions.

To address this class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data.

SMOTE generates synthetic samples for the minority class so that the model receives more balanced training data.

## Model

The project uses **XGBoost Classifier**.

Model configuration:

- `n_estimators = 100`
- `max_depth = 4`
- `learning_rate = 0.1`
- Evaluation metric: `logloss`
- `random_state = 42`

## Evaluation Metrics

The model is evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC

The project also compares model performance at different probability thresholds.

### Threshold Tuning

Instead of relying only on the default probability threshold of `0.5`, thresholds from `0.10` to `0.90` are evaluated.

The notebook uses:

`0.25`

as the tuned threshold for the final fraud predictions.

## Feature Importance

XGBoost feature importance is calculated to understand which input features contributed most to the model's predictions.

A feature importance plot is also generated using Matplotlib.

## Output

The project generates a `submission.csv` file containing:

- `TransactionID`
- `FraudProbability`
- `PredictedIsFraud`

Example structure:

| TransactionID | FraudProbability | PredictedIsFraud |
|---|---:|---:|
| 10001 | 0.8421 | 1 |
| 10002 | 0.0632 | 0 |
| 10003 | 0.7315 | 1 |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Google Colab
- Jupyter Notebook

## Project Files

- `Case Study 2.ipynb` — Complete data analysis, preprocessing, model training, evaluation, threshold tuning, and prediction
- `submission.csv` — Final transaction predictions and fraud probabilities
- `README.md` — Project documentation

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install the required libraries if needed.
3. Run the notebook.
4. When prompted, upload:

`credit_card_transactions.csv`

5. Execute the cells sequentially.
6. The notebook will train the XGBoost model and generate `submission.csv`.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for credit card fraud detection, including class imbalance handling with SMOTE, XGBoost classification, probability threshold tuning, model evaluation, and feature importance analysis.
