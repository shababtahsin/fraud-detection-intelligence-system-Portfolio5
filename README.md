
# 🛡️ Credit Card Fraud Detection
### Machine Learning · Deep Learning · NLP · Streamlit

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.98-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/App-Streamlit-orange)]()
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## Project Overview

This project looks at credit card fraud detection using a dataset of **284,807 transactions**.

Only **492 transactions (0.17%)** are fraudulent, which makes this an extreme class-imbalance problem.

That means accuracy by itself is not very useful. A model could classify almost every transaction as legitimate and still report very high accuracy while failing to identify actual fraud.

The project therefore focuses on:

- fraud recall
- precision
- ROC-AUC
- class imbalance
- model comparison
- sentiment analysis
- practical model deployment

Several machine-learning and deep-learning approaches were tested, with **XGBoost producing the strongest overall results**.

A Streamlit application was also built so the trained models can be tested through a simple user interface.

---

## Business Problem

Fraud detection involves a trade-off.

If the model misses fraudulent transactions, money can be lost.

If it flags too many legitimate transactions, customers may be inconvenienced and investigators may waste time reviewing false alarms.

The main question was:

> Can a model identify a useful proportion of fraudulent transactions without creating an excessive number of false positives?

The dataset makes this difficult because fraud represents only **0.17% of all transactions**.

---

## Dataset

The main dataset is the public **Credit Card Fraud Detection** dataset from Kaggle.

| Metric | Value |
|---|---:|
| Transactions | 284,807 |
| Fraudulent Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Fraud Rate | 0.17% |
| Features | 30 predictors + target |

Most transaction features are anonymised PCA components named `V1` through `V28`.

The dataset also contains:

- `Time`
- `Amount`
- `Class`

`Class = 1` represents fraud and `Class = 0` represents a legitimate transaction.

---

## Why Accuracy Is Misleading

Because legitimate transactions make up more than 99% of the dataset, a model could predict almost everything as legitimate and still appear highly accurate.

For this reason, I focused more heavily on:

- **Precision** — how many transactions flagged as fraud were actually fraudulent
- **Recall** — how much of the actual fraud the model found
- **F1 Score** — balance between precision and recall
- **ROC-AUC** — how well the model separates the two classes

---

## Models Tested

The project compares several approaches:

- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- XGBoost
- Multi-Layer Perceptron neural network
- models with an additional sentiment feature

### Model Results

| Model | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.94 | 0.84 | 0.66 | 0.74 |
| KNN | 0.91 | 0.90 | 0.74 | 0.81 |
| Decision Tree | 0.87 | 0.72 | 0.72 | 0.72 |
| MLP Baseline | 0.97 | 0.85 | 0.78 | 0.81 |
| MLP + Sentiment | 0.97 | 0.86 | 0.79 | 0.82 |
| **XGBoost** | **0.98** | **0.90** | **0.82** | **0.86** |
| **XGBoost + Sentiment** | **0.98** | **0.91** | **0.82** | **0.86** |

The best overall model was **XGBoost**, with:

- **0.98 ROC-AUC**
- approximately **90% precision**
- approximately **82% recall**
- **0.86 F1 score**

XGBoost was therefore used as one of the main models in the Streamlit application.

---

## Project Workflow

### 1. Data Exploration

The first stage was used to understand the dataset before modelling.

This included:

- checking dimensions and data types
- checking for missing values
- reviewing the fraud / legitimate class split
- looking at transaction amounts
- looking at transaction timing
- comparing distributions between fraudulent and legitimate transactions

The main issue identified immediately was the extreme class imbalance.

---

### 2. Data Preparation

The data was prepared before model training.

This included:

- separating features and target
- creating training and test sets
- scaling transaction amount where required
- using stratified sampling to preserve the fraud proportion
- handling class imbalance during model development

**SMOTE** was also tested as part of the imbalance-handling process.

The important point was to avoid judging models on accuracy alone and instead compare their ability to actually find fraudulent cases.

---

### 3. Classical Machine Learning

Four main machine-learning approaches were compared:

- Logistic Regression
- KNN
- Decision Tree
- XGBoost

Each model has different strengths.

Logistic Regression gives a useful interpretable baseline, while XGBoost can capture more complex relationships between the transaction variables.

The models were compared using the same evaluation metrics so that the trade-offs were easier to see.

---

### 4. Deep Learning

A **Multi-Layer Perceptron (MLP)** was also trained on the transaction data.

The MLP achieved competitive results, reaching approximately:

- **0.97 ROC-AUC**
- **0.78 recall**
- **0.81 F1**

The neural network performed well, but XGBoost still produced the stronger overall balance of precision and recall for this dataset.

This was a useful reminder that a more complicated model is not automatically the best model.

---

### 5. Sentiment Experiment

The project also includes an NLP and sentiment-analysis experiment.

Customer complaint text was processed to explore whether an additional text-based risk signal could improve fraud classification.

The sentiment work included techniques such as:

- text cleaning
- token processing
- TF-IDF
- sentiment classification

Models were then tested with an additional sentiment feature.

The improvement was small.

For example:

| Model | Without Sentiment | With Sentiment |
|---|---:|---:|
| XGBoost Precision | 0.90 | 0.91 |
| XGBoost Recall | 0.82 | 0.82 |
| MLP F1 | 0.81 | 0.82 |

This result is important because the complaint data does **not** have a genuine transaction-level join key.

The mapping used for the experiment was simulated, so the sentiment results should be treated as a demonstration of how text features could be added to a fraud workflow rather than evidence that sentiment materially improves this particular fraud dataset.

---

## Streamlit Fraud Detector

The repository includes a working Streamlit application:

`streamlit_app.py`

The app loads the saved:

- Logistic Regression model
- XGBoost model
- preprocessing scaler

and allows a user to enter transaction features directly.

It also includes sample buttons for loading:

- a legitimate transaction
- a fraudulent transaction

The application then returns fraud probabilities from both models.

### Run the App

```bash
streamlit run streamlit_app.py
```

The interface is deliberately simple.

Its purpose is to show how a trained model can be moved out of a notebook and placed behind an interface that another user can interact with.

---

## Saved Models

The repository contains trained model files so the models do not need to be retrained every time the application is opened.

```text
models/
├── decision_tree.joblib
├── deep_learning_model.keras
├── logistic_regression.joblib
├── mlp_baseline.keras
├── mlp_with_sentiment.keras
├── sentiment_logreg.joblib
├── tfidf_sentiment.joblib
├── xgboost.joblib
└── xgboost_with_sentiment.joblib
```

This also separates model training from model inference.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data Analysis | Python · pandas · NumPy |
| Machine Learning | Scikit-learn · XGBoost · imbalanced-learn |
| Deep Learning | TensorFlow · Keras |
| NLP | NLTK · spaCy · TF-IDF · TextBlob |
| Visualisation | matplotlib · seaborn |
| Application | Streamlit |
| Model Storage | Joblib · Keras |

---

## Repository Structure

```text
fraud-detection-intelligence-system-Portfolio5/
│
├── Capstone_Project_Fixed FINAL.ipynb
│   └── Main analysis and modelling notebook
│
├── Capstone_Project_Report.docx
│   └── Written project report
│
├── models/
│   ├── decision_tree.joblib
│   ├── deep_learning_model.keras
│   ├── logistic_regression.joblib
│   ├── mlp_baseline.keras
│   ├── mlp_with_sentiment.keras
│   ├── sentiment_logreg.joblib
│   ├── tfidf_sentiment.joblib
│   ├── xgboost.joblib
│   └── xgboost_with_sentiment.joblib
│
├── streamlit_app.py
├── scaler_amount.pkl
├── scaler_amount_ws.pkl
├── requirements.txt.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Data Availability

The main `creditcard.csv` dataset is approximately **150 MB** and is not included in the repository.

It can be downloaded from the Kaggle Credit Card Fraud Detection dataset.

After downloading it, place it in the project working directory before running the full notebook.

---

## What I Learned From This Project

The biggest lesson from the project was that fraud detection is not mainly an accuracy problem.

With a dataset this imbalanced, the more important questions are:

- How much fraud is being detected?
- How many false alarms are being created?
- What happens when the decision threshold changes?
- Is the more complicated model actually better?
- Can the model be used outside the notebook?

XGBoost performed better than the other models overall, while the sentiment experiment produced only a small improvement.

That difference was useful because it showed why new features and more complicated modelling should be tested rather than assumed to improve performance.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/shababtahsin/fraud-detection-intelligence-system-Portfolio5.git

cd fraud-detection-intelligence-system-Portfolio5
```

Install the required packages:

```bash
pip install -r requirements.txt.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

For the full modelling process, open:

```text
Capstone_Project_Fixed FINAL.ipynb
```

---

## Author

**Shah Tahsin**  
Business Data Analyst | SQL · Python · Power BI

[GitHub](https://github.com/shababtahsin)
