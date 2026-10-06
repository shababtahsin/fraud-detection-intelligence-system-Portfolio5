# 🛡️ Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fraud%20Detection-brightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-F1%200.83-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Inference%20Demo-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Project Overview

This project explores credit-card fraud detection using a highly imbalanced dataset containing **284,807 transactions**.

Only **492 transactions — approximately 0.17% — are fraudulent**.

Because of that imbalance, overall accuracy is not a useful model-selection metric on its own. A model could predict almost every transaction as legitimate and still appear extremely accurate while failing to detect fraud.

The project therefore focuses on:

- 🎯 Precision
- 🔎 Recall
- ⚖️ F1 Score
- 📈 ROC-AUC
- ⚠️ Class imbalance
- 🤖 Classical machine learning
- 🧠 Deep learning
- 💬 NLP and sentiment experimentation
- 🖥️ Model inference through Streamlit

The main modelling question was:

> **Can fraudulent transactions be identified while maintaining a useful balance between missed fraud and false alarms?**

---

# 🎯 Business Problem

Fraud detection involves two competing costs.

### False Negative

A fraudulent transaction is classified as legitimate.

Possible impact:

- financial loss
- delayed investigation
- customer harm

### False Positive

A legitimate transaction is classified as fraud.

Possible impact:

- customer inconvenience
- unnecessary investigation
- operational workload

The objective is therefore not simply to maximise accuracy.

The more useful question is:

> **How much fraud can the model detect while keeping false-positive behaviour manageable?**

---

# 📊 Dataset

The main structured dataset is the public **Credit Card Fraud Detection** dataset.

| Metric | Value |
|---|---:|
| Transactions | **284,807** |
| Fraudulent Transactions | **492** |
| Legitimate Transactions | **284,315** |
| Fraud Rate | **0.17%** |
| Raw Columns | **31** |
| Target | `Class` |

### Raw Variables

The dataset contains:

```text
Time
V1 – V28
Amount
Class
```

`V1` through `V28` are anonymised PCA-transformed variables.

`Class` is the prediction target:

```text
0 = Legitimate
1 = Fraud
```

---

## Modelling Features

The `Time` variable was excluded from the main fraud classifiers.

The structured fraud models therefore use:

```text
V1 – V28
Amount
```

for a total of **29 predictive features**.

---

# ⚠️ Extreme Class Imbalance

The dataset contains:

```text
Fraud:       492
Legitimate:  284,315
```

This means legitimate transactions represent more than **99.8% of the data**.

A classifier that predicted every transaction as legitimate would therefore achieve very high accuracy while providing almost no fraud-detection value.

For that reason, model comparison focused on:

| Metric | Meaning |
|---|---|
| **Precision** | When the model flags fraud, how often is it correct? |
| **Recall** | How much of the actual fraud does the model detect? |
| **F1 Score** | Balance between precision and recall |
| **ROC-AUC** | How well the model ranks fraud above legitimate transactions across thresholds |

---

# 🧪 Train/Test Strategy

The structured dataset was split using:

```python
train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.20,
    random_state=42
)
```

This produced approximately:

```text
Training Set: 227,845 transactions
Test Set:      56,962 transactions
```

Fraud prevalence was preserved through stratification:

```text
Training Fraud Rate ≈ 0.173%
Test Fraud Rate     ≈ 0.172%
```

The test set remained separate from model training and was used for final evaluation.

---

# 🤖 Models Tested

The project compares several modelling approaches:

### Classical Machine Learning

- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- XGBoost

### Deep Learning

- Multi-Layer Perceptron

### Experimental Extensions

- XGBoost with simulated sentiment feature
- MLP with simulated sentiment feature

---

# 📈 Classical Model Results

The following results come from the executed test-set evaluation in the notebook.

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.9563** | 0.8205 | 0.6531 | 0.7273 |
| KNN | 0.9130 | **0.9211** | 0.7143 | 0.8046 |
| Decision Tree | 0.8773 | 0.7551 | 0.7551 | 0.7551 |
| **XGBoost** | 0.9268 | 0.8764 | **0.7959** | **0.8342** |

---

## 🏆 Model Comparison

No single model won every metric.

### Logistic Regression

Produced the highest ROC-AUC:

**0.9563**

This means it performed strongly at ranking fraudulent transactions above legitimate ones across different decision thresholds.

---

### KNN

Produced the highest precision:

**0.9211**

When KNN classified a transaction as fraud, it was correct relatively often.

However, its recall was lower than XGBoost.

---

### XGBoost

Produced the strongest overall balance between precision and recall among the classical models:

```text
Precision: 0.8764
Recall:    0.7959
F1:        0.8342
```

XGBoost therefore produced the **highest classical-model F1 score**.

This made it a useful choice when considering both:

- detecting fraudulent transactions
- limiting unnecessary fraud alerts

---

# 🧠 Deep Learning

A Multi-Layer Perceptron neural network was also trained.

The architecture included:

```text
Input
↓
Dense 128 + ReLU
↓
Dropout
↓
Dense 64 + ReLU
↓
Dropout
↓
Dense 32 + ReLU
↓
Dropout
↓
Sigmoid Output
```

`Amount` was standardised before training the neural network.

### Baseline MLP Results

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9533** |
| Precision | **0.8085** |
| Recall | **0.7755** |
| F1 | **≈ 0.79** |

The MLP performed well, but the classical XGBoost model produced a stronger F1 balance in this project.

This demonstrates that a more complex model is not automatically the most useful model.

---

# ⚖️ SMOTE Experiment

SMOTE was explored as an additional class-imbalance technique.

The training data originally contained approximately:

```text
Legitimate: 227,451
Fraud:           394
```

SMOTE generated synthetic minority examples to create a balanced training sample.

```text
Legitimate: 227,451
Synthetic Fraud: 227,451
```

This experiment was useful for understanding oversampling techniques.

However:

> **The main model-performance table in this README reports the original test-set model results, not SMOTE-based final models.**

SMOTE should therefore be treated as an exploratory modelling step rather than the basis of the reported final model comparison.

---

# 💬 NLP & Customer Complaint Analysis

A separate customer-complaint dataset was used to explore unstructured text analysis.

The NLP work included:

- text cleaning
- token processing
- TF-IDF
- word-frequency exploration
- word clouds
- sentiment classification
- basic entity and keyword analysis

This part of the project demonstrates how unstructured customer data can be incorporated into a broader analytical workflow.

---

# 🧪 Sentiment Classification

A Logistic Regression sentiment classifier was trained using TF-IDF features.

The sentiment test results showed an important limitation.

### Sentiment Model

```text
Overall Accuracy ≈ 73%
```

However, performance was highly uneven between sentiment classes.

For the minority sentiment class:

```text
Precision ≈ 0.33
Recall    ≈ 0.02
F1        ≈ 0.04
```

This means the classifier performed poorly at detecting that class despite reasonable overall accuracy.

This is another example of why **class-level metrics matter more than headline accuracy**.

---

# 🔬 Simulated Sentiment Integration

The credit-card dataset and complaint dataset do not share a genuine transaction or customer join key.

Because of that, sentiment could not be linked to fraud transactions in a real business relationship.

Instead, sentiment probabilities were cycled across the transaction dataset as a **simulation**.

```python
sent_feature = np.tile(
    all_sent_probs,
    int(np.ceil(len(df) / len(all_sent_probs)))
)[:len(df)]
```

This allowed the project to demonstrate technically how an additional feature could be introduced into the fraud models.

However:

> **The sentiment experiment is a feature-integration demonstration, not evidence that customer sentiment predicts fraud.**

A real implementation would require a reliable shared key such as:

- Customer ID
- Account ID
- Transaction ID
- Timestamp linkage

---

# 📊 XGBoost + Simulated Sentiment

The sentiment-enhanced XGBoost experiment produced:

| Metric | Baseline XGBoost | + Simulated Sentiment |
|---|---:|---:|
| ROC-AUC | 0.9268 | **0.9372** |
| Precision | 0.8764 | **0.8864** |
| Recall | **0.7959** | **0.7959** |
| F1 | 0.8342 | **0.8387** |

The numerical improvement was small.

Because the sentiment feature was simulated rather than genuinely linked to transactions, this difference should **not be interpreted as evidence that sentiment improves fraud prediction**.

---

# 🧠 MLP + Simulated Sentiment

The sentiment-enhanced MLP produced approximately:

| Metric | Baseline MLP | + Simulated Sentiment |
|---|---:|---:|
| ROC-AUC | **0.9533** | **0.9533** |
| Precision | **0.8085** | 0.7980 |
| Recall | 0.7755 | **0.8061** |
| F1 | ≈ 0.79 | ≈ **0.80** |

The sentiment model increased recall slightly while reducing precision.

Overall performance remained very similar.

This reinforces the main conclusion:

> Adding more features does not automatically produce a meaningfully better model.

---

# 🖥️ Streamlit Fraud Detector

The repository includes a Streamlit inference interface:

```text
streamlit_app.py
```

The app allows a user to enter:

```text
V1 – V28
Amount
```

and receive model scores from:

- Logistic Regression
- XGBoost

The interface also provides sample:

- legitimate transaction
- fraudulent transaction

inputs for demonstration.

---

## Streamlit Ensemble Score

The interface calculates:

```text
(Logistic Regression Score + XGBoost Score)
÷ 2
```

and compares the result with a threshold of:

```text
0.50
```

This should be interpreted as a **demonstration ensemble score**.

The ensemble itself was not separately validated as part of the notebook model-comparison table, and the 0.50 threshold was not selected through a formal business-cost optimisation process.

In a production fraud environment, a threshold would normally be selected using factors such as:

- fraud recall targets
- false-positive cost
- investigator capacity
- transaction value
- customer experience

---

# 💾 Saved Models

The repository includes trained model artifacts so selected models can be used without retraining.

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

Additional scaler files are stored at the project root:

```text
scaler_amount.pkl
scaler_amount_ws.pkl
```

---

# 📌 Key Findings

## 🟢 1. Accuracy Is Not Enough

With fraud representing only **0.17%** of transactions, headline accuracy can be misleading.

Precision, recall, F1 and ROC-AUC provide a much more useful picture of fraud-detection performance.

---

## 🟢 2. XGBoost Produced the Strongest Classical F1 Balance

XGBoost achieved:

```text
Precision: 87.64%
Recall:    79.59%
F1:        83.42%
```

This was the strongest F1 score among the classical models tested.

---

## 🔵 3. Logistic Regression Produced the Highest Classical ROC-AUC

Logistic Regression achieved:

**ROC-AUC = 0.9563**

This shows that simpler models can still perform extremely well on structured fraud data.

---

## 🟠 4. KNN Produced the Highest Precision

KNN achieved:

**Precision = 92.11%**

but detected a smaller proportion of total fraud than XGBoost.

This demonstrates the trade-off between:

```text
Precision
vs
Recall
```

---

## 🧠 5. Deep Learning Was Competitive but Not Dominant

The baseline MLP achieved approximately:

```text
ROC-AUC:   0.953
Precision: 0.809
Recall:    0.776
```

The neural network was competitive but did not clearly outperform the classical models.

---

## 💬 6. Sentiment Added Very Little Reliable Information

The simulated sentiment experiments produced only small numerical changes.

Because the complaint and transaction datasets were not genuinely linked, the experiment demonstrates **technical integration rather than validated fraud-prediction value**.

---

# 🎯 Main Business Takeaway

The biggest lesson from this project is:

> **Fraud detection is a trade-off problem, not an accuracy contest.**

Different models optimise different parts of that trade-off.

In this project:

- Logistic Regression produced the strongest classical ROC-AUC
- KNN produced the highest precision
- XGBoost produced the strongest classical F1 balance
- the MLP remained competitive
- simulated sentiment added little reliable value

A useful fraud system therefore needs more than a high model score.

It also needs:

- appropriate evaluation metrics
- sensible decision thresholds
- awareness of class imbalance
- monitoring of false positives and false negatives
- reliable feature sources
- clear deployment logic

---

# 🧱 Project Workflow

```text
Credit Card Transactions
          ↓
Exploratory Analysis
          ↓
Stratified Train/Test Split
          ↓
Classical ML
          │
          ├── Logistic Regression
          ├── KNN
          ├── Decision Tree
          └── XGBoost
          ↓
Deep Learning MLP
          ↓
Model Evaluation
          ↓
Precision · Recall · F1 · ROC-AUC
          ↓
Experimental NLP / Sentiment Integration
          ↓
Saved Models
          ↓
Streamlit Inference Demo
```

---

# 📁 Repository Structure

```text
fraud-detection-intelligence-system-Portfolio5/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── Capstone_Project_Fixed FINAL.ipynb
├── Capstone_Project_Report.docx
│
├── streamlit_app.py
│
├── scaler_amount.pkl
├── scaler_amount_ws.pkl
│
└── models/
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

---

# 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Data Analysis | Python · pandas · NumPy |
| Machine Learning | Scikit-learn · XGBoost |
| Imbalance Experiment | imbalanced-learn · SMOTE |
| Deep Learning | TensorFlow · Keras |
| NLP | NLTK · spaCy · TF-IDF |
| Visualisation | matplotlib · seaborn |
| Statistical Evaluation | Scikit-learn metrics |
| Application | Streamlit |
| Model Storage | Joblib · Keras |

---

# 🧠 Skills Demonstrated

### Machine Learning

```text
Classification
Train/Test Splitting
Stratified Sampling
Logistic Regression
KNN
Decision Trees
XGBoost
Class-Imbalance Analysis
```

### Model Evaluation

```text
Precision
Recall
F1 Score
ROC-AUC
ROC Curves
Confusion Matrices
```

### Deep Learning

```text
Keras Sequential Models
Dense Layers
Dropout
Early Stopping
Binary Classification
```

### NLP

```text
Text Cleaning
TF-IDF
Sentiment Classification
Word Clouds
Feature Integration
```

### Deployment

```text
Saved Model Artifacts
Inference
Streamlit Interface
```

---

# 📥 Data Availability

The main credit-card dataset is not stored in this repository because of its size.

Dataset:

**Kaggle — Credit Card Fraud Detection**

The original dataset contains:

```text
284,807 transactions
492 fraud cases
31 columns
```

After downloading the dataset, place:

```text
creditcard.csv
```

in the location expected by the notebook before running the full modelling workflow.

---

# ▶️ Getting Started

Clone the repository:

```bash
git clone https://github.com/shababtahsin/fraud-detection-intelligence-system-Portfolio5.git

cd fraud-detection-intelligence-system-Portfolio5
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the modelling notebook:

```text
Capstone_Project_Fixed FINAL.ipynb
```

To run the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

> Depending on the local file layout, saved model and scaler paths must match the paths expected by the application.

---

# 📌 Overall Conclusion

This project demonstrates how extreme class imbalance changes the way classification models should be evaluated.

Across the classical models:

- **Logistic Regression** achieved the highest ROC-AUC at **0.9563**
- **KNN** achieved the highest precision at **0.9211**
- **XGBoost** achieved the strongest classical recall/F1 balance with **0.7959 recall** and **0.8342 F1**

The MLP achieved competitive performance, showing that deep learning can work well on the dataset without automatically outperforming traditional models.

The NLP and sentiment experiments also demonstrated an important analytical lesson:

> **A technically usable feature is not automatically a meaningful business feature.**

Because complaint sentiment was not linked to transactions through a genuine shared key, its use was treated as a simulation rather than evidence of predictive fraud value.

The main conclusion is that **effective fraud analysis requires the right metrics, careful interpretation and an understanding of the trade-off between detecting fraud and creating false alarms.**

---

## 👤 Author

**Shah Tahsin**  
Business Data Analyst | SQL · Python · Power BI · Machine Learning

[GitHub](https://github.com/shababtahsin)
