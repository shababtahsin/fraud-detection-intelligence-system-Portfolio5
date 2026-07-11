# 🛡️ AI-Driven Fraud Detection System
### Multi-Modal ML + NLP + Deep Learning Pipeline for Credit Card Fraud, Built on 284,807 Real Transactions

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.98-brightgreen)]()
[![Deployment](https://img.shields.io/badge/Deployed-FastAPI%20%2B%20Streamlit-orange)]()
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## 📌 TL;DR (for non-technical readers)

Banks lose billions every year to credit card fraud. The catch: fraud is **rare** — only 1 in every 578 transactions in this dataset is fraudulent — so a lazy model can just guess "not fraud" every time and still be 99.8% "accurate" while catching zero criminals.

This project builds a system that actually finds the needle in the haystack. It:

- Tests 6 different machine learning approaches and picks the best one (**XGBoost, catching fraud with 0.98 AUC** — near-perfect separation between fraud and legitimate activity)
- Reads real customer complaint text and pulls sentiment/risk signals out of it, the same way a bank's compliance team would
- Wraps everything into a **live web app** where you type in transaction details and get an instant fraud probability

**Bottom line: this is a working, end-to-end fraud detection product — not a notebook that stops at a confusion matrix.**

---

## 🎯 Business Problem

Credit card fraud detection is a **needle-in-a-haystack, cost-asymmetric** problem:

| Challenge | Why it matters |
|---|---|
| **Extreme class imbalance** | Only 492 of 284,807 transactions (0.17%) are fraudulent |
| **Asymmetric cost of errors** | A missed fraud costs real money; a false alarm costs customer trust |
| **Unstructured signal is ignored by most models** | Customer complaints contain early warning signals that pure transaction data misses |
| **Static rule-based systems age fast** | Fraud patterns shift — the system needs to generalise, not memorise |

**Business question answered:** *Can we build a fraud detection system that materially improves fraud capture without drowning the fraud team in false alarms?*

**Answer: Yes.** The final model achieves **89–91% precision** and **80–82% recall** at a 0.98 AUC — meaning it catches the large majority of fraud while keeping false alarms low enough to be operationally usable.

---

## 🏆 Key Results

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.94 | 0.84 | 0.66 | 0.74 |
| KNN | 0.91 | 0.90 | 0.74 | 0.81 |
| Decision Tree | 0.87 | 0.72 | 0.72 | 0.72 |
| MLP (Deep Learning, baseline) | 0.97 | 0.85 | 0.78 | 0.81 |
| MLP + Sentiment | 0.97 | 0.86 | 0.79 | 0.82 |
| **XGBoost** | **0.98** | **0.90** | **0.82** | **0.86** |
| **XGBoost + Sentiment** | **0.98** | **0.91** | **0.82** | **0.86** |

**Winner: XGBoost** — best AUC, best precision/recall trade-off of any model tested, and cheap enough to run in real time. It's also the model wired into the deployed app.

An honest note most portfolio projects leave out: sentiment integration gave only a **marginal** lift here. The complaint-to-transaction mapping is simulated, not a true join key, so I'm not overselling it — the value is in demonstrating the *pipeline*, not claiming the sentiment signal alone moved the needle in this dataset.

---

## 🧠 What This Project Actually Does

A single unified pipeline spanning **structured ML, deep learning, NLP, multi-agent scoring, and retrieval-augmented generation** — built in 7 phases:

**Phase 1 — Data Import & Exploration**
Loaded and audited 284,807 transactions × 31 columns; confirmed zero missing values; visualised the fraud/legit class split and the Amount/Time distributions to understand what "normal" looks like before modelling.

**Phase 2 — Classical Machine Learning**
Trained and benchmarked 4 models (Logistic Regression, KNN, Decision Tree, XGBoost) on a stratified train/test split, then applied **SMOTE** (Synthetic Minority Over-sampling) to fix the class imbalance and re-validated with cross-validation — SMOTE proved essential, without it every model had high accuracy but useless recall.

**Phase 3 — NLP on 5,423 Customer Complaints**
Cleaned and lemmatised real customer complaint text (lowercasing, punctuation/stopword removal, lemmatization), built a word cloud to surface dominant risk themes, engineered regex-based keyword risk flags, vectorised the corpus with TF-IDF, and ran Named Entity Recognition with spaCy to extract organisations, locations, and monetary entities from complaint text.

**Phase 4 — Deep Learning**
Built and trained a Multi-Layer Perceptron (MLP) on scaled transaction features as a deep learning baseline — competitive with XGBoost out of the gate, validating that a neural approach was worth pursuing further (a CNN was tested and correctly ruled out as unsuitable for this tabular structure).

**Phase 5 — Sentiment Fusion + Multi-Agent Risk Scoring**
Trained a sentiment classifier on the complaint corpus, engineered a sentiment feature into the transaction data, and retrained both XGBoost and the MLP with it. Then simulated a **multi-agent system** — independent rule-based "agents" each flag risk from a different angle (amount anomaly, time-of-day, sentiment, model score) — combined into a single interpretable Meta-Risk Score.

**Phase 6 — Retrieval-Augmented Generation (RAG)**
Built a document store from the cleaned complaint corpus, vectorised it, implemented cosine-similarity retrieval, and connected an LLM (Gemini) layer to generate plain-English explanations for why a transaction was flagged — turning a black-box score into something a compliance analyst can actually read and act on.

**Phase 7 — Model Saving & Deployment**
Serialized the final models (XGBoost, Logistic Regression, scaler) with joblib, ran sample inference on unseen transactions, exported flagged transactions for review, and shipped a working **FastAPI + Streamlit** application for real-time and batch scoring.

---

## 🚀 Live Deployment

The repo includes a fully working **Streamlit app** (`streamlit_app.py`) that:

- Loads the trained XGBoost and Logistic Regression models via `joblib`
- Lets you manually enter transaction features, or one-click load a real **legit** or **fraud** sample transaction
- Returns fraud probability from both models plus an ensemble score
- Visualises the model comparison live in-browser

```bash
streamlit run streamlit_app.py
```

This isn't a deployment *stub* — it's a runnable interface a non-technical stakeholder (a fraud analyst, a manager) could actually use to test a transaction in seconds.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Deep Learning | TensorFlow / Keras (MLP) |
| NLP | NLTK, spaCy, TF-IDF, WordCloud, TextBlob |
| Retrieval / RAG | Sentence-Transformers, FAISS, Google Generative AI (Gemini) |
| Deployment | FastAPI, Streamlit, Uvicorn, Joblib |
| Validation | Pydantic |

---

## 📂 Repository Structure

```
├── Capstone_Project_Fixed_FINAL.ipynb   # Full 7-phase analysis notebook
├── streamlit_app.py                     # Deployed fraud-scoring web app
├── Customer_Complaints_Sentiment.csv    # NLP source data (5,423 complaints)
├── requirements.txt                     # Full dependency list
├── README.md
└── .gitignore
```

> **Note on data:** The primary transaction dataset (`creditcard.csv`, ~150MB, 284,807 rows) is excluded from this repo via `.gitignore` due to GitHub's file size limits. It's the standard [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) — publicly available, drop it in the working directory to reproduce.

---

## 📊 Business Impact

- Demonstrates a fraud detection approach that can be **operationalised**, not just benchmarked — precision/recall are balanced at a threshold a real fraud team could actually staff against
- The Meta-Risk Score and RAG explanation layer turn model output into something **auditable and explainable** — critical in a regulated financial services context where "the model said so" isn't an acceptable answer to a regulator
- Shows the full analyst-to-engineer skill chain: problem framing → EDA → feature engineering → model selection with honest trade-off analysis → deployment

---

## ⚙️ Getting Started

```bash
# clone the repo
git clone <your-repo-url>
cd <repo-name>

# install dependencies
pip install -r requirements.txt

# download creditcard.csv from Kaggle and place it in the project directory

# run the notebook for the full analysis, or launch the app directly:
streamlit run streamlit_app.py
```

---

## 👤 About This Project

Built by **Shah Tahsin** — a Perth-based Business Data Analyst combining a law background (LLB, University of Manchester), a Master of Management (Curtin University), and a Graduate Certificate in Data Science & AI (Institute of Data / Curtin University). This capstone was completed as part of that program and reflects an end-to-end, portfolio-grade approach: real data, honest model comparison, and a deployed product — not just a notebook.

- 🔗 LinkedIn: [add your URL]
- 🔗 Portfolio / GitHub: [add your URL]
- 📧 Contact: [add your email]

---

*If you're a recruiter or hiring manager: this project is one of five in my portfolio, spanning SQL Server, Power BI, Python, and MySQL across credit risk, retail analytics, procurement, and fraud domains. Happy to walk through the full technical decision-making on a call.*
