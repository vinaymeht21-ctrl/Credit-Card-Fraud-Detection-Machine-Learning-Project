# 🕵️ Fraud Detection App

A machine learning web app that flags potentially fraudulent financial transactions in real time, built with **scikit-learn** and **Streamlit**.

## Overview

Mobile money and bank transfer fraud typically shows up as a tiny fraction of all transactions, which makes it a classic imbalanced-classification problem. This project trains a Logistic Regression pipeline on transaction-level data (type, amount, and account balances before/after the transaction) and serves it through an interactive Streamlit app that supports both single-transaction checks and batch CSV scoring.

## Features

- **Single transaction prediction** — enter transaction details in a form and get an instant fraud probability.
- **Batch scoring** — upload a CSV of transactions and download the results with fraud probabilities attached.
- **Adjustable decision threshold** — tune the precision/recall trade-off from the sidebar instead of a hardcoded cutoff.
- **Model info page** — explains the pipeline and the imbalanced-data trade-offs in plain language.

## Dataset

Trained on the [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download) from Kaggle — a PaySim-style synthetic dataset of mobile money transactions with an `isFraud` label. The dataset itself isn't included in this repo (see [Notes](#notes)); download it from Kaggle if you want to retrain the model.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Modeling | scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`) |
| App / UI | Streamlit |
| Data handling | pandas, numpy |
| Notebook | Jupyter |

## Model Details

- **Preprocessing:** `StandardScaler` on `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`; `OneHotEncoder` on transaction `type`.
- **Classifier:** `LogisticRegression(class_weight="balanced")` — balances the classes during training so the model doesn't just predict "not fraud" every time, at the cost of more false positives.
- **Exported with:** `joblib` as `fraud_detection_pipeline.pkl`.

Full EDA and training steps are in [`model.ipynb`](model.ipynb).

## Project Structure

```
.
├── app.py                          # Streamlit app
├── model.ipynb                     # EDA + model training notebook
├── fraud_detection_pipeline.pkl    # Trained model pipeline (used by app.py)
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<vinaymeht21-ctrl>/<Credit-Card-Fraud-Detection-Machine-Learning-Project>.git
cd <Credit-Card-Fraud-Detection-Machine-Learning-Project>
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

- **Single transaction:** fill in the transaction type and balances, click **Predict**, and read off the fraud probability and verdict.
- **Batch upload:** download the sample template from the app, fill it in (or export from your own data) with columns `type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`, upload it, and download the scored results.

## Limitations & Future Improvements

- Trained on synthetic data (PaySim-style); real-world transaction patterns may differ.
- Logistic Regression is used for interpretability — trying tree-based models (Random Forest, XGBoost) could improve recall/precision further.
- No authentication or persistence layer; this is a demo/portfolio app, not production infrastructure.
- Could add SHAP-based explanations per prediction to show *why* a transaction was flagged.

## Notes

- `fraud_detection_pipeline.pkl` is committed to this repo since it's small (a few KB) and is what makes the app runnable out of the box.
- The raw training CSV is **not** committed (see `.gitignore`) since it's large and easily re-downloaded from Kaggle.

## License

MIT License

Copyright (c) 2026 Vinay Mehta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

**Vinay Mehta** — [LinkedIn](#) · [GitHub](#) · [Portfolio](#)