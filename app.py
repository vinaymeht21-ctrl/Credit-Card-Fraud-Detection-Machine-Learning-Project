"""
Fraud Detection App
--------------------
Streamlit front-end for the `fraud_detection_pipeline.pkl` model
(StandardScaler + OneHotEncoder -> LogisticRegression, trained on the
Kaggle "Fraud Detection Dataset" / PaySim-style data).

Run with:
    streamlit run app.py

Make sure `fraud_detection_pipeline.pkl` is in the same folder as this
file (or update MODEL_PATH below).
"""

import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "fraud_detection_pipeline.pkl"
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🕵️",
    layout="centered",
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_error = None
except FileNotFoundError:
    model = None
    model_error = (
        f"Couldn't find `{MODEL_PATH}`. Place the pickle file in the same "
        "folder as app.py (or update MODEL_PATH at the top of the script)."
    )
except Exception as e:  # noqa: BLE001
    model = None
    model_error = f"Failed to load the model: {e}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run the pipeline on a dataframe that has REQUIRED_COLUMNS."""
    proba = model.predict_proba(df[REQUIRED_COLUMNS])[:, 1]
    pred = (proba >= st.session_state.get("threshold", 0.5)).astype(int)
    out = df.copy()
    out["fraud_probability"] = proba
    out["prediction"] = np.where(pred == 1, "FRAUD", "LEGIT")
    return out


def sample_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
            {
                "type": "PAYMENT",
                "amount": 9839.64,
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
        ]
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🕵️ Fraud Detection")
st.caption(
    "Logistic-regression pipeline trained on transaction balance and type "
    "features to flag likely fraudulent transactions."
)

if model_error:
    st.error(model_error)
    st.stop()

mode = st.sidebar.radio(
    "Mode", ["Single transaction", "Batch upload (CSV)", "About"], index=0
)

st.sidebar.divider()
st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    key="threshold",
    help="Probability above which a transaction is flagged as FRAUD. "
    "Lower it to catch more fraud at the cost of more false alarms.",
)

# --------------------------------------------------------------------------
# Single transaction mode
# --------------------------------------------------------------------------
if mode == "Single transaction":
    st.subheader("Enter transaction details")

    col1, col2 = st.columns(2)
    with col1:
        txn_type = st.selectbox("Transaction type", TXN_TYPES, index=4)
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        old_orig = st.number_input(
            "Old balance (sender, origin)", min_value=0.0, value=1000.0, step=100.0
        )
    with col2:
        new_orig = st.number_input(
            "New balance (sender, origin)", min_value=0.0, value=0.0, step=100.0
        )
        old_dest = st.number_input(
            "Old balance (receiver, dest)", min_value=0.0, value=0.0, step=100.0
        )
        new_dest = st.number_input(
            "New balance (receiver, dest)", min_value=0.0, value=0.0, step=100.0
        )

    if st.button("Predict", type="primary", use_container_width=True):
        row = pd.DataFrame(
            [
                {
                    "type": txn_type,
                    "amount": amount,
                    "oldbalanceOrg": old_orig,
                    "newbalanceOrig": new_orig,
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": new_dest,
                }
            ]
        )
        result = predict(row).iloc[0]

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
        if result["prediction"] == "FRAUD":
            c2.error(f"⚠️ Prediction: {result['prediction']}")
        else:
            c2.success(f"✅ Prediction: {result['prediction']}")

        with st.expander("Details"):
            st.write(
                {
                    "balanceDiffOrig (sender balance change)": old_orig - new_orig,
                    "balanceDiffDest (receiver balance change)": new_dest - old_dest,
                    "threshold used": st.session_state["threshold"],
                }
            )

# --------------------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------------------
elif mode == "Batch upload (CSV)":
    st.subheader("Upload a CSV of transactions")
    st.write(
        "The file must contain these columns: "
        f"`{'`, `'.join(REQUIRED_COLUMNS)}`"
    )

    st.download_button(
        "Download a sample template CSV",
        data=sample_template().to_csv(index=False),
        file_name="transactions_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read that CSV: {e}")
            data = None

        if data is not None:
            missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                results = predict(data)
                n_fraud = (results["prediction"] == "FRAUD").sum()
                st.success(
                    f"Scored {len(results)} transactions — "
                    f"{n_fraud} flagged as FRAUD."
                )
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    data=results.to_csv(index=False),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

# --------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------
else:
    st.subheader("About this model")
    st.markdown(
        """
This app wraps a scikit-learn `Pipeline`:

- **Preprocessing:** `StandardScaler` on `amount`, `oldbalanceOrg`,
  `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`; `OneHotEncoder`
  on transaction `type`.
- **Model:** `LogisticRegression(class_weight="balanced")`.
- **Trained on:** a PaySim-style mobile money transaction dataset
  (Kaggle "Fraud Detection Dataset").

**Note on `class_weight="balanced"`:** the training data is heavily
imbalanced (fraud is a tiny fraction of transactions). Balancing the
class weights pushes the model to catch more fraud cases (higher
recall) at the cost of more false alarms (lower precision) than a raw
accuracy score would suggest. Use the threshold slider in the sidebar
to tune that trade-off for your use case, and validate against your
own labeled data before relying on this for real decisions.
        """
    )
