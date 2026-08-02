import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Machine Learning Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("Machine Learning Classification Models")
st.write("BITS WILP - Machine Learning Assignment 2")

# ----------------------------
# Model Dictionary
# ----------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.header("Settings")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(model_files.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)

# ----------------------------
# Main Logic
# ----------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    # Last column should be target

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    model = joblib.load(model_files[selected_model])

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(y, predictions)

    recall = recall_score(y, predictions)

    f1 = f1_score(y, predictions)

    auc = roc_auc_score(y, probabilities)

    mcc = matthews_corrcoef(y, predictions)

    st.subheader("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{precision:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Recall", f"{recall:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(5,5))

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(ax=ax)

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Prediction Sample")

    result = X.copy()

    result["Actual"] = y

    result["Predicted"] = predictions

    st.dataframe(result.head(20))

else:

    st.info("Please upload test_data.csv")

