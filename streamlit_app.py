import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------

st.set_page_config(
    page_title="Machine Learning Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

st.write(
    "Upload a CSV dataset and evaluate different classification models."
)

# -------------------------------------------
# Model Selection
# -------------------------------------------

selected_model = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

# -------------------------------------------
# File Upload
# -------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # ---------------------------------------
    # Check Target Column
    # ---------------------------------------

    if "income" not in data.columns:
        st.error(
            "Target column 'income' not found in the uploaded CSV."
        )
        st.stop()

    # ---------------------------------------
    # Features and Target
    # ---------------------------------------

    X = data.drop("income", axis=1)
    y = data["income"]

    # ---------------------------------------
    # Handle Missing Values
    # ---------------------------------------

    X = X.fillna("Unknown")

    # ---------------------------------------
    # Encode Categorical Features
    # ---------------------------------------

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # Force Numeric Data Type

    X = X.astype(float)

    # ---------------------------------------
    # Encode Target Column
    # ---------------------------------------

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ---------------------------------------
    # Split Dataset
    # ---------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------
    # Select Model
    # ---------------------------------------

    if selected_model == "Logistic Regression":

        model = LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            random_state=42
        )

    elif selected_model == "Decision Tree":

        model = DecisionTreeClassifier(
            random_state=42
        )

    elif selected_model == "KNN":

        model = KNeighborsClassifier()

    elif selected_model == "Naive Bayes":

        model = GaussianNB()

    elif selected_model == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # ---------------------------------------
    # Train Model
    # ---------------------------------------

    model.fit(X_train, y_train)

    # ---------------------------------------
    # Predictions
    # ---------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # ---------------------------------------
    # AUC Score
    # ---------------------------------------

    try:
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    except Exception:
        auc = 0.0

    # ---------------------------------------
    # Display Metrics
    # ---------------------------------------

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col1.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col2.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col3.metric(
        "MCC",
        f"{mcc:.4f}"
    )

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    st.write(cm)

    # ---------------------------------------
    # Classification Report
    # ---------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred
    )

    st.text(report)
