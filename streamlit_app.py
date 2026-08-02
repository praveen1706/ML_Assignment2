import streamlit as st
import pandas as pd
import numpy as np

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

st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

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

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        if "income" not in data.columns:
            st.error("Target column 'income' not found.")
            st.stop()

        # Features and target
        X = data.drop("income", axis=1)
        y = data["income"]

        # Encode target to 0/1
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

        # Handle missing values
        X = X.fillna("Unknown")

        # Convert categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Ensure all numeric
        X = X.astype(float)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Select model
        if selected_model == "Logistic Regression":

            model = LogisticRegression(
                solver="liblinear",
                max_iter=5000
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

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

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

        try:
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(
                y_test,
                y_prob
            )

        except Exception:
            auc = 0.0

        # Display Metrics
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col1.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    except Exception as e:

        st.error("Actual Error")
        st.code(str(e))
