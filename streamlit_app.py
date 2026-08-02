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
    classification_report,
)

st.set_page_config(
    page_title="Machine Learning Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

st.write(
    "Upload a CSV dataset and evaluate different classification models."
)

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
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "income" not in data.columns:
        st.error("Target column 'income' not found.")
    else:

        X = data.drop("income", axis=1)
        y = data["income"]

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if needed
        if y.dtype == "object":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        if selected_model == "Logistic Regression":
            model = LogisticRegression(
                max_iter=5000,
                solver="liblinear"
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

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

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
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col1.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(
            classification_report(
                y_test,
                y_pred
            )
        )
