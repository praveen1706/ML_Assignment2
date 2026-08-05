import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

# PAGE SETUP

st.set_page_config(
    page_title="Machine Learning Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

# MODEL SELECTION

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

st.write("Selected Model:", selected_model)

# FILE UPLOAD

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

# FEATURES & TARGET

        X = data.drop("income", axis=1)
        y = data["income"]

# Encode target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
# Handle missing values
        X = X.fillna("Unknown")

# One-hot encoding
        X = pd.get_dummies(X, drop_first=True)

# Numeric conversion
        X = X.astype(float)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
# TRAIN TEST SPLIT

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

# MODEL CREATION

        if selected_model == "Logistic Regression":

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=5000,
                        class_weight="balanced"
                    )
                )
            ])

        elif selected_model == "Decision Tree":

            model = DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )

        elif selected_model == "KNN":

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5
                    )
                )
            ])

        elif selected_model == "Naive Bayes":

            model = GaussianNB()

        elif selected_model == "Random Forest":

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
# TRAIN MODEL

        model.fit(X_train, y_train)

        st.success(
            f"Model Trained Successfully: {selected_model}"
        )

# PREDICTIONS

        y_pred = model.predict(X_test)

        st.subheader("Prediction Distribution")
        st.write(
            pd.Series(y_pred).value_counts()
        )

# METRICS

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

        try:

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            auc = roc_auc_score(
                y_test,
                y_prob
            )

        except Exception:

            auc = 0.0

# DISPLAY METRICS

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

# CONFUSION MATRIX

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        st.write(cm)

# CLASSIFICATION REPORT

        st.subheader("Classification Report")

        st.text(
            classification_report(
                y_test,
                y_pred
            )
        )
        
# CLASSIFICATION DISTRIBUTION

        st.subheader("Class Distribution")
        st.write(data["income"].value_counts())
    

    except Exception as e:

        st.error("Actual Error")
        st.code(str(e))

