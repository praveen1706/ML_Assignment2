import streamlit as st
import pandas as pd
import joblib
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
    page_title="Machine Learning Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

st.write(
    "Upload test data and evaluate different machine learning models."
)

model_options = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}
selected_model = st.selectbox(
"Choose Model",
[
"Logistic Regression",
"Decision Tree",
"KNN",
"Naive Bayes",
"Random Forest"
]
)
if selected_model == "Logistic Regression": LogisticRegression(
    solver="liblinear",
    max_iter=5000,
    random_state=42)
elif selected_model == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
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

selected_model = st.selectbox(
    "Select Model",
    list(model_options.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    if "income" not in data.columns:
        st.error("Target column 'income' not found.")
    else:

        X = data.drop("income", axis=1)
        y = data["income"]

        model = joblib.load(
            model_options[selected_model]
        )

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = 0

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(
            y,
            y_pred,
            zero_division=0
        )
        recall = recall_score(
            y,
            y_pred,
            zero_division=0
        )
        f1 = f1_score(
            y,
            y_pred,
            zero_division=0
        )
        mcc = matthews_corrcoef(
            y,
            y_pred
        )

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

        st.subheader("Confusion Matrix")

        st.write(
            confusion_matrix(y, y_pred)
        )

        st.subheader("Classification Report")

        st.text(
            classification_report(
                y,
                y_pred
            )
        )
