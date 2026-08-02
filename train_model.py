import os
import pandas as pd
import numpy as np
import joblib

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

# Load dataset
df = pd.read_csv("dataset/adult.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Handle missing values
df.replace([" ?", "?"], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and target
X = df.drop("income", axis=1)
y = df["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save test data
test_data = X_test.copy()
test_data["income"] = y_test.values
test_data.to_csv("test_data.csv", index=False)

# Create model directory
os.makedirs("models", exist_ok=True)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = 0

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    mcc = matthews_corrcoef(y_test, predictions)

    print("=" * 60)
    print(name)
    print("Accuracy :", accuracy)
    print("AUC :", auc)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 :", f1)
    print("MCC :", mcc)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

print(results_df)

results_df.to_csv("model_results.csv", index=False)
