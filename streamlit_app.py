from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Model Selection

if selected_model == "Logistic Regression":

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced"
        ))
    ])

elif selected_model == "Decision Tree":

    model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

elif selected_model == "KNN":

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])

elif selected_model == "Naive Bayes":

    model = GaussianNB()

elif selected_model == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

# Train Model

model.fit(X_train, y_train)

# Debug Information

st.success(f"Model Trained: {selected_model}")
st.write("Training Shape:", X_train.shape)

# Predictions

y_pred = model.predict(X_test)

st.subheader("Prediction Distribution")
st.write(pd.Series(y_pred).value_counts())
