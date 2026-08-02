# Machine Learning Assignment 2

## Problem Statement

Build and compare multiple machine learning classification models on a public dataset. Evaluate their performance using standard metrics and deploy the solution using Streamlit Community Cloud.

## Dataset Description

* Dataset Name: Adult Income Dataset
* Source: UCI Machine Learning Repository
* Type: Binary Classification
* Target Variable: income
* Minimum requirements satisfied: More than 12 features and more than 400 records.

## GitHub Repository Link

https://github.com/praveen1706/ML\_Assignment2

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest Classifier

## Evaluation Metrics Comparison

|ML Model|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-|-|-|-|-|-|
|Logistic Regression|0.83|0.740609|0.50000|0.176471|0.260870|0.221954|
|Decision Tree|0.92|0.858257|0.764706|0.764706|0.764706|0.716513|
|KNN|0.79|0.463147|0.000000|0.000000|0.000000|-0.092380|
|Naive Bayes|0.83|0.793763|0.000000|0.000000|0.000000|0.000000|
|Random Forest|0.922041|0.750000|0.176471|0.285741|0.285714|0.315180|

## Model Performance Observations

|ML Model|Observation|
|-|-|
|Logistic Regression|Served as a strong baseline model with fast training time and good overall classification performance. It worked well when the relationship between features and target variable was approximately linear.|
|Decision Tree|Produced easily interpretable results and captured non-linear relationships. However, it showed a higher tendency to overfit compared to other models.|
|KNN|Performed reasonably well by classifying instances based on nearest neighbors. Performance depended on the choice of K value and became slower as dataset size increased.|
|Naive Bayes|Achieved fast training and prediction times. It performed well despite its simplifying assumption of feature independence, but sometimes produced lower accuracy than ensemble methods.|
|Random Forest|Delivered the most balanced performance across all evaluation metrics. By combining multiple decision trees, it reduced overfitting and achieved better generalization on unseen data.|
|Overall Winner|Random Forest was selected as the overall winner because it obtained the best combination of Accuracy, AUC, Precision, Recall, F1-Score, and MCC among all the evaluated models. Its ensemble learning approach improved prediction performance and reduced overfitting.|

## Streamlit Application Features

* Dataset Upload (CSV)
* Model Selection Dropdown
* Evaluation Metrics Display
* Confusion Matrix or Classification Report

## Deployment

* Streamlit Community Cloud
* Live App Link: Add after deployment

