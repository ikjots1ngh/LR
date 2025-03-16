import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

st.title("Logistic Regression Assignment")

# Load dataset from GitHub instead of a local file path
csv_url = "https://raw.githubusercontent.com/ikjots1ngh/logistic-regression-app/main/Titanic_train.csv"
df = pd.read_csv(csv_url)

st.write("### Dataset Preview")
st.dataframe(df)

# Data Cleaning
df = df.dropna()
df = df.drop(columns=['Cabin', 'Embarked', 'Name', 'Sex', 'Ticket', 'PassengerId'], errors='ignore')
st.write(f"### Cleaned Data Shape: {df.shape}")

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Histogram of numerical columns
st.write("### Histogram of Numerical Values")
fig, ax = plt.subplots(figsize=(10, 6))
df.hist(bins=20, figsize=(10, 6), color="skyblue", edgecolor="black", ax=ax)
plt.suptitle("Histogram of Numerical Values")
st.pyplot(fig)

# Define features and target variable
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X, y)

# Predictions
y_pred = classifier.predict(X)

# Confusion Matrix
st.write("### Confusion Matrix")
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.write("### Classification Report")
st.text(classification_report(y, y_pred))

# ROC Curve
st.write("### ROC Curve")
fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba(X)[:, 1])
auc = roc_auc_score(y, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="red", label="Logistic Model (AUC = %0.2f)" % auc)
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)
