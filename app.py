import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_option}.pkl")

    X = data.drop("target", axis=1)
    y = data["target"]

    X = scaler.transform(X)

    predictions = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)
