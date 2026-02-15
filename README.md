# ML_Assignment_2 - Breast Cancer Tumor Classification

### Problem Statement

The objective of this project is to classify breast tumors as **malignant** or **benign** using machine learning classification models. Early detection of malignant tumors is critical for medical diagnosis. By training multiple machine learning models, we aim to identify the most accurate model for predicting tumor malignancy.

---

### Dataset Description

The dataset contains **569 samples** with **30 numerical features** computed from digitized images of fine needle aspirate (FNA) of breast mass.

**Features include:**
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- And other statistical measurements of cell nuclei.

**Target Variable:**
- **0 → Malignant**
- **1 → Benign**

---

### Models Used

Six different classification models were implemented and compared:

- **Logistic Regression**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Random Forest**
- **XGBoost**

---

### Model Evaluation Results

| **Model**               | **Accuracy** | **AUC**  | **Precision** | **Recall** | **F1 Score** | **MCC**   |
|-------------------------|--------------|----------|---------------|------------|--------------|-----------|
| Logistic Regression      | 0.982456     | 0.995370 | 0.986111      | 0.986111   | 0.986111     | 0.962302  |
| Decision Tree            | 0.912281     | 0.915675 | 0.955882      | 0.902778   | 0.928571     | 0.817412  |
| K-Nearest Neighbors (KNN)| 0.956140     | 0.978836 | 0.958904      | 0.972222   | 0.965517     | 0.905447  |
| Naive Bayes             | 0.929825     | 0.986772 | 0.944444      | 0.944444   | 0.944444     | 0.849206  |
| Random Forest           | 0.956140     | 0.993882 | 0.958904      | 0.972222   | 0.965517     | 0.905447  |
| XGBoost                 | 0.956140     | 0.990079 | 0.946667      | 0.986111   | 0.965986     | 0.905824  |

---

### Observations

| **Model**               | **Observation**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **Logistic Regression**  | Performs very well due to linear separability of the dataset. It shows high precision, recall, and accuracy. |
| **Decision Tree**        | Slight overfitting observed with high precision but lower recall. The model struggles to generalize as well as others. |
| **K-Nearest Neighbors (KNN)** | Performs well, especially with high recall and precision. Scaling the data boosts its performance. |
| **Naive Bayes**          | Good performance with decent precision and recall. However, it slightly lags behind other models in terms of F1 Score. |
| **Random Forest**        | Solid generalization and performance across most metrics. It balances precision and recall well, showing strong stability. |
| **XGBoost**              | Shows the best overall performance with a high AUC and MCC. This is the most reliable model, with high recall and precision. |

---

### Conclusion

**XGBoost** is the best-performing model in terms of **AUC** and **MCC**, and is the most reliable model for predicting tumor malignancy. **Logistic Regression** also performs well, especially in terms of precision and recall, while **Random Forest** shows strong generalization ability. **Decision Tree** and **Naive Bayes** perform less robustly due to overfitting and the assumption of feature independence, respectively.

---

### How to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ML_Assignment_2.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Upload the `test_data.csv` file when prompted.

---

### Deployment Link

You can access the deployed app on Streamlit Cloud here:

[Streamlit App Link](https://2025AB05249.streamlit.app)

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
