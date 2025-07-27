# Credit Card Fraud Detection using Autoencoders

A machine learning-based fraud detection system that leverages unsupervised learning via autoencoders to identify suspicious credit card transactions. The solution includes a user-friendly Streamlit interface for manual and bulk analysis of transactions.

---

## 🚀 Project Overview

This project is developed as part of my MCA Final Year. It addresses the growing challenge of fraudulent transactions in the financial sector using anomaly detection. The autoencoder is trained only on legitimate transactions, and fraud is detected based on reconstruction loss.

---

## 💡 Key Features

- 🧠 **Unsupervised Learning**: Trained only on non-fraudulent transactions using autoencoders.
- 📉 **Reconstruction Loss Thresholding**: Identify frauds based on deviation from normal patterns.
- 🖥️ **Streamlit Interface**: Easy-to-use web UI for both technical and non-technical users.
- 📄 **CSV Upload**: Bulk prediction for transaction datasets.
- ⚙️ **Threshold Tuning**: Users can adjust sensitivity via a slider.
- 📥 **Download Results**: Export predictions and reconstruction losses as CSV.

---

## 📊 Dataset

This project uses the publicly available **Credit Card Fraud Detection Dataset** provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- It contains **284,807 transactions**, with 492 fraudulent cases.
- All features (V1–V28) are anonymized via PCA, along with `Time`, `Amount`, and `Class` (fraud label).

➡️ **Dataset link**: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> ⚠️ Due to its large size, the dataset is not included in this repository. Please download it directly from the link above.


- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend/ML**: Python, TensorFlow/Keras
- **Libraries**: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn




