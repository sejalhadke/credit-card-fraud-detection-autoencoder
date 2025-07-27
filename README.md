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

## 🧰 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend/ML**: Python, TensorFlow/Keras
- **Libraries**: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn




