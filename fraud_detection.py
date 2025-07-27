import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load model without compiling (to avoid optimizer/metric warnings)
autoencoder = tf.keras.models.load_model("model.h5", compile=False)

# App Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("This app uses a trained Autoencoder model to detect fraudulent transactions.")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Fraud Detection Threshold", min_value=0.001, max_value=0.1, value=0.02, step=0.001)

# Input mode selection
mode = st.radio("Choose Input Mode", ["🧮 Manual Entry", "📂 Upload CSV"], horizontal=True)

# --- Manual Input ---
if mode == "🧮 Manual Entry":
    st.subheader("Enter Transaction Features Manually")
    st.info("Provide values for 29 transaction features (V1 to V28 and Amount).")
    input_data = []
    for i in range(1, 29):
        value = st.number_input(f"V{i}", value=0.0, format="%.4f")
        input_data.append(value)
    amount = st.number_input("Amount", value=0.0, format="%.2f")
    input_data.append(amount)

    if st.button("🔍 Predict"):
        input_array = np.array([input_data])
        reconstruction = autoencoder.predict(input_array)
        loss = np.mean(np.square(input_array - reconstruction), axis=1)[0]
        st.metric(label="Reconstruction Loss", value=f"{loss:.5f}")

        if loss > threshold:
            st.error("⚠️ Fraud Detected!")
        else:
            st.success("✅ Transaction is Normal.")

# --- CSV Upload ---
else:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Upload CSV with 29 columns: V1 to V28 and Amount", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(data.head())

            if data.shape[1] != 29:
                st.error("CSV must contain exactly 29 columns (V1–V28 and Amount).")
            else:
                if st.button("🔍 Predict on CSV"):
                    reconstructions = autoencoder.predict(data.values)
                    losses = np.mean(np.square(data.values - reconstructions), axis=1)
                    predictions = (losses > threshold).astype(int)

                    result_df = data.copy()
                    result_df["Reconstruction_Loss"] = losses
                    result_df["Prediction"] = ["Fraud" if p == 1 else "Normal" for p in predictions]

                    st.success(f"Detected {sum(predictions)} fraud(s) out of {len(predictions)} transactions.")
                    st.dataframe(result_df.head(10))

                    csv = result_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "fraud_detection_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
