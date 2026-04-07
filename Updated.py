import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from fpdf import FPDF
import xgboost as xgb  # New: XGBoost Support

# Initialize API key, tokenizer, and model
apikey = "HUGGINGFACE_TOKEN_REMOVED"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize encryption key and cipher
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        return decrypted_data
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

# Title
st.title('AI Assistant for Data Science 🤖')

# Sidebar Information
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('Upload a CSV file to begin. I will analyze the data, provide insights, generate visualizations, and suggest appropriate machine learning models to tackle your problem.')
    st.divider()
    st.caption("<p style ='text-align:center'> made with ❤️ by Team Avengers</p>", unsafe_allow_html=True)

# File Upload
user_csv = st.file_uploader("Upload your CSV file", type="csv")
if user_csv is not None:
    file_data = user_csv.read()
    encrypted_file = encrypt_data(file_data.decode(errors='ignore'))
    st.write("Your data has been successfully encrypted.")

    try:
        decrypted_file = decrypt_data(encrypted_file)
        df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
    except Exception as e:
        st.error(f"An error occurred during decryption: {e}")
        st.stop()

    st.header('📊 Exploratory Data Analysis')
    st.write("First rows of your dataset:")
    st.write(df.head())

    # Feature Selection
    feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
    target_col = st.selectbox("Select Target Column", df.columns.tolist())

    if feature_cols and target_col:
        st.header('⚡ Train a Machine Learning Model')
        
        model_choice = st.selectbox(
            "Choose an ML Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression", "XGBoost"]
        )

        perform_cv = st.checkbox("Perform Cross-Validation")

        # Train-Test Split
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_mapping = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
            "Support Vector Regression": SVR(C=1.0, kernel='rbf'),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        }

        model = model_mapping[model_choice]

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, predictions)

            st.write(f"**Model Performance**")
            st.write(f"🔹 RMSE: {rmse:.2f}")
            st.write(f"🔹 R² Score: {r2:.2f}")

            # Save the model
            model_filename = f"trained_model_{model_choice.replace(' ', '_')}.pkl"
            with open(model_filename, 'wb') as model_file:
                pickle.dump(model, model_file)

            # Provide model download link
            st.download_button(
                label="📥 Download Trained Model",
                data=open(model_filename, "rb").read(),
                file_name=model_filename,
                mime="application/octet-stream"
            )

            # Provide predictions download link
            predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            csv_data = predictions_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Feature Importance Plot for Tree-Based Models
            if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": feature_importances})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)

                st.write("### 🔥 Feature Importance")
                fig, ax = plt.subplots()
                sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)
