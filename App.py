import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ===============================
# Load trained ANN model
# ===============================
model = tf.keras.models.load_model('model.h5')

# ===============================
# Load encoders & scaler
# ===============================
with open('onehot_encoder_geo.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ===============================
# Streamlit UI
# ===============================
st.title("Customer Churn Prediction")

geography = st.selectbox(
    "Geography",
    geo_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

age = st.slider("Age", 18, 100, 30)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Tenure (years)", 0, 10, 5)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ===============================
# Create input DataFrame
# ===============================
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# ===============================
# Encode categorical features
# ===============================

# Encode Gender
input_data["Gender"] = gender_encoder.transform(input_data["Gender"])

# OneHot Encode Geography
geo_encoded = geo_encoder.transform(input_data[["Geography"]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(["Geography"])
)

# Combine data
input_data = pd.concat(
    [input_data.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)

# ===============================
# Scale features
# ===============================
input_scaled = scaler.transform(input_data)

# ===============================
# Predict
# ===============================
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]

# ===============================
# Output
# ===============================
st.subheader("Prediction Result")

if churn_probability >= 0.5:
    st.error(f"⚠️ Customer is likely to churn\n\nProbability: {churn_probability:.2f}")
else:
    st.success(f"✅ Customer is unlikely to churn\n\nProbability: {churn_probability:.2f}")
