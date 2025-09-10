import streamlit as st
import pickle, json
import numpy as np

# ---- Load Model ----
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# ---- Load Columns ----
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract location names (skip numeric columns)
locations = [col for col in data_columns if col not in ['total_sqft','bath','bhk']]

# ---- UI Styling ----
st.set_page_config(page_title="Bengaluru House Price Prediction", page_icon="🏠", layout="centered")
st.markdown(
    """
    <style>
    .stApp {background-color: #0e1117; color: white;}
    .result-box {
        background-color: #135d36;
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---- Title ----
st.markdown("<h1 style='text-align: center;'>Bengaluru House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please enter a few details below to predict the house price: 🧑‍💻</p>", unsafe_allow_html=True)

# ---- Inputs ----
col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)

with col2:
    bath = st.number_input("Bath", min_value=1, max_value=10, value=2, step=1)

# Location dropdown full width
location = st.selectbox("Location", sorted(locations))

# Sqft input full width
sqft = st.number_input("Area in Square Foot", min_value=200, max_value=10000, value=1000, step=50)

# ---- Predict Button ----
if st.button("Estimate Price 💵"):
    x = np.zeros(len(data_columns))

    # numeric features
    x[data_columns.index('total_sqft')] = sqft
    x[data_columns.index('bath')] = bath
    x[data_columns.index('bhk')] = bhk

    # location one-hot encoding
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # prediction
    predicted_price = model.predict([x])[0]
    st.markdown(
        f"<div class='result-box'>The estimated price for your property is approximately ₹ {round(predicted_price, 2)} lakhs.</div>",
        unsafe_allow_html=True
    )
