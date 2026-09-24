import streamlit as st
import pandas as pd
import joblib 

# Load saved files 
lr_model = joblib.load("linear_regression.joblib")
dt_model = joblib.load("decision_tree.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")

# Read Dataset (Only for Structure)
df = pd.read_csv("diamonds.csv")

cat_cols = ["cut", "color", "clarity"]

# Streamlit UI
st.set_page_config(page_title="Diamond Price Prediction App", page_icon="💎", layout="wide")
st.title(" 💎 Diamond Price Prediction App")
model = st.sidebar.selectbox("Select Model", ["Linear Regression", "Decision Tree"])

st.subheader("Enter Diamond Features")

carat = st.number_input("Carat", min_value=0.0, max_value=5.0)
cut = st.selectbox("Cut", sorted(df["cut"].unique()))
color = st.selectbox("Color", sorted(df["color"].unique()))
clarity = st.selectbox("Clarity", sorted(df["clarity"].unique()))
depth = st.number_input("Depth", value=61.5)
table = st.number_input("Table", value=57.0)
x = st.number_input("X (Length in mm)", value=5.5)
y = st.number_input("Y (Width in mm)", value=5.5)
z = st.number_input("Z (Depth in mm)", value=3.5)

# Prediction 
if st.button("Predict Price"):
    # Create DataFrame for input features
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    # Encode categorical features
    input_data_encoded = encoder.transform(input_data[cat_cols])
    
    # Handle array output from OneHotEncoder/OrdinalEncoder safely
    if hasattr(input_data_encoded, "toarray"):
        input_data_encoded = input_data_encoded.toarray()
        
    input_data_encoded_df = pd.DataFrame(
        input_data_encoded, 
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Separate numerical features by dropping categorical columns once
    input_numeric = input_data.drop(columns=cat_cols)
    
    # Combine encoded categorical features with numerical features
    input_data_final = pd.concat([input_numeric, input_data_encoded_df], axis=1)

    # Scale the features
    input_data_scaled = scaler.transform(input_data_final)

    # Make prediction based on selected model
    if model == "Linear Regression":
        prediction = lr_model.predict(input_data_scaled)
    else:
        prediction = dt_model.predict(input_data_scaled)

    st.success(f"Predicted Diamond Price: ${prediction[0]:,.2f}")