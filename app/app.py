import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Load saved preprocessor and models
# Path:
# ElectricityBillPredictor/model/
# ----------------------------------------------------
preprocessor = joblib.load("../model/preprocessor.pkl")
bill_model = joblib.load("../model/bill_model.pkl")
unit_model = joblib.load("../model/unit_model.pkl")

# ----------------------------------------------------
# Streamlit page settings
# ----------------------------------------------------
st.set_page_config(page_title="Electricity Predictor", page_icon="⚡")
st.title("⚡ Electricity Bill & Unit Consume Predictor")
st.write("Enter your usage details to predict monthly electricity bill (₹) and unit consumption (kWh).")

# ----------------------------------------------------
# Inputs from user
# ----------------------------------------------------
people = st.number_input("Number of People", min_value=1, max_value=20, value=3)

house_size = st.selectbox("House Size", ["small", "medium", "large"])

ac_hours = st.slider("AC usage (hours per day)", 0, 12, 2)
fan_hours = st.slider("Fan usage (hours per day)", 0, 24, 8)

fridge = st.selectbox("Do you have a Fridge?", ["Yes", "No"])
fridge = 1 if fridge == "Yes" else 0

washing_machine = st.slider("Washing Machine usage (times per week)", 0, 10, 3)

tv_hours = st.slider("TV usage (hours per day)", 0, 10, 2)

laptop_hours = st.slider("Laptop usage (hours per day)", 0, 12, 5)

season = st.selectbox("Season", ["summer", "winter", "monsoon"])

# ----------------------------------------------------
# Prediction Button
# ----------------------------------------------------
if st.button("Predict ⚡"):

    # 1) Create input DataFrame (same format as dataset)
    input_data = pd.DataFrame([{
        "people": people,
        "house_size": house_size,
        "ac_hours": ac_hours,
        "fan_hours": fan_hours,
        "fridge": fridge,
        "washing_machine": washing_machine,
        "tv_hours": tv_hours,
        "laptop_hours": laptop_hours,
        "season": season
    }])

    # 2) Transform input using preprocessor
    transformed_data = preprocessor.transform(input_data)

    # 3) Predict Unit Consume
    predicted_units = unit_model.predict(transformed_data)[0]

    # 4) Predict Bill
    predicted_bill = bill_model.predict(transformed_data)[0]

    # 5) Show results
    st.success(f"✅ Predicted Monthly Unit Consumption: {predicted_units:.2f} kWh")
    st.success(f"✅ Predicted Monthly Electricity Bill: ₹ {predicted_bill:.2f}")

    # Bonus: show approx rate per unit
    if predicted_units > 0:
        per_unit_rate = predicted_bill / predicted_units
        st.info(f"💡 Approx rate per unit: ₹ {per_unit_rate:.2f} / unit")
