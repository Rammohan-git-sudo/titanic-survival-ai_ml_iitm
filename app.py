# ==========================================
# TITANIC SURVIVAL WEB APP
# ==========================================

# Import libraries
import streamlit as st
import joblib


# ------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------

model = joblib.load("models/titanic_model.pkl")


# ------------------------------------------
# APP TITLE
# ------------------------------------------

st.title("Titanic Survival Prediction App")

st.write("Enter passenger details below:")


# ------------------------------------------
# USER INPUTS
# ------------------------------------------

# Passenger class
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

# Gender
sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

# Age
age = st.slider(
    "Age",
    1,
    80,
    25
)

# Fare
fare = st.number_input(
    "Fare",
    0.0,
    500.0,
    50.0
)

# Family size
family_size = st.slider(
    "Family Size",
    1,
    10,
    1
)

# Embarked
embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)


# ------------------------------------------
# CONVERT INPUTS TO NUMBERS
# ------------------------------------------

# Convert gender
sex_value = 0 if sex == "Male" else 1

# Convert embarked values
embarked_map = {
    "S": 0,
    "C": 1,
    "Q": 2
}

embarked_value = embarked_map[embarked]


# ------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------

if st.button("Predict Survival"):

    # Create input data
    data = [[
        pclass,
        sex_value,
        age,
        fare,
        family_size,
        embarked_value
    ]]

    # Make prediction
    prediction = model.predict(data)

    # Display result
    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")