# ==========================================
# LOAD SAVED MODEL AND MAKE PREDICTION
# ==========================================

# Import joblib
import joblib


# ------------------------------------------
# STEP 1: LOAD TRAINED MODEL
# ------------------------------------------

model = joblib.load("models/titanic_model.pkl")

print("Model loaded successfully!")


# ------------------------------------------
# STEP 2: NEW PASSENGER DATA
# ------------------------------------------

# Feature order:
# Pclass
# Sex
# Age
# Fare
# FamilySize
# Embarked

new_passenger = [[
    3,      # Passenger class
    1,      # Female
    22,     # Age
    7.25,   # Fare
    2,      # Family size
    0       # Embarked (S)
]]


# ------------------------------------------
# STEP 3: MAKE PREDICTION
# ------------------------------------------

prediction = model.predict(new_passenger)


# ------------------------------------------
# STEP 4: DISPLAY RESULT
# ------------------------------------------

print("\n===== PREDICTION RESULT =====")

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")