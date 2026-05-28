# ==========================================
# TITANIC SURVIVAL PREDICTION PROJECT
# ==========================================

# ------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------------------------------
# STEP 2: LOAD DATASET
# ------------------------------------------

df = pd.read_csv("data/train.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())


# ------------------------------------------
# STEP 3: HANDLE MISSING VALUES
# ------------------------------------------

# Fill missing Age values using median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values using mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Cabin column
df.drop("Cabin", axis=1, inplace=True)


# ------------------------------------------
# STEP 4: CONVERT CATEGORICAL DATA
# ------------------------------------------

# Convert Sex column
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Convert Embarked column
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})


# ------------------------------------------
# STEP 5: FEATURE ENGINEERING
# ------------------------------------------

# Create FamilySize feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# ------------------------------------------
# STEP 6: SELECT FEATURES AND TARGET
# ------------------------------------------

# IMPORTANT:
# We are using ONLY 6 features

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "Embarked"
]

# Input features
X = df[features]

# Target variable
y = df["Survived"]


# ------------------------------------------
# STEP 7: SPLIT DATA
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------
# STEP 8: CREATE MODEL
# ------------------------------------------

model = LogisticRegression(max_iter=1000)


# ------------------------------------------
# STEP 9: TRAIN MODEL
# ------------------------------------------

model.fit(X_train, y_train)

print("\n===== MODEL TRAINING COMPLETED =====")


# ------------------------------------------
# STEP 10: MAKE PREDICTIONS
# ------------------------------------------

predictions = model.predict(X_test)


# ------------------------------------------
# STEP 11: CHECK ACCURACY
# ------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL ACCURACY =====")
print("Accuracy:", accuracy)


# ------------------------------------------
# STEP 12: SAVE MODEL
# ------------------------------------------

joblib.dump(model, "models/titanic_model.pkl")

print("\n===== MODEL SAVED SUCCESSFULLY =====")