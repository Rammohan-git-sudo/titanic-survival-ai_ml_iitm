# 🚢 Titanic Survival Prediction AI App

## 📌 Project Overview

This project is an end-to-end Machine Learning classification application that predicts whether a passenger survived the Titanic disaster based on passenger details.

The project covers the complete AI engineering workflow:

* Data preprocessing
* Feature engineering
* Machine Learning model training
* Model evaluation
* Model persistence
* Streamlit web application
* GitHub integration
* Cloud deployment

---

# 🎯 Problem Statement

Predict whether a passenger survived or not using historical Titanic passenger data.

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 0     | Did Not Survive |
| 1     | Survived        |

This is a **Binary Classification Problem**.

---

# 🛠️ Technologies Used

| Technology      | Purpose              |
| --------------- | -------------------- |
| Python          | Programming Language |
| Pandas          | Data Processing      |
| Scikit-learn    | Machine Learning     |
| Streamlit       | Web Application      |
| Joblib          | Model Saving/Loading |
| Git & GitHub    | Version Control      |
| Streamlit Cloud | Deployment           |

---

# 📂 Project Structure

```text
titanic_ai_project/
│
├── data/
│   └── train.csv
│
├── models/
│   └── titanic_model.pkl
│
├── notebooks/
│
├── main.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Machine Learning Workflow

## 1. Data Loading

Loaded Titanic dataset using Pandas.

## 2. Data Cleaning

* Filled missing Age values using median
* Filled missing Embarked values using mode
* Removed Cabin column due to excessive missing values

## 3. Feature Engineering

Created a new feature:

```python
FamilySize = SibSp + Parch + 1
```

## 4. Feature Encoding

Converted categorical values into numerical format.

### Sex Encoding

* Male → 0
* Female → 1

### Embarked Encoding

* S → 0
* C → 1
* Q → 2

## 5. Train-Test Split

Used:

```python
test_size=0.2
random_state=42
```

## 6. Model Training

Trained Logistic Regression classification model.

## 7. Model Evaluation

Evaluated using Accuracy Score.

## 8. Model Saving

Saved trained model using Joblib.

---

# 🤖 Features Used for Prediction

* Passenger Class
* Sex
* Age
* Fare
* Family Size
* Embarked Location

---

# 💻 Streamlit Web Application

Built an interactive AI web application using Streamlit.

### Features

* Dropdown menus
* Sliders
* Real-time prediction
* User-friendly interface

---

# 🚀 Running the Project Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🌍 Deployment

The project is deployed using Streamlit Community Cloud.

Deployment Workflow:

```text
GitHub Repository
        ↓
Streamlit Cloud
        ↓
Dependency Installation
        ↓
Model Loading
        ↓
Public AI Web Application
```

---

# 📈 Future Improvements

* Random Forest / XGBoost models
* Better UI design
* Probability score display
* Docker deployment
* FastAPI backend
* Cloud deployment using AWS/GCP
* MLOps pipeline integration

---

# 📚 Key Concepts Learned

* Classification
* Feature Engineering
* Train-Test Split
* Logistic Regression
* Model Persistence
* Streamlit Deployment
* GitHub Workflow
* AI Application Deployment

---

# 👨‍💻 Author

**Rammohan Ijjada**

Aspiring AI Engineer passionate about:

* Machine Learning
* AI Engineering
* Model Deployment
* Real-world AI Applications

---
