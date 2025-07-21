import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# App title
st.set_page_config(page_title="Employee Salary Classification", layout="centered")
st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or ≤50K based on input features.")

# Input fields
st.header("Input Data")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Doctorate'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Exec-managerial', 'Craft-repair', 'Sales', 'Prof-specialty'])
hours_per_week = st.slider("Hours per week", min_value=1, max_value=100, value=40)
experience = st.number_input("Years of experience", min_value=0, max_value=50, value=5)

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("#### Preview Input:")
st.dataframe(input_df)

# Mock training data
def load_mock_data():
    data = {
        'age': np.random.randint(20, 60, 100),
        'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Doctorate'], 100),
        'occupation': np.random.choice(['Tech-support', 'Exec-managerial', 'Craft-repair', 'Sales', 'Prof-specialty'], 100),
        'hours-per-week': np.random.randint(20, 60, 100),
        'experience': np.random.randint(1, 30, 100),
        'salary': np.random.choice(['<=50K', '>50K'], 100)
    }
    return pd.DataFrame(data)

# Encode categorical variables
def preprocess(df):
    label_encoders = {}
    for col in ['education', 'occupation']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Train model
data = load_mock_data()
data, encoders = preprocess(data)

X = data.drop('salary', axis=1)
y = LabelEncoder().fit_transform(data['salary'])

model = RandomForestClassifier()
model.fit(X, y)

# Predict
if st.button("Predict Salary Class"):
    user_input = input_df.copy()
    for col in ['education', 'occupation']:
        user_input[col] = encoders[col].transform(user_input[col])
    prediction = model.predict(user_input)[0]
    label = ">50K" if prediction == 1 else "≤50K"
    st.success(f"Predicted Salary Class: **{label}**")
