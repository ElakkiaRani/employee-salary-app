import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Employee Salary Prediction")

# Load sample data
df = pd.read_csv("employee_data.csv")
st.write("### Dataset", df)

# Train model
X = df[["YearsExperience"]]
y = df["Salary"]
model = LinearRegression()
model.fit(X, y)

# Input
experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
