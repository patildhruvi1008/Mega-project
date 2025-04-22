import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("models/branch_predictor.pkl", "rb"))

st.title("Engineering Branch Predictor")
st.write("Enter your academic details to predict your most suitable branch.")

# User inputs
category = st.selectbox("Category", options=[0, 1, 2, 3])  # Adjust if your model uses encoded categories
jee_marks = st.number_input("JEE Marks", min_value=0, max_value=300)
tenth_marks = st.number_input("10th Marks (%)", min_value=0.0, max_value=100.0)
twelfth_marks = st.number_input("12th Marks (%)", min_value=0.0, max_value=100.0)

# Prediction
if st.button("Predict Branch"):
    input_data = pd.DataFrame([[category, jee_marks, tenth_marks, twelfth_marks]],
                              columns=['Category', 'JEE Marks', '10th Marks', '12th Marks'])
    prediction = model.predict(input_data)[0]
    st.success(f"🎓 Predicted Branch: {prediction}")
