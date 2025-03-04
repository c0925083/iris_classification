import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('iris_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# UI for input
st.title("Iris Flower Classification")
st.write("Enter flower measurements to predict the species:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data = scaler.transform(input_data)
    
    # Predict class
    prediction = model.predict(input_data)[0]
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    st.write(f"Predicted Species: **{species[prediction]}**")
