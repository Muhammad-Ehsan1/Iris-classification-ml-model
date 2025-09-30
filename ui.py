import streamlit as st
import pickle
import json
import numpy as np

# --- Load model ---
model = pickle.load(open("logistic_regression.pkl", "rb"))

# --- Load columns.json ---
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Features
numeric_features = data_columns[:4]  

# --- Streamlit UI ---
st.set_page_config(page_title="Iris Classification", page_icon="🌷", layout="centered")

st.markdown(
    """
    <h2 style="text-align:center; color:#2E86C1;">🌷 Iris Classification App</h2>
    <p style="text-align:center; color:gray;">Enter flower details for classification</p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# Inputs
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("sepal_length", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    sepal_width = st.number_input("sepal_width", min_value=0.0, max_value=10.0, step=0.1, value=1.0)

with col2:
    petal_length = st.number_input("petal_length", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    petal_width = st.number_input("petal_width", min_value=0.0, max_value=10.0, step=0.1, value=1.0)

st.write("---")

# Predict Button
if st.button("🔮 Predict Iris Type", use_container_width=True):
    # Prepare input vector
    x = np.zeros(len(data_columns))
    x[0] = sepal_length
    x[1] = sepal_width
    x[2] =  petal_length
    x[3] = petal_width
  
    # Predict
    prediction = model.predict([x])[0]

    st.success(f"🌷 Predictic type is : {prediction}")
    st.balloons()
    if prediction == "setosa":
        st.image("sentosa.jpg",caption="Setosa",width=200)
    if prediction == "virginica":
        st.image("verginica.jpg",caption="virginica",width=200)
    if prediction == "versicolor":
        st.image("versicolor.jpeg",caption="versicolor",width=200)
    
 