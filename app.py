# to run this app, use the command: python -m streamlit run app.py 



import streamlit as st      # pip install streamlit #used to create web apps for ML models
import pandas as pd
import joblib           # pip install joblib  # used to load the pickled model files


# ---------------------- PAGE CONFIGURATION ----------------------
st.set_page_config(
    page_title="Heart Stroke Predictor 💓",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = joblib.load('knn_heart_disease.pkl')
scaler = joblib.load('scaler_heart_disease.pkl')
columns = joblib.load('columns_heart_disease.pkl') 

st.title("Heart Stroke Prediction App")
st.markdown("This app predicts whether a person is likely to have a heart stroke or not.")
st.markdown("Please provide the necessary details below:")

age  = st.slider("Age", 1, 120, 25)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg):", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (in mg/dl):", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved:", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise):", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", ['Upsloping', 'Flat', 'Downsloping'])



if st.button("Predict"):
    raw_data = {
        'Age': age,
        'Sex': 1 if sex == 'M' else 0,
        'ChestPainType': chest_pain_type,         
        'RestingBP' : resting_bp,
        'Cholesterol': cholesterol,   
        'FastingBS': 1 if fasting_blood_sugar == 'Yes' else 0,
        'RestingECG': resting_ecg,   
        'MaxHR': max_heart_rate,
        'ExerciseAngina': 1 if exercise_induced_angina == 'Yes' else 0,
        'Oldpeak': oldpeak,   
        'ST_Slope': slope
    }

    input_df = pd.DataFrame([raw_data]) 

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0


    input_df = input_df[columns]         
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.error("The model predicts that the person is likely to have a heart stroke.")
    else:
        st.success("The model predicts that the person is unlikely to have a heart stroke.")



