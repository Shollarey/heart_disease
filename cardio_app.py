import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from preprocessing_utils import preprocessing
import joblib
import numpy as np

#Define Model Architecture
class CardioANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CardioANN, self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(64)
        self.hidden2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.hidden3 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(16, output_size)
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(self.relu(self.hidden3(x)))
        x = self.output(x)
        return x
    
#Load Pretrained Model & Scaler
@st.cache_resource
def load_model():
    model = CardioANN(16, 64, 2)
    model.load_state_dict(torch.load("cardio_ann_model_extended.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


# Sidebars 
def get_user_input():
    st.sidebar.title("🩺 Patient Profile")
    # st.sidebar.header("Patient Information")
    st.sidebar.markdown("Provide the following details:")

    # Objective Measurements
    age_years = st.sidebar.number_input("Age (years)", min_value=1, max_value=120 , value=40)
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender == "Female" else 2

    #Examinations
    ap_hi = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", 50, 250, 120)
    ap_lo = st.sidebar.number_input("Diatolic Blood Pressure (mmHg)", 30, 200, 80)
    cholesterol = st.sidebar.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {
        1: "Normal", 2: "Above normal", 3: "Well above normal"
        }[x])
    gluc = st.sidebar.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {
        1: "Normal", 2: "Above normal", 3: "Well above normal"
        }[x])

    #Subjective 
    smoke = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: {
        0: "No", 1: "Yes"
        }[x])

    alco = st.sidebar.selectbox("Alcohol Intake", [0, 1], format_func=lambda x:{
        0: "No", 1: "Yes"
        }[x])

    active = st.sidebar.selectbox("Physical Activity", [0, 1], format_func=lambda x: {
        0: "No", 1: "Yes"
        }[x])
    
    # Convert inputs to DataFrame
    age_days = age_years * 365
    input_dict = {
        "age": age_days,
        "height": height,
        "weight": weight,
        "gender": gender,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }
    df_input = pd.DataFrame([input_dict])
    return df_input

def predict(model, features):
    with torch.no_grad():
        outputs = model(features)
        _, pred = torch.max(outputs, 1)
        
        if pred.numel() == 1:
            return pred.item()
        else:
            return pred.tolist()

def main():
    st.set_page_config(
        page_title="Heart Disease Risk Predicitor ❤️",
        page_icon="❤️", 
        layout="wide",
        initial_sidebar_state="expanded")
    
    st.title("❤️ Cardiovascular Disease Prediction")
    st.write("""
    This app predicts the likelihood of **cardiovascular disease** using a machine learning model trained in PyTorch.  
    Fill in the patient’s details on the sidebar and click **Predict** to see the result.
    """)

    model = load_model()
    
    
    # Preprocess input
    processed_df, _ = preprocessing(get_user_input(), scaler=load_scaler(), fit_scaler=False)

    # Convert to torch tensor
    features = torch.tensor(processed_df.values, dtype=torch.float32)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if st.sidebar.button("Predict"):
        pred = predict(model, features)

        if pred == 1:
            st.error("🚨 **High Risk Detected!**\nConsult a healthcare provider for further evaluation.")
        else:
            st.success("✅ **Low Risk**\nKeep maintaining a healthy lifestyle!")
    else:
        st.info("Please fill out the form and click **Predict** to see the result.")


    st.markdown("---")
    st.write("### 🧾 Feature Documentation")
    st.markdown("""
    | Feature | Description | Type | Values |
    |----------|--------------|------|---------|
    | **Age** | Patient's age (in years) | int | 1–120 |
    | **Height** | Height in centimeters | int | 100–250 |
    | **Weight** | Weight in kilograms | float | 30–200 |
    | **Gender** | Biological sex | categorical | Female, Male |
    | **Systolic BP (ap_hi)** | Systolic blood pressure | int | 50–250 |
    | **Diastolic BP (ap_lo)** | Diastolic blood pressure | int | 30–200 |
    | **Cholesterol** | Cholesterol level | categorical | 1–3 |
    | **Glucose** | Glucose level | categorical | 1–3 |
    | **Smoking** | Currently smokes | binary | No, Yes |
    | **Alcohol Intake** | Consumes alcohol | binary | No, Yes |
    | **Physical Activity** | Active lifestyle | binary | No, Yes |
    """)

    st.caption("⚠️ *Disclaimer: This app is for educational purposes and not for medical diagnosis.*")

    st.markdown("---")
    st.markdown("""
    ### Documentation
    - **Framework:** PyTorch  
    - **Deployment:** Streamlit  
    - **Author:** Olusola Aremu  
    """)

    st.markdown("---")
    st.caption("Built with ❤️ by Olusola Aremu using Streamlit and PyTorch.")


if __name__ == "__main__":
    main()