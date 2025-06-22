import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from ibm_watson_machine_learning.foundation_models import Model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = "6dGe1BgS-aeVu1QM5zVT9QfpoNTeBoXNvY1Gfn_QHk1E"
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
MODEL_ID = "granite-13b-instruct-v2"

# Initialize IBM Granite Model
def init_granite_model():
    return Model(
        model_id=MODEL_ID,
        params={"decoding_method": "greedy"},
        credentials={"apikey": API_KEY, "project_id": PROJECT_ID}
    )

granite_model = init_granite_model()

# ----------------- Core Functionalities -------------------

def answer_patient_query(query):
    prompt = f"You are a helpful healthcare assistant. Answer the following question clearly and empathetically:\nQuestion: {query}"
    response = granite_model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

def predict_disease(symptoms):
    prompt = f"A patient reports the following symptoms: {symptoms}. Provide the most likely medical conditions and next steps."
    response = granite_model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

def generate_treatment_plan(condition, patient_info):
    prompt = f"Provide a personalized treatment plan for the condition '{condition}' based on this patient profile: {patient_info}. Include medication, lifestyle changes, and follow-up steps."
    response = granite_model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

def generate_health_insights(metrics_df):
    prompt = f"Analyze the following health metrics over time and give insights and recommendations:\n{metrics_df.to_string()}"
    response = granite_model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

# ----------------- Sample Data Utilities -------------------

def generate_sample_health_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=10)
    data = {
        "Date": dates,
        "Heart Rate": np.random.randint(65, 100, len(dates)),
        "Blood Pressure": np.random.randint(110, 140, len(dates)),
        "Blood Glucose": np.random.randint(80, 140, len(dates))
    }
    return pd.DataFrame(data)

# ----------------- UI Display Functions -------------------

def display_patient_chat():
    st.subheader("🩺 Patient Chat")
    query = st.text_input("Enter your medical question")
    if st.button("Ask") and query:
        with st.spinner("Fetching response..."):
            answer = answer_patient_query(query)
            st.success(answer)

def display_disease_prediction():
    st.subheader("🧬 Disease Prediction")
    symptoms = st.text_area("Describe your symptoms")
    if st.button("Predict Condition") and symptoms:
        with st.spinner("Analyzing symptoms..."):
            prediction = predict_disease(symptoms)
            st.success(prediction)

def display_treatment_plans():
    st.subheader("💊 Treatment Plan Generator")
    condition = st.text_input("Enter diagnosed condition")
    patient_info = st.text_area("Enter patient details (age, weight, other conditions, etc.)")
    if st.button("Generate Plan") and condition and patient_info:
        with st.spinner("Generating treatment plan..."):
            plan = generate_treatment_plan(condition, patient_info)
            st.success(plan)

def display_health_analytics():
    st.subheader("📊 Health Analytics Dashboard")
    data = generate_sample_health_data()
    metric = st.selectbox("Select Metric", data.columns[1:])
    fig = px.line(data, x="Date", y=metric, title=f"{metric} Over Time")
    st.plotly_chart(fig)

    if st.button("Get Health Insights"):
        with st.spinner("Analyzing trends..."):
            insights = generate_health_insights(data)
            st.info(insights)

# ----------------- Main Application -------------------

def main():
    st.set_page_config(page_title="AuraCare", layout="wide")
    st.title("🧠 AuraCare: Intelligent Healthcare Assistant")
    menu = ["Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics"]
    choice = st.sidebar.radio("Navigate", menu)

    if choice == "Patient Chat":
        display_patient_chat()
    elif choice == "Disease Prediction":
        display_disease_prediction()
    elif choice == "Treatment Plans":
        display_treatment_plans()
    elif choice == "Health Analytics":
        display_health_analytics()

if __name__ == "__main__":
    main()
