import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib.metadata

from chatbot import get_chatbot_response
from mortgage_model import run_mortgage_model

print("openai version:", importlib.metadata.version("openai"))

# Run the DTI analysis and get image paths
output_paths = run_mortgage_model()

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/loan_data.csv")
    except Exception as e:
        np.random.seed(42)
        n = 500
        races = ['White', 'Black', 'Hispanic', 'Asian']
        data = pd.DataFrame({
            'applicant_race': np.random.choice(races, n),
            'loan_status': np.random.choice(['Approved', 'Denied'], n, p=[0.7, 0.3]),
            'credit_score': np.random.randint(300, 850, size=n)
        })
    return data

data = load_data()

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def show_home():
    st.title("Welcome to the Fair Housing Loan Approval Dashboard")
    st.write("Choose an option below to view visualizations or ask questions about the graphs:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Equality of Opportunity"):
            st.session_state.page = "Opportunity"
        if st.button("Policy Simulation"):
            st.session_state.page = "Simulation"
    with col2:
        if st.button("Equality of Outcome"):
            st.session_state.page = "Outcome"
        if st.button("Credit Score Analysis"):
            st.session_state.page = "CreditScore"
    with col3:
        if st.button("Graph Chatbot"):
            st.session_state.page = "Chatbot"
        if st.button("DTI Graphs"):
            st.session_state.page = "DTIGraphs"

def show_opportunity():
    st.header("Equality of Opportunity: Baseline Approval Rates")
    approval_data = data.groupby("applicant_race")["loan_status"].value_counts(normalize=True).unstack()
    if "Approved" in approval_data.columns:
        fig, ax = plt.subplots()
        ax.bar(approval_data.index, approval_data["Approved"], color="skyblue")
        ax.set_ylabel("Approval Rate")
        ax.set_xlabel("Applicant Race")
        ax.set_title("Baseline Approval Rates")
        st.pyplot(fig)
    else:
        st.write("Data does not include an 'Approved' status.")
    if st.button("Back to Home"):
        st.session_state.page = "Home"

def show_outcome():
    st.header("Equality of Outcome: Approval Rate Deviation")
    approval_data = data.groupby("applicant_race")["loan_status"].value_counts(normalize=True).unstack()
    overall_approval = data['loan_status'].value_counts(normalize=True).get("Approved", 0)
    deviations = approval_data["Approved"] - overall_approval
    fig, ax = plt.subplots()
    ax.bar(deviations.index, deviations, color="orchid")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Deviation from Overall Approval Rate")
    ax.set_xlabel("Applicant Race")
    ax.set_title("Approval Rate Deviation by Race")
    st.pyplot(fig)
    if st.button("Back to Home"):
        st.session_state.page = "Home"

def show_simulation():
    st.header("Policy Simulation: Adjust Approval Rates")
    approval_data = data.groupby("applicant_race")["loan_status"].value_counts(normalize=True).unstack()
    st.write("Simulate policy adjustments to address disparities in loan approvals.")
    adjust_race = st.selectbox("Select a race to adjust", options=data['applicant_race'].unique())
    adjust_pct = st.slider("Increase approval rate by (%)", 0, 50, 10)
    if adjust_race in approval_data.index and "Approved" in approval_data.columns:
        baseline_rate = approval_data.loc[adjust_race, "Approved"]
        simulated_rate = baseline_rate + (1 - baseline_rate) * (adjust_pct / 100)
        st.write(f"**Baseline approval rate for {adjust_race}:** {baseline_rate:.2%}")
        st.write(f"**Simulated approval rate for {adjust_race}:** {simulated_rate:.2%}")
        simulated_approval_data = approval_data.copy()
        simulated_approval_data.loc[adjust_race, "Approved"] = simulated_rate

        fig, ax = plt.subplots(figsize=(8, 6))
        width = 0.35
        indices = np.arange(len(approval_data.index))
        baseline_vals = approval_data["Approved"].values
        simulated_vals = simulated_approval_data["Approved"].values

        ax.bar(indices - width/2, baseline_vals, width, label="Baseline", color="skyblue")
        ax.bar(indices + width/2, simulated_vals, width, label="Simulated", color="salmon")
        ax.set_xticks(indices)
        ax.set_xticklabels(approval_data.index)
        ax.set_ylabel("Approval Rate")
        ax.set_title("Baseline vs. Simulated Approval Rates")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Unable to simulate policy adjustments due to data issues.")
    if st.button("Back to Home"):
        st.session_state.page = "Home"

def show_credit_score():
    st.header("Credit Score vs. Loan Approval")
    if "credit_score" in data.columns:
        fig, ax = plt.subplots()
        approved = data[data["loan_status"] == "Approved"]
        denied = data[data["loan_status"] == "Denied"]
        ax.scatter(approved["credit_score"], [1]*len(approved), color="green", label="Approved", alpha=0.5)
        ax.scatter(denied["credit_score"], [0]*len(denied), color="red", label="Denied", alpha=0.5)
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Loan Status (1 = Approved, 0 = Denied)")
        ax.set_title("Credit Score vs. Loan Approval Outcome")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No credit score data available.")
    if st.button("Back to Home"):
        st.session_state.page = "Home"

def show_chatbot():
    st.header("Graph Chatbot")
    st.write("Ask any questions about the graphs displayed in this dashboard. The chatbot will only answer questions related to these graphs.")
    question = st.text_input("Your question about the graphs:")
    if st.button("Ask"):
        if not question.strip():
            st.write("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer = get_chatbot_response(question)
            st.write("**Answer:**", answer)
    if st.button("Back to Home"):
        st.session_state.page = "Home"

def show_dti_graphs():
    st.header("DTI-Based Mortgage Analysis Results")

    st.subheader("Approval by Income Group")
    st.image(output_paths["income_bar"])

    st.subheader("Approval by Gender")
    st.image(output_paths["gender_bar"])

    st.subheader("Approval by Race")
    st.image(output_paths["race_bar"])

    st.subheader("Approval by Race & Gender")
    st.image(output_paths["race_gender_bar"])

    with open(output_paths["summary"], "r", encoding="utf-8") as f:
        summary = f.read()
        st.subheader("Analysis Summary")
        st.text(summary)

    if st.button("Back to Home"):
        st.session_state.page = "Home"

# Routing logic
if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "Opportunity":
    show_opportunity()
elif st.session_state.page == "Outcome":
    show_outcome()
elif st.session_state.page == "Simulation":
    show_simulation()
elif st.session_state.page == "CreditScore":
    show_credit_score()
elif st.session_state.page == "Chatbot":
    show_chatbot()
elif st.session_state.page == "DTIGraphs":
    show_dti_graphs()
