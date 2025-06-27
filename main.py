import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from app.threatdetect import threatDetectFun
from app.about_page import about_page
from app.evaluation_app import show_evaluation_history
import random
from pathlib import Path

current_dir = Path(__file__).resolve().parent
FILEPATH = current_dir / "data" / "data.csv"



# Add custom CSS to style buttons
st.markdown("""
    <style>
    .stButton>button {
        background-color: transparent;
        color: black;
        border: none;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton>button:hover {
        color: black;
        cursor: pointer;
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: none;
        color: black;
    }

    .stButton > button:active {
        background-color: transparent !important;
        color: #111 !important;
        box-shadow: none !important;
    }
    .stAppDeployButton{
        visibility: hidden;
    }
    .st-emotion-cache-bm2z3a{
        align-items: flex-start;
        padding-left: 32px;        
    }
    .st-emotion-cache-1w723zb{
        max-width:1000px        
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"Could not load history: {e}")
        return pd.DataFrame()

def show_current(seed):
    threatDetectFun(seed)

def show_AboutPage():
    about_page()

def show_history():
    show_evaluation_history(FILEPATH)

def sideBar():
    # Sidebar menu
    st.sidebar.title("Dashboard")
    if "page" not in st.session_state:
        st.session_state.page = "About"

    # side bar buttons
    if st.sidebar.button("🔍 About"):
        st.session_state.page = "About"
    if st.sidebar.button("📈 History"):
        st.session_state.page = "History"
    if st.sidebar.button("📊 Predict"):
        st.session_state.page = "Current"

    # redirecting to those pages
    if st.session_state.page == "Current":
        n = random.randint(1,1000)
        show_current(n)
    elif st.session_state.page == "History":
        show_history()
    elif st.session_state.page == "About":
        show_AboutPage()

# Main page logic
def main():
    sideBar()

if __name__ == "__main__":
    main()




