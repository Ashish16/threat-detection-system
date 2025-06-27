import streamlit as st
from pathlib import Path

current_dir = Path(__file__).resolve().parent
html_path = current_dir / "templates" / "about_page.html"

def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def about_page():
    html_content = load_html(html_path)
    st.markdown(html_content, unsafe_allow_html=True)

# Call your function
# about_page()
