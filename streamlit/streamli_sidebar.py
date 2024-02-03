# functions/sidebar.py
import streamlit as st
import os

def create_sidebar():
    st.sidebar.title('Navegación')
    pages = [file.replace('.py', '') for file in os.listdir('page') if file.endswith('.py')]
    selected_page = st.sidebar.selectbox("Seleccione una página", pages)
    return selected_page