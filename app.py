import streamlit as st
from streamlit_pages.home import show_home
from streamlit_pages.qaa import show_qaa
from streamlit_pages.backtesting import show_backtesting
# from streamlit_pages.config import show_config

def main():
    st.sidebar.title("Menú de Navegación")
    # Corrección: Incluir todas las opciones correctas aquí
    choice = st.sidebar.radio("Ir a", ("Inicio", "Cálculo QAA", "Visualización de Resultados"))

    if choice == "Inicio":
        show_home()
    elif choice == "Cálculo QAA":
        show_qaa()
    elif choice == "Visualización de Resultados":
        show_backtesting()

if __name__ == "__main__":
    main()

    