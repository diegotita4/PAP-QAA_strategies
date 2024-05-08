import streamlit as st

from streamlit_pages.qaa import show_qaa
from streamlit_pages.basic import show_basic
from streamlit_pages.backtesting import show_backtesting
# from streamlit_pages.config import show_config

# Importa correctamente la clase QAA desde tu módulo backtest
from backtest import QAA  

# Decorador para cachear datos; ensure data is only reloaded when necessary
@st.cache(allow_output_mutation=True, show_spinner=True)
def load_data(tickers, start_date, end_date):
    strategy = QAA(tickers=tickers, start_date=start_date, end_date=end_date)
    strategy.load_data()  # Asume que esta función es la responsable de cargar los datos
    return strategy

def main():
    st.sidebar.title("MENÚ DE NAVEGACIÓN")
    # Lista de opciones en el menú lateral
    choice = st.sidebar.radio(" ", ("Cálculo de estrategias QAA", "Backtesting individual", "Backtesting general"))

    # Navegación entre diferentes páginas en la aplicación
    if choice == "Cálculo de estrategias QAA":
        show_qaa()
    elif choice == "Backtesting individual":
        show_backtesting()
    elif choice == 'Backtesting general':
        show_basic()

    # Botones de navegación en el menú lateral
    #if st.sidebar.button(":violet[Cálculo de estrategias QAA]", type="secondary", use_container_width=True):
        #show_qaa()

    #if st.sidebar.button(":violet[Backtesting individual]", type="secondary", use_container_width=True):
        #show_backtesting()

    #if st.sidebar.button(":violet[Backtesting general]", type="secondary", use_container_width=True):
        #show_basic()

if __name__ == "__main__":
    main()