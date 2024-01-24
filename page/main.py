#------------CODIGO DEL DASHBOARD------------------------------------------------------------------
# instala las librerias necesarias desde la consola bash
# Para instalar cualquier libreria en visual code hay que hacer lo siguiente. 

# 1. ACTIVAR ENTORNO VISUAL. 
# En windows tienes que meterte a la ruta del proyecto. cd ruta/del/proyecto / <-------- esas lineas (/) al reves
# EN consola tipo BASH ESCRIBIR LA SIGUIENTE LINEA: python -m venv venv
# 2. Activar el entorno virtual
# En windows; .\venv\Scripts\activate
# En Mac; source venv/bin/activate
# pip install streamlit yfinance
# Reiniciar Visual code y listo! correr el codigo completo

# Importamos librerias
import streamlit as st
import yfinance as yf

import sys
sys.path.insert(1, '../functions')  # Asegúrate de que el directorio functions esté en el PATH
from sidebar import create_sidebar


def page1():
    st.title("Página 1")
    st.write("Bienvenido a la página 1")


if __name__ == "__main__":
    selected_page = create_sidebar()
    if selected_page == 'page1':
        page1()