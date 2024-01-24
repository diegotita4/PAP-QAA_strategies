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
import pandas as pd
import importlib.util

import os
import sys
# print("Directorio actual:", os.getcwd())
# sys.path.insert(1, '../functions')
# print("Python Path:", sys.path)

# from sidebar import create_sidebar

def show():
    st.title("Página Principal")
    st.markdown("Aqui pondremos todo lo que debera estar en la pagina principal")



def create_sidebar():
    st.sidebar.title('Navegación')
    pages = [file.replace('.py', '') for file in os.listdir('page') if file.endswith('.py')]
    selected_page = st.sidebar.selectbox("Seleccione una página", pages)
    return selected_page


def load_page(page_name):
    # Construye el path al archivo .py en la carpeta page
    page_path = os.path.join('page', f'{page_name}.py')
    # Importa el módulo correspondiente al nombre de la página
    spec = importlib.util.spec_from_file_location(page_name, page_path)
    page_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page_module)
    # Ejecuta la función show() del módulo importado
    page_module.show()

# Esto debe estar al final de tu main.py
if __name__ == "__main__":
    selected_page = create_sidebar()
    load_page(selected_page)