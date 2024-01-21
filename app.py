# Bienvenidos al archivo de prueba para entender STREAMLIT una libreria que nos ayudará a lo largo del SEMESTRE para el PAP :)
#------------CODIGO DEL DASHBOARD------------------------------------------------------------------
# instala las librerias necesarias desde la consola bash
# Para instalar cualquier libreria en visual code hay que hacer lo siguiente. 

# 1. ACTIVAR ENTORNO VISUAL. 
# EN consola tipo BASH ESCRIBIR LA SIGUIENTE LINEA: python -m venv venv
# 2. Activar el entorno virtual
# En windows; .\venv\Scripts\activate
# En Mac; source venv/bin/activate
# pip install streamlit yfinance
# Reiniciar Visual code y listo! correr el codigo completo


# pip install streamlit
# pip install yfinance
# Importa las librerías necesarias

import streamlit as st 
import yfinance as yf 

# Título de la aplicación
st.title('Dashboard Financiero')

# Símbolo del activo financiero
symbol = st.text_input('Introduce el símbolo del activo financiero (por ejemplo, AAPL):')

# Descargar datos del activo financiero desde Yahoo Finance
if symbol:
    data = yf.download(symbol)
    st.write('Datos históricos del activo financiero:')
    st.write(data.head())

    # Gráfico interactivo
    st.line_chart(data['Close'])

# Información adicional
st.info('Este es un dashboard de prueba con Streamlit.')
