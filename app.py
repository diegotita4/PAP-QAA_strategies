# Bienvenidos al archivo de prueba para entender STREAMLIT una libreria que nos ayudará a lo largo del SEMESTRE para el PAP :)
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


import streamlit as st
import yfinance as yf

# Título de la aplicación
st.title('Dashboard Financiero')

# Sidebar para navegación
st.sidebar.header('Navegación')
section = st.sidebar.radio('Selecciona una sección', ['Inicio', 'Datos Históricos'])

# Contenido principal
if section == 'Inicio':
    st.write('¡Bienvenido a la sección de inicio!')
    st.info('Este es un dashboard de prueba con Streamlit.')

elif section == 'Datos Históricos':
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
    st.info('Esta sección muestra datos históricos del activo financiero.')

