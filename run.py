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

# Importamos librerias
import streamlit as st
import yfinance as yf

# Título de la aplicación
st.title('Dashboard Financiero')

# Barra lateral principal con pestañas
selected_tab_main = st.sidebar.radio('Selecciona una sección:',
                                     ['Inicio', 'Datos Históricos'])

# Barra lateral secundaria con botones
selected_tab_other = st.sidebar.selectbox('Otra Navegación:',
                                          ['Sección 1', 'Sección 2', 'Sección 3'])

# Contenido principal
if selected_tab_main == 'Inicio':
    st.header('Bienvenido a nuestro Dashboard Financiero')
    st.write('Este es un dashboard de prueba con Streamlit.')

elif selected_tab_main == 'Datos Históricos':
    # Símbolo del activo financiero
    symbol = st.text_input('Introduce el símbolo del activo financiero (por ejemplo, AAPL):')

    # Descargar datos del activo financiero desde Yahoo Finance
    if symbol:
        data = yf.download(symbol)
        st.write('Datos históricos del activo financiero:')
        st.write(data.head())

        # Gráfico interactivo
        st.line_chart(data['Close'])

# Contenido de la otra barra lateral
if selected_tab_other == 'Sección 1':
    st.sidebar.header('Contenido de la Sección 1')
    st.sidebar.write('Este es el contenido de la Sección 1.')

elif selected_tab_other == 'Sección 2':
    st.sidebar.header('Contenido de la Sección 2')
    st.sidebar.write('Este es el contenido de la Sección 2.')

elif selected_tab_other == 'Sección 3':
    st.sidebar.header('Contenido de la Sección 3')
    st.sidebar.write('Este es el contenido de la Sección 3.')

## Documentacion de los diferentes widgets https://docs.streamlit.io/1.14.0/library/api-reference/widgets