import streamlit as st
import pandas as pd
from backtest import backtesting_dinamico  # Asegúrate de que el path sea correcto

def show_backtesting():
    st.title('Página de Backtesting')
    st.write("Bienvenido al Backtesting de estrategias QAA.")

    # Entradas del usuario
    tickers = st.text_input("Ingresa los tickers separados por comas", "ABBV, MET, OXY, PERI")
    rf = st.number_input("Ingresa la tasa libre de riesgo", value=0.02)
    optimization_strategy = st.selectbox("Elige la estrategia de optimización", 
                                         ['Minimum Variance', 'Maximum Sharpe Ratio', 'Equal Weights'])
    optimization_model = st.selectbox("Elige el modelo de optimización", ['SLSQP', 'Monte Carlo', 'L-BFGS-B'])
    valor_portafolio_inicial = st.number_input("Valor inicial del portafolio", value=1000000)
    frecuencias_rebalanceo_meses = st.slider("Frecuencia de rebalanceo en meses", 1, 12, 6)

    # Botón para realizar el backtesting
    if st.button('Realizar Backtesting'):
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
        resultados_backtesting = backtesting_dinamico(
            tickers=tickers_list,
            start_date='2020-01-02',
            start_backtesting='2023-01-23',
            end_date='2024-01-23',
            frecuencias_rebalanceo_meses=[frecuencias_rebalanceo_meses], 
            rf=rf, 
            optimization_strategy=optimization_strategy, 
            optimization_model=optimization_model,
            valor_portafolio_inicial=valor_portafolio_inicial
        )
        # Mostrar el DataFrame resultante
        st.write(resultados_backtesting)
