import streamlit as st
from functions import QAA
import pandas as pd

def show_qaa():
    st.title('Análisis de Estrategias de Optimización de Cartera')
    st.write("Selecciona los parámetros y calcula los pesos óptimos para diferentes estrategias y modelos de optimización.")

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tickers = st.text_input("Tickers:", "ABBV, MET, OXY, PERI")
            start_date_data = st.text_input("Fecha de inicio:", "2020-01-02")
            end_date_data = st.text_input("Fecha de término:", "2024-01-23")

        with col2:
            rf = st.number_input("Tasa libre de riesgo:", value=0.02)

        with col3:
            lower_bound = st.number_input("Límite inferior:", value=0.10)
            higher_bound = st.number_input("Límite superior:", value=0.99)

        submitted = st.form_submit_button("Calcular Todas las Estrategias")

    if submitted:
        calculate_all_strategies(tickers, start_date_data, end_date_data, rf, lower_bound, higher_bound)

def calculate_portfolio_metrics(qaa_instance):
    portfolio_return = qaa_instance.calculate_portfolio_return()
    portfolio_volatility = qaa_instance.calculate_portfolio_volatility()
    return portfolio_return, portfolio_volatility



def calculate_all_strategies(tickers, start_date, end_date, rf, lower_bound, higher_bound):
    optimization_models = ['SLSQP', 'Monte Carlo', 'COBYLA']
    strategies = ['Minimum Variance', 'Omega Ratio', 'Semivariance', 'Roy Safety First Ratio',
                  'Sortino Ratio', 'Fama French', 'CVaR', 'HRP', 'Sharpe Ratio',
                  'Black Litterman', 'Total Return']
    tickers_list = [ticker.strip() for ticker in tickers.split(',')]
    results = {}

    for strategy in strategies:
        results[strategy] = pd.DataFrame(columns=['Modelo', 'Pesos Óptimos', 'Rendimiento Estimado', 'Volatilidad Estimada'])
        for model in optimization_models:
            qaa_instance = QAA(tickers=tickers_list, start_date=start_date, end_date=end_date, rf=rf, lower_bound=lower_bound, higher_bound=higher_bound)
            qaa_instance.set_optimization_strategy(strategy)
            qaa_instance.set_optimization_model(model)
            qaa_instance.optimize()
            if qaa_instance.optimal_weights is not None:
                formatted_weights = [f"{weight * 100:.2f}%" for weight in qaa_instance.optimal_weights]
                portfolio_return, portfolio_volatility = calculate_portfolio_metrics(qaa_instance)
                results[strategy] = results[strategy].append({'Modelo': model, 'Pesos Óptimos': ", ".join(formatted_weights), 'Rendimiento Estimado': f"{portfolio_return:.2f}%", 'Volatilidad Estimada': f"{portfolio_volatility:.2f}%"}, ignore_index=True)
            else:
                results[strategy] = results[strategy].append({'Modelo': model, 'Pesos Óptimos': "No convergió o no implementado", 'Rendimiento Estimado': "N/A", 'Volatilidad Estimada': "N/A"}, ignore_index=True)

    display_results(results)

def calculate_portfolio_metrics(qaa_instance):
    # Asigna valores de rendimiento y volatilidad a partir de la instancia QAA, asegúrate de que estos métodos estén implementados
    portfolio_return = qaa_instance.calculate_portfolio_return()  # Debe ser implementado en QAA
    portfolio_volatility = qaa_instance.calculate_portfolio_volatility()  # Debe ser implementado en QAA
    return portfolio_return * 100, portfolio_volatility * 100

def display_results(results):
    for strategy, data in results.items():
        st.subheader(f"{strategy}")
        st.table(data) 

