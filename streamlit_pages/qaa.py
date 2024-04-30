import streamlit as st
from functions import QAA
import pandas as pd
from streamlit_extras.badges import badge
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stoggle import stoggle

def show_qaa():
    st.title(':violet[Cálculo de estrategias QAA]')
    st.write("Selecciona los parámetros con el objetivo de obtener los pesos para las estrategias QAA.")

    stoggle("Estrategias QAA usadas",
            """Mínima Varianza, Máximo Ratio de Sharpe, Semivarianza, Omega, Hierarchical Risk Parity (HRP), Conditional Value ar Risk (CVaR), Black Litterman, Famma French, Total Return AA, Roy Safety First Ratio, Sortino Ratio.""")
    
    add_vertical_space(5)

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        add_vertical_space(2)

        with col1:
            tickers = st.text_input("TICKERS", "A, B")
            rf = st.number_input("TASA LIBRE DE RIESGO (%)", value=0.0000)

        with col2:
            start_date_data = st.text_input("FECHA INICIAL (aaaa-mm-dd)", "2000-01-01")
            end_date_data = st.text_input("FECHA FINAL (aaaa-mm-dd)", "2024-01-01")

        with col3:
            lower_bound = st.number_input("LÍMITE INFERIOR", value=0.01)
            higher_bound = st.number_input("LÍMITE SUPERIOR", value=0.99)

        submitted = st.form_submit_button(":violet[CALCULA ESTRATEGIAS QAA]", type="secondary", use_container_width=True)
        
    add_vertical_space(5)

    if submitted:
        calculate_all_strategies(tickers, start_date_data, end_date_data, rf, lower_bound, higher_bound)

    add_vertical_space(5)

    badge(type="github", name="diegotita4/PAP")

def calculate_all_strategies(tickers, start_date, end_date, rf, lower_bound, higher_bound):
    optimization_models = ['SLSQP', 'Monte Carlo', 'COBYLA']
    strategies = ['Minimum Variance', 'Omega Ratio', 'Semivariance', 'Roy Safety First Ratio',
                  'Sortino Ratio', 'Fama French', 'CVaR', 'HRP', 'Sharpe Ratio',
                  'Black Litterman', 'Total Return']
    tickers_list = [ticker.strip() for ticker in tickers.split(',')]
    results = {}

    for strategy in strategies:
        results[strategy] = pd.DataFrame(columns=['MODELO', 'PESOS', 'RENDIMIENTO', 'VOLATILIDAD'])
        for model in optimization_models:
            qaa_instance = QAA(tickers=tickers_list, start_date=start_date, end_date=end_date, rf=rf, lower_bound=lower_bound, higher_bound=higher_bound)
            qaa_instance.set_optimization_strategy(strategy)
            qaa_instance.set_optimization_model(model)
            qaa_instance.optimize()
            if qaa_instance.optimal_weights is not None:
                formatted_weights = [f"{weight * 100:.2f}%" for weight in qaa_instance.optimal_weights]
                portfolio_return, portfolio_volatility = calculate_portfolio_metrics(qaa_instance)
                new_row = pd.DataFrame({
                    'MODELO': [model], 
                    'PESOS': [", ".join(formatted_weights)], 
                    'RENDIMIENTO': [f"{portfolio_return:.2f}%"], 
                    'VOLATILIDAD': [f"{portfolio_volatility:.2f}%"]
                })
            else:
                new_row = pd.DataFrame({
                    'MODELO': [model], 
                    'PESOS': ["No convergió o no implementado"], 
                    'RENDIMIENTO': ["N/A"], 
                    'VOLATILIDAD': ["N/A"]
                })
            results[strategy] = pd.concat([results[strategy], new_row], ignore_index=True)

    display_results(results)

def calculate_portfolio_metrics(qaa_instance):
    if hasattr(qaa_instance, 'calculate_portfolio_return') and hasattr(qaa_instance, 'calculate_portfolio_volatility'):
        portfolio_return = qaa_instance.calculate_portfolio_return()
        portfolio_volatility = qaa_instance.calculate_portfolio_volatility()
        return portfolio_return * 100, portfolio_volatility * 100
    else:
        raise AttributeError("QAA instance is missing required methods.")

def display_results(results):
    for strategy, data in results.items():
        st.subheader(f":violet[{strategy}]", divider="violet")
        st.table(data)
