import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from backtest import dynamic_backtesting
import random

list_strategy = ['Minimum Variance', 'Omega Ratio', 'Semivariance',
                 'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                 'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return']

def validate_date(date_text):
    try:
        return datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        return None

def parse_int_from_string(input_string):
    try:
        return int(input_string.replace(',', ''))
    except ValueError:
        return None

@st.cache_data(show_spinner=False)
def run_backtesting(tickers, start_date, start_backtesting, end_date, frequency, rf, initial_value, commission):
    strategy_results = {}
    for i, strategy in enumerate(list_strategy):
        results, daily_data, portfolio_values = dynamic_backtesting(
            tickers=tickers,
            start_date_data=start_date,
            start_backtesting=start_backtesting,
            end_date=end_date,
            rebalance_frequency_months=frequency,
            rf=rf,
            optimization_strategy=strategy,
            optimization_model='SLSQP',
            initial_portfolio_value=initial_value,
            commission=commission
        )
        strategy_results[strategy] = (results, daily_data, portfolio_values)
    return strategy_results

def display_results(strategy_results, show_all_strategies):
    sorted_strategies = sorted(strategy_results.items(), key=lambda x: x[1][2][-1], reverse=True)
    if show_all_strategies:
        strategies_to_display = sorted_strategies
    else:
        strategies_to_display = random.sample(sorted_strategies, 4)

    for strategy, (result_df, daily_data, portfolio_values) in strategies_to_display:
        st.markdown(f"### {strategy}")
        st.dataframe(result_df.assign(total_portfolio_value=lambda x: x['total_portfolio_value'].apply("${:,.2f}".format)))
        plot_strategy_performance(daily_data, portfolio_values, strategy)
        final_value = portfolio_values[-1]
        st.metric(label="Final Portfolio Value", value=f"${final_value:,.2f} (USD)")
        st.markdown("---")

def plot_strategy_performance(daily_data, portfolio_values, strategy_name):
    colors = px.colors.qualitative.Plotly
    strategy_index = list_strategy.index(strategy_name) % len(colors)
    fig = px.line(
        x=daily_data.index, 
        y=portfolio_values, 
        title=f"{strategy_name} - Valor del Portafolio a lo largo del Backtesting",
        labels={"x": "Date", "y": "Valor del Portafolio ($)"},
        line_shape='linear'
    )
    fig.update_traces(line=dict(color=colors[strategy_index]))
    st.plotly_chart(fig, use_container_width=True)

def show_basic():
    st.title('Backtesting Sencillo (Un método)')
    show_all_strategies = st.checkbox("Mostrar más estrategias")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tickers = st.text_input("Tickers:", "ABBV, MET, OXY, PERI")
            start_date_data = st.text_input("Fecha histórica:", "2020-01-02")
            start_backtesting = st.text_input("Fecha inicio:", "2023-01-23")
            end_date = st.text_input("Fecha final:", "2024-01-23")
        with col2:
            rebalance_frequency_months = st.slider("Frecuencia de rebalanceo (M):", 1, 12, 6)
            rf = st.number_input("Tasa libre de riesgo:", value=0.02, format="%.4f")
        with col3:
            initial_portfolio_value = st.text_input("Valor del portafolio:", value='1,000,000')
            commission = st.number_input("Comisión por transacción:", value=0.0025, format="%.4f")

        submitted = st.form_submit_button("Ejecutar Backtesting....")

    if submitted:
        if not (validate_date(start_date_data) and validate_date(start_backtesting) and validate_date(end_date)):
            st.error("Ingresa valores válidos en el siguiente formato: YYYY-MM-DD")
            return

        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
        if len(tickers_list) < 2:
            st.error("Ingrese por lo menos 2 tickers válidos.")
            return

        parsed_initial_value = parse_int_from_string(initial_portfolio_value)
        if parsed_initial_value is None:
            st.error("Ingresa un número válido para el Valor inicial del Portafolio.")
            return

        strategy_results = run_backtesting(tickers_list, start_date_data, start_backtesting, end_date,
                                           rebalance_frequency_months, rf, parsed_initial_value, commission)
        display_results(strategy_results, show_all_strategies)

show_basic()
