import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import dynamic_backtesting  
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def show_backtesting():
    st.title('Resultados del Backtesting')
    st.write("Bienvenido a la demostración de resultados del Backtesting para estrategias QAA.")

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tickers = st.text_input("Tickers:", "ABBV, MET, OXY, PERI")
            start_date_data = st.text_input("Fecha histórica:", "2020-01-02")
            start_backtesting = st.text_input("Fecha inicio:", "2023-01-23")
            end_date = st.text_input("Fecha final:", "2024-01-23")
        
        with col2:
            rebalance_frequency_months = st.slider("Frecuencia de rebalanceo (M):", 1, 12, 6)
            rf = st.number_input("Tasa libre de riesgo:", value=0.02)
            lower_bound= st.number_input("Limite inferior:", value=0.10)
            higher_bound= st.number_input("Limite superior:", value=0.99)
        
        with col3:
            optimization_strategy = st.selectbox("Estrategia QAA:", 
                                                 ['Minimum Variance', 'Omega Ratio', 'Semivariance',
                                                  'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                                                  'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return'])
            optimization_model = st.selectbox("Modelo de optimización:", ['SLSQP', 'Monte Carlo', 'COBYLA'])            
            initial_portfolio_value = st.text_input("Valor del portafolio:", value='1,000,000')
            commission = st.number_input("Comisión por transacción:", value=0.0025, format="%.4f")
  
        submitted = st.form_submit_button("Ejecutar Backtesting...")

    if submitted:
        if not (validate_date(start_date_data) and validate_date(start_backtesting) and validate_date(end_date)):
            st.error("Por favor, ingrese fechas válidas en el formato YYYY-MM-DD.")
            return

        initial_portfolio_value = float(initial_portfolio_value.replace(',', ''))
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]

        resultados_backtesting, daily_data, portfolio_values = dynamic_backtesting(
            tickers=tickers_list,
            start_date_data=start_date_data,
            start_backtesting=start_backtesting,
            end_date=end_date,
            lower_bound=lower_bound,
            higher_bound=higher_bound,
            rebalance_frequency_months=rebalance_frequency_months, 
            rf=rf, 
            optimization_strategy=optimization_strategy, 
            optimization_model=optimization_model,
            initial_portfolio_value=initial_portfolio_value,
            commission=commission
        )

        st.session_state['resultados_backtesting'] = resultados_backtesting
        st.session_state['daily_data'] = daily_data
        st.session_state['portfolio_values'] = portfolio_values

    if 'resultados_backtesting' in st.session_state:
        show_rebalance_lines = st.checkbox("Mostrar líneas de rebalanceo en gráfico")

        if 'daily_data' in st.session_state and 'portfolio_values' in st.session_state:
            rebalance_dates = st.session_state['resultados_backtesting']['end_date'].unique()
            st.write(f"Resultados para {optimization_strategy}:")
            st.dataframe(st.session_state['resultados_backtesting'])

            # Plot the portfolio value with optional rebalance lines
            plot_portfolio_value(st.session_state['daily_data'], st.session_state['portfolio_values'], rebalance_dates, show_rebalance_lines)

        # Always plot the last asset weights pie chart
        plot_asset_weights_pie_chart(st.session_state['resultados_backtesting'], tickers_list)

def plot_asset_weights_pie_chart(resultados_backtesting, tickers):
    last_weights = resultados_backtesting.iloc[-1]
    weights = [last_weights[f'weight_{ticker}'] for ticker in tickers]
    labels = tickers
    
    fig = px.pie(values=weights, names=labels, title='Latest Asset Weights in Portfolio', 
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_value(daily_data, portfolio_values, rebalance_dates=None, show_rebalance_lines=False):
    fig = px.line(daily_data, x=daily_data.index, y=portfolio_values, labels={'y': 'Portfolio Value ($)', 'x': 'Date'},
                  title='Portfolio Value Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value ($)', legend_title='Legend')
    fig.add_scatter(x=daily_data.index, y=portfolio_values, mode='lines', name='Portfolio Value', line=dict(color='blue'))

    if show_rebalance_lines and rebalance_dates is not None:
        for date in rebalance_dates:
            fig.add_vline(x=date, line_width=2, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)
