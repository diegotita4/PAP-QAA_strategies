import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import dynamic_backtesting  
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib.lines import Line2D
import plotly.express as px

def show_backtesting():
    st.title('Backtesting Page')
    st.write("Welcome to the QAA Strategy Backtesting.")

    # User inputs grouped by similarity and layout preferences
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tickers = st.text_input("Enter tickers separated by commas", "ABBV, MET, OXY, PERI")
            start_date_data = st.text_input("Start date for historical data", "2020-01-02")
            start_backtesting = st.text_input("Start date for backtesting", "2023-01-23")
            end_date = st.text_input("End date for backtesting", "2024-01-23")
        
        with col2:
            rebalance_frequency_months = st.slider("Rebalance frequency in months", 1, 12, 6)
            rf = st.number_input("Enter the risk-free rate", value=0.02)
            lower_bound= st.number_input("Enter the lower bound", value=0.10)
            higher_bound= st.number_input("Enter the higher  bound", value=0.99)
        
        with col3:
            optimization_strategy = st.selectbox("Choose the optimization strategy", 
                                                 ['Minimum Variance', 'Omega Ratio', 'Semivariance', 'Martingale', 
                                                  'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                                                  'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return'])
            optimization_model = st.selectbox("Choose the optimization model", ['SLSQP', 'Monte Carlo', 'COBYLA'])            
            initial_portfolio_value = st.number_input("Initial portfolio value", value=1000000)
            commission = st.number_input("Transaction commission", value=0.0025)

        submitted = st.form_submit_button("Run Backtesting")

    if submitted or 'resultados_backtesting' in st.session_state:
        if submitted:
            tickers_list = [ticker.strip() for ticker in tickers.split(',')]
            resultados_backtesting, daily_data, portfolio_values = dynamic_backtesting(
                tickers=tickers_list,
                start_date_data=start_date_data,
                start_backtesting=start_backtesting,
                end_date=end_date,
                lower_bound= lower_bound,
                higher_bound = higher_bound,
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
        else:
            resultados_backtesting = st.session_state['resultados_backtesting']
            daily_data = st.session_state['daily_data']
            portfolio_values = st.session_state['portfolio_values']

        show_rebalance_lines = st.checkbox("Show rebalance lines on graph")
        rebalance_dates = resultados_backtesting['end_date'].unique()

        st.write(f"Results for {optimization_strategy}:")
        st.dataframe(resultados_backtesting)

        # Plot the portfolio value with optional rebalance lines
        plot_portfolio_value(daily_data, portfolio_values, rebalance_dates, show_rebalance_lines)

            # Plot the last asset weights pie chart
        plot_asset_weights_pie_chart(resultados_backtesting, tickers_list)


def plot_asset_weights_pie_chart(resultados_backtesting, tickers):
    # Extract the last row for the latest weights
    last_weights = resultados_backtesting.iloc[-1]
    weights = [last_weights[f'weight_{ticker}'] for ticker in tickers]
    labels = tickers
    
    fig = px.pie(values=weights, names=labels, title='Latest Asset Weights in Portfolio')
    fig.update_traces(textposition='inside', textinfo='percent+label')
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
