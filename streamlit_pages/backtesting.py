import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import dynamic_backtesting  
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib.lines import Line2D

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
        
        with col3:
            optimization_strategy = st.selectbox("Choose the optimization strategy", 
                                                 ['Minimum Variance', 'Omega Ratio', 'Semivariance', 'Martingale', 
                                                  'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                                                  'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return'])
            optimization_model = st.selectbox("Choose the optimization model", ['SLSQP', 'Monte Carlo', 'COBYLA'])            
            initial_portfolio_value = st.number_input("Initial portfolio value", value=1000000)
            commission = st.number_input("Transaction commission", value=0.0025)

        submitted = st.form_submit_button("Run Backtesting")
        if submitted:
            tickers_list = [ticker.strip() for ticker in tickers.split(',')]
            resultados_backtesting, daily_data, portfolio_values = dynamic_backtesting(
                tickers=tickers_list,
                start_date_data=start_date_data,
                start_backtesting=start_backtesting,
                end_date=end_date,
                rebalance_frequency_months=rebalance_frequency_months, 
                rf=rf, 
                optimization_strategy=optimization_strategy, 
                optimization_model=optimization_model,
                initial_portfolio_value=initial_portfolio_value,
                commission=commission
            )
            # Display the resulting DataFrame
            st.write(resultados_backtesting)

            # Plot the portfolio value
            plot_portfolio_value(daily_data, portfolio_values)

            # Plot the last asset weights pie chart
            plot_asset_weights_pie_chart(resultados_backtesting, tickers_list)


def plot_portfolio_value(daily_data, portfolio_values):
    plt.figure(figsize=(7, 4))
    plt.plot(daily_data.index, portfolio_values, label='Portfolio Value', color='blue')  # Use the index directly
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def plot_asset_weights_pie_chart(resultados_backtesting, tickers):
    # Extract the last row for the latest weights
    last_weights = resultados_backtesting.iloc[-1]
    weights = [last_weights[f'weight_{ticker}'] for ticker in tickers]
    labels = tickers

    # Plot pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(weights, labels=labels, autopct='%1.1f%%')
    plt.title('Latest Asset Weights in Portfolio')
    st.pyplot(plt)