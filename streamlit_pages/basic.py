import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from backtest import dynamic_backtesting

list_strategy = ['Minimum Variance', 'Omega Ratio', 'Semivariance', #'Martingale', 
                 'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                 'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return']

def show_basic():
    st.title('Backtesting Page')
    st.write("Welcome to the QAA Strategy Backtesting.")

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
            initial_portfolio_value = st.number_input("Initial portfolio value", value=1000000)
            commission = st.number_input("Transaction commission", value=0.0025)

        submitted = st.form_submit_button("Run Backtesting")

        if submitted:
            strategy_results = {}
            progress_text = st.empty()  # Placeholder for dynamic text
            progress_bar = st.progress(0)
            num_strategies = len(list_strategy)

            for i, strategy in enumerate(list_strategy):
                tickers_list = [ticker.strip() for ticker in tickers.split(',')]
                resultados_backtesting, daily_data, portfolio_values = dynamic_backtesting(
                    tickers=tickers_list,
                    start_date_data=start_date_data,
                    start_backtesting=start_backtesting,
                    end_date=end_date,
                    rebalance_frequency_months=rebalance_frequency_months, 
                    rf=rf, 
                    optimization_strategy=strategy, 
                    optimization_model='SLSQP',  # Assuming a single model for simplicity
                    initial_portfolio_value=initial_portfolio_value,
                    commission=commission
                )
                strategy_results[strategy] = (resultados_backtesting, daily_data, portfolio_values)
                current_progress = (i + 1) / num_strategies
                progress_bar.progress(current_progress)
                progress_text.text(f"Processing: {int(current_progress * 100)}% Complete")  # Update text
            
            progress_text.text("All strategies have been processed. Displaying results...")
            sorted_results = sorted(strategy_results.items(), key=lambda item: item[1][0]["total_portfolio_value"].iloc[-1], reverse=True)

            # Plot all strategies' portfolio values
            plot_all_strategies(dict(sorted_results))

            # Display DataFrames and Highlight Total Portfolio Values
            for strategy, (result_df, daily_data, portfolio_values) in sorted_results:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"Results for {strategy}:")
                    st.dataframe(result_df)
                with col2:
                    st.metric(label="Total Portfolio Value", value=f"${result_df['total_portfolio_value'].iloc[-1]:,.2f}")

def plot_all_strategies(strategy_results):
    # Create a line plot for portfolio values
    line_fig = px.line()

    # We will first identify the best performing strategy based on the last 'total_portfolio_value'
    best_strategy = None
    highest_value = -float('inf')  # Initialize with a very low number

    for strategy, (result_df, daily_data, portfolio_values) in strategy_results.items():
        line_fig.add_scatter(x=daily_data.index, y=portfolio_values, mode='lines', name=strategy)
        # Determine if this strategy has the highest ending portfolio value
        if result_df['total_portfolio_value'].iloc[-1] > highest_value:
            highest_value = result_df['total_portfolio_value'].iloc[-1]
            best_strategy = (strategy, result_df)

    line_fig.update_layout(title="Portfolio Value Over Time by Strategy",
                           xaxis_title='Date',
                           yaxis_title='Portfolio Value ($)',
                           legend_title='Strategy')

    # Now plot only the pie chart for the best strategy
    if best_strategy:
        strategy_name, result_df = best_strategy
        last_weights = {col.split('_')[-1]: result_df[col].iloc[-1] for col in result_df.columns if col.startswith('weight_')}
        pie_labels = list(last_weights.keys())
        pie_values = list(last_weights.values())
        
        # Custom color palette for the pie chart
        custom_colors = px.colors.qualitative.Pastel1
        
        pie_fig = px.pie(names=pie_labels, values=pie_values, title=f'Asset Weights for {strategy_name}',
                         color_discrete_sequence=custom_colors)

    # Use more space for the line chart than for the pie chart
    col1, col2 = st.columns([3, 1])  # Adjusting column width proportions here
    with col1:
        st.plotly_chart(line_fig, use_container_width=True)
    with col2:
        if best_strategy:
            st.plotly_chart(pie_fig, use_container_width=True)
