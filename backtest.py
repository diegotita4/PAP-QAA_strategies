import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
# pip install optuna
import optuna
import warnings
from matplotlib.lines import Line2D
import logging
import matplotlib.pyplot as plt
from functions import QAA

def dynamic_backtesting(tickers, start_date_data, start_backtesting, end_date, rebalance_frequency_months, rf, optimization_strategy, optimization_model, initial_portfolio_value, commission=0.0025):
    start_date = pd.to_datetime(start_backtesting)
    end_date = pd.to_datetime(end_date)
    portfolio_value = initial_portfolio_value
    previous_num_shares = pd.Series(0, index=tickers)
    results = []
    
    # Initial process at start_backtesting
    current_date = start_date
    rebalance_end_date = current_date   
    if rebalance_end_date > end_date:
        rebalance_end_date = end_date
    
    strategy = QAA(
        tickers=tickers,
        start_date=start_date_data,  # Historical data from start_date
        end_date=rebalance_end_date.strftime('%Y-%m-%d'),
        rf=rf
    )
    strategy.set_optimization_strategy(optimization_strategy)
    strategy.set_optimization_model(optimization_model)
    strategy.load_data()
    strategy.optimize()
    
    optimal_weights = strategy.optimal_weights
    
    investment_value_per_ticker = portfolio_value * optimal_weights
    # Calculate new price adjusted by the commission
    current_prices = strategy.data.iloc[-1]
    adjusted_prices = current_prices * (1 + commission)

    # Calculate number of shares based on the adjusted price
    num_shares = (investment_value_per_ticker / adjusted_prices).apply(np.floor)
    invested_value = num_shares * current_prices  # We use the original price to calculate the actual invested value
    remaining_cash = portfolio_value - invested_value.sum()

    
    previous_num_shares = num_shares.copy()
    portfolio_value = invested_value.sum() + remaining_cash
    
    result_row = {
        'data_origin_date': start_date_data,
        'end_date': rebalance_end_date.strftime('%Y-%m-%d'),
        **{f'weight_{ticker}': optimal_weights[i] for i, ticker in enumerate(tickers)},
        **{f'shares_{ticker}': num_shares[ticker] for ticker in tickers},
        **{f'value_{ticker}': invested_value[ticker] for ticker in tickers},
        'remaining_cash': remaining_cash,
        'total_portfolio_value': portfolio_value
    }
    
    results.append(result_row)
    
    current_date = rebalance_end_date
    while current_date <= end_date:
        rebalance_end_date = current_date + relativedelta(months=rebalance_frequency_months)
        if rebalance_end_date > end_date:
            rebalance_end_date = end_date
        
        strategy = QAA(
            tickers=tickers, 
            start_date=start_date_data, 
            end_date=rebalance_end_date.strftime('%Y-%m-%d'), 
            rf=rf
        )
        strategy.set_optimization_strategy(optimization_strategy)
        strategy.set_optimization_model(optimization_model)
        strategy.load_data()
        strategy.optimize()
        
        current_prices = strategy.data.iloc[-1]
        optimal_weights = strategy.optimal_weights
        
        if not previous_num_shares.equals(pd.Series(0, index=tickers)):
            portfolio_value = (previous_num_shares * current_prices).sum()

        investment_value_per_ticker = portfolio_value * optimal_weights
    
        # Adjust prices for commission
        adjusted_prices = current_prices * (1 + commission)
        num_shares = (investment_value_per_ticker / adjusted_prices).apply(np.floor)
        invested_value = num_shares * current_prices
        remaining_cash = portfolio_value - invested_value.sum()
        
        diff_shares = num_shares - previous_num_shares
        
        previous_num_shares = num_shares.copy()
        portfolio_value = invested_value.sum() + remaining_cash

        result_row = {
            'data_origin_date': start_date_data,
            'end_date': rebalance_end_date.strftime('%Y-%m-%d'),
            **{f'weight_{ticker}': optimal_weights[i] for i, ticker in enumerate(tickers)},
            **{f'shares_{ticker}': num_shares[ticker] for ticker in tickers},
            **{f'diff_{ticker}': diff_shares[ticker] for ticker in tickers},
            **{f'value_{ticker}': invested_value[ticker] for ticker in tickers},
            'remaining_cash': remaining_cash,
            'total_portfolio_value': portfolio_value
        }
        
        results.append(result_row)
        current_date = rebalance_end_date
        if rebalance_end_date == end_date:
            break
    results = pd.DataFrame(results)

    # Daily Data Preparation

    # Prepare the dataframe with dynamic column names for each ticker
    df_columns = ['end_date'] + [f'shares_{ticker}' for ticker in tickers] + ['remaining_cash']
    df = results[df_columns].copy()
    df['end_date'] = pd.to_datetime(df['end_date'])
    df.set_index('end_date', inplace=True)

    # Generate a date range for the daily data
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a new dataframe with a record for each day, forward-filling the rebalancing information
    daily_data = pd.DataFrame(index=date_range)
    daily_data = daily_data.join(df, how='left').ffill().reset_index().rename(columns={'index': 'end_date'})

    # Fetch historical stock data using yfinance for the entire range of daily_data
    stock_data = yf.download(tickers, start=daily_data['end_date'].min(), end=daily_data['end_date'].max())
    prices = stock_data['Adj Close']

    # Update the index of daily_data to match the dates from the stock data
    daily_data.set_index('end_date', inplace=True)
    daily_data = daily_data.reindex(prices.index).ffill()

    # Calculate the daily portfolio value
    portfolio_value = daily_data['remaining_cash']
    for ticker in tickers:
        portfolio_value += daily_data[f'shares_{ticker}'] * prices[ticker]

    return results, daily_data, portfolio_value

#-------------------------

def plot_portfolio_value(resultados_backtesting, tickers):
    # Prepare the dataframe with dynamic column names for each ticker
    df_columns = ['end_date'] + [f'shares_{ticker}' for ticker in tickers] + ['remaining_cash']
    df = resultados_backtesting[df_columns].copy()
    df['end_date'] = pd.to_datetime(df['end_date'])
    df.set_index('end_date', inplace=True)

    # Generate a date range for the daily data
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a new dataframe with a record for each day, forward-filling the rebalancing information
    daily_data = pd.DataFrame(index=date_range)
    daily_data = daily_data.join(df, how='left').ffill().reset_index().rename(columns={'index': 'end_date'})

    # Fetch historical stock data using yfinance for the entire range of daily_data
    stock_data = yf.download(tickers, start=daily_data['end_date'].min(), end=daily_data['end_date'].max())
    prices = stock_data['Adj Close']

    # Update the index of daily_data to match the dates from the stock data
    daily_data.set_index('end_date', inplace=True)
    daily_data = daily_data.reindex(prices.index).ffill()

    # Calculate the daily portfolio value
    portfolio_value = daily_data['remaining_cash']
    for ticker in tickers:
        portfolio_value += daily_data[f'shares_{ticker}'] * prices[ticker]

    # Plot the portfolio value over time
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', color='green')

    # Add vertical lines for rebalance dates
    # for date in df.index:
    #     plt.axvline(x=date, color='red', linestyle='--')

    # Create custom legends
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='Portfolio Value')]#,
                    #    Line2D([0], [0], color='red', linestyle='--', label='Rebalance Date')]
    plt.legend(handles=legend_elements)

    plt.title('Daily Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value in $')
    plt.grid(True)
    plt.show()

def dynamic_backtesting_x2(tickers, start_date_data, start_backtesting, end_date, rf, optimization_strategy, rebalance_frequency_months=6, input_cash_frequency_months=12, input_cash_amount=0, withdraw_frequency_months=12, withdraw_amount=0, optimization_model='SLSQP', initial_portfolio_value=1_000_000, commission=0.0025):
    start_date = pd.to_datetime(start_backtesting)
    end_date = pd.to_datetime(end_date)
    portfolio_value = initial_portfolio_value
    previous_num_shares = pd.Series(0, index=tickers)
    results = []

    # Initial process at start_backtesting
    current_date = start_date

    # --- Function to calculate new shares and update portfolio ---
    def calculate_new_shares(current_prices, target_value_per_ticker):
        adjusted_prices = current_prices * (1 + commission)
        num_shares = (target_value_per_ticker / adjusted_prices).apply(np.floor)
        invested_value = num_shares * current_prices
        remaining_cash = portfolio_value - invested_value.sum()
        return num_shares, remaining_cash

    # --- Function to execute rebalancing ---
    def execute_rebalance():
        nonlocal portfolio_value, previous_num_shares
        strategy = QAA(tickers=tickers, start_date=start_date_data, end_date=current_date.strftime('%Y-%m-%d'), rf=rf)
        strategy.set_optimization_strategy(optimization_strategy)
        strategy.set_optimization_model(optimization_model)
        strategy.load_data()
        strategy.optimize()
        optimal_weights = strategy.optimal_weights
        current_prices = strategy.data.iloc[-1]

        investment_value_per_ticker = portfolio_value * optimal_weights
        num_shares, remaining_cash = calculate_new_shares(current_prices, investment_value_per_ticker)

        diff_shares = num_shares - previous_num_shares
        previous_num_shares = num_shares.copy()
        portfolio_value = (num_shares * current_prices).sum() + remaining_cash
        return optimal_weights, num_shares, diff_shares, remaining_cash

    # --- Main Loop (Monthly Iteration) ---
    while current_date <= end_date:
        # Rebalancing
        if current_date.month % rebalance_frequency_months == 0:
            optimal_weights, num_shares, diff_shares, remaining_cash = execute_rebalance()
            action = 'Rebalance' 
        else:
            optimal_weights = None
            num_shares = previous_num_shares
            diff_shares = pd.Series(0, index=tickers)
            remaining_cash = None 
            action = None 

        # Capital Injection
        if current_date.month % input_cash_frequency_months == 0:
            portfolio_value += input_cash_amount
            current_prices = QAA(tickers=tickers, start_date=start_date_data, end_date=current_date.strftime('%Y-%m-%d'), rf=rf).load_data()[0].iloc[-1]
            investment_value_per_ticker = portfolio_value * previous_num_shares / previous_num_shares.sum()  
            num_shares, remaining_cash = calculate_new_shares(current_prices, investment_value_per_ticker)
            diff_shares = num_shares - previous_num_shares
            previous_num_shares = num_shares.copy()
            action = 'Input'

        # Capital Withdrawal (Assuming selling shares proportionally)
        if current_date.month % withdraw_frequency_months == 0 and portfolio_value >= withdraw_amount:
            withdraw_proportion = withdraw_amount / portfolio_value
            current_prices = QAA(tickers=tickers, start_date=start_date_data, end_date=current_date.strftime('%Y-%m-%d'), rf=rf).load_data()[0].iloc[-1]
            shares_to_sell = (previous_num_shares * withdraw_proportion).apply(np.floor)
            portfolio_value -= (shares_to_sell * current_prices).sum()
            num_shares = previous_num_shares - shares_to_sell
            diff_shares = num_shares - previous_num_shares 
            previous_num_shares = num_shares.copy()
            remaining_cash = 0  
            action = 'Withdraw'

        # End-of-Month Weight Calculation (Assuming based on current portfolio value)
        current_prices = QAA(tickers=tickers, start_date=start_date_data, end_date=current_date.strftime('%Y-%m-%d'), rf=rf).load_data()[0].iloc[-1]
        current_value_per_ticker = num_shares * current_prices
        if current_value_per_ticker.sum() > 0:
            end_of_month_weights = current_value_per_ticker / current_value_per_ticker.sum()
        else: 
            end_of_month_weights = pd.Series(0, index=tickers)

        result_row = {
            'data_origin_date': start_date_data,
            'end_date': current_date.strftime('%Y-%m-%d'),
            'action': action, 
            **{f'weight_{ticker}': optimal_weights[i] if optimal_weights is not None else None for i, ticker in enumerate(tickers)},
            **{f'shares_{ticker}': num_shares[ticker] for ticker in tickers},
            **{f'diff_{ticker}': diff_shares[ticker] for ticker in tickers},
            **{f'value_{ticker}': current_value_per_ticker[ticker] for ticker in tickers},
            **{f'eom_weight_{ticker}': end_of_month_weights[i] for i, ticker in enumerate(tickers)},
            'remaining_cash': remaining_cash,
            'total_portfolio_value': portfolio_value
        }
        results.append(result_row)

        current_date += relativedelta(months=1) 

    results = pd.DataFrame(results)

    # Daily Data Preparation

    # Prepare the dataframe with dynamic column names for each ticker
    df_columns = ['end_date'] + [f'shares_{ticker}' for ticker in tickers] + ['remaining_cash']
    df = results[df_columns].copy()
    df['end_date'] = pd.to_datetime(df['end_date'])
    df.set_index('end_date', inplace=True)

    # Generate a date range for the daily data
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a new dataframe with a record for each day, forward-filling the rebalancing information
    daily_data = pd.DataFrame(index=date_range)
    daily_data = daily_data.join(df, how='left').ffill().reset_index().rename(columns={'index': 'end_date'})

    # Fetch historical stock data using yfinance for the entire range of daily_data
    stock_data = yf.download(tickers, start=daily_data['end_date'].min(), end=daily_data['end_date'].max())
    prices = stock_data['Adj Close']

    # Update the index of daily_data to match the dates from the stock data
    daily_data.set_index('end_date', inplace=True)
    daily_data = daily_data.reindex(prices.index).ffill()

    # Calculate the daily portfolio value
    portfolio_value = daily_data['remaining_cash']
    for ticker in tickers:
        portfolio_value += daily_data[f'shares_{ticker}'] * prices[ticker]

    return results, daily_data, portfolio_value

# EXAMPLE (DOES NOT WORK YET)

# a,b,c = dynamic_backtesting_x2(
#     tickers=tickers, 
#     start_date_data='2020-01-02', 
#     start_backtesting='2023-01-23', 
#     end_date='2024-01-23', 
#     rebalance_frequency_months = rebalance_frequency_months , 
#     rf=rf, 
#     optimization_strategy=optimization_strategy, 
#     optimization_model=optimization_model,
#     initial_portfolio_value = initial_portfolio_value
# )
