import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def calculate_portfolio_return(weights, returns):
    portfolio_return = np.dot(returns.mean(), weights)
    return portfolio_return

def calculate_portfolio_volatility(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_volatility

def objective_function(weights, returns, cov_matrix, strategy='maximize_return'):
    if strategy == 'maximize_return':
        return -calculate_portfolio_return(weights, returns)
    elif strategy == 'minimize_volatility':
        return calculate_portfolio_volatility(weights, cov_matrix)

# Ejemplo de uso
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
benchmark_ticker = 'SPY'
start_date = '2020-01-01'
end_date = '2022-01-01'

data = get_data(tickers + [benchmark_ticker], start_date, end_date)
returns = calculate_returns(data)
cov_matrix = returns.cov()

# Estrategia 1: Maximize Return
initial_weights = np.ones(len(tickers)) / len(tickers)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
result_strategy1 = minimize(objective_function, initial_weights, args=(returns, cov_matrix, 'maximize_return'),
                             method='SLSQP', constraints=constraints)

weights_strategy1 = result_strategy1.x

# Estrategia 2: Minimize Volatility
result_strategy2 = minimize(objective_function, initial_weights, args=(returns, cov_matrix, 'minimize_volatility'),
                             method='SLSQP', constraints=constraints)

weights_strategy2 = result_strategy2.x

# Comparación con el Benchmark
benchmark_returns = data[benchmark_ticker].pct_change().dropna()

portfolio_return_strategy1 = -result_strategy1.fun
portfolio_volatility_strategy1 = calculate_portfolio_volatility(weights_strategy1, cov_matrix)

portfolio_return_strategy2 = -result_strategy2.fun
portfolio_volatility_strategy2 = calculate_portfolio_volatility(weights_strategy2, cov_matrix)

benchmark_return = benchmark_returns.mean()
benchmark_volatility = benchmark_returns.std()

print("Resultados de la Estrategia 1:")
print("Rendimiento de la cartera:", portfolio_return_strategy1)
print("Volatilidad de la cartera:", portfolio_volatility_strategy1)

print("\nResultados de la Estrategia 2:")
print("Rendimiento de la cartera:", portfolio_return_strategy2)
print("Volatilidad de la cartera:", portfolio_volatility_strategy2)

print("\nBenchmark:")
print("Rendimiento del benchmark:", benchmark_return)
print("Volatilidad del benchmark:", benchmark_volatility)
