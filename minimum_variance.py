# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Function to extract data from Yahoo Finance
def get_yahoo_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    covariances = returns.cov()
    return returns, covariances

# Function get minimum variance portfolio
def get_minimum_variance_portfolio_weights(returns, covariances):
    num_assets = len(returns.columns)
    
    # dim of matrix are correct?
    if covariances.shape != (num_assets, num_assets):
        raise ValueError("Mismatch in dimensions of covariance matrix and number of assets")

    inv_covariances = np.linalg.inv(covariances)
    ones = np.ones(num_assets)

    weights_min_variance = np.dot(inv_covariances, ones) / np.dot(np.dot(ones, inv_covariances), ones)
    weights_min_variance = weights_min_variance / np.sum(weights_min_variance)

    return weights_min_variance

# Function of returns and variances
def calculate_return_and_variance(weights, returns, covariances):
    if len(weights) != returns.shape[1]:
        raise ValueError("Mismatch in number of assets and length of weights")

    portfolio_return = np.dot(weights, returns.mean())
    portfolio_variance = np.dot(np.dot(weights, covariances), weights)
    return portfolio_return, portfolio_variance


# User input for stock tickers
tickers = input("Enter stock tickers (comma-separated): ").split(',')

# Time period for data
start_date = '2020-01-01'
end_date = '2023-01-01'

# Get data from Yahoo Finance
returns, covariances = get_yahoo_data(tickers, start_date, end_date)

# Calculate the minimum variance portfolio
weights_min_variance = get_minimum_variance_portfolio_weights(returns, covariances)

# Print results
print("\nWeights of the minimum variance portfolio:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {weights_min_variance[i]:.4f}")

# Calculate the return and variance of the minimum variance portfolio
return_min_variance, variance_min_variance = calculate_return_and_variance(weights_min_variance, returns, covariances)

print("\nReturn of the minimum variance portfolio:", return_min_variance)
print("Variance of the minimum variance portfolio:", variance_min_variance)

# Generate the minimum variance frontier
weights_frontier = []
variances_frontier = []

for target_return in np.linspace(min(returns.mean()), max(returns.mean()), num=100):
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: np.dot(w, returns.mean()) - target_return}]

    result = minimize(lambda w: np.dot(np.dot(w, covariances), w), weights_min_variance, constraints=constraints)
    optimal_weight = result.x

    weights_frontier.append(optimal_weight)
    variances_frontier.append(np.dot(np.dot(optimal_weight, covariances), optimal_weight))

# Plot the minimum variance frontier
plt.figure(figsize=(10, 6))
plt.scatter(variances_frontier, [r for r in np.linspace(min(returns.mean()), max(returns.mean()), num=100)], c=variances_frontier, cmap='viridis', marker='o')
plt.title('Minimum Variance Frontier')
plt.xlabel('Variance')
plt.ylabel('Return')
plt.colorbar(label='Variance')
plt.show()

