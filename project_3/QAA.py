
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Quantitative Analysis and Allocation (QAA)                                                 -- #
# -- script: main.py - Python script with the main functionality                                         -- #
# -- authors: diegotita4 - Antonio-IF - JoAlfonso - J3SVS - Oscar148                                     -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/diegotita4/PAP                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ----------------------------------------------------------------------------------------------------

# LIBRARIES
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------------------------------------------------

# CLASS DEFINITION
class QAA:
    """
    Class QAA: Quantitative Analysis and Allocation.

    This class provides functionalities for conducting quantitative analysis and asset allocation.

    Class Functions:
    - __init__(self, tickers=None, benchmark=None, rf=None, lower_bound=None, higher_bound=None, start_date=None, end_date=None, optimization_model=None, QAA_strategy=None): Constructor of the QAA class.
    - assets_metrics(self): Loads data, calculates returns, and computes statistical variables.
    - portfolio_metrics(self, returns): Calculates and displays the performance and volatility of the portfolio compared to the benchmark.
    - fixed_parameters(self, returns): Preprocesses and sets fixed parameters for optimization.
    - QAA_strategy_selection(self, returns): Executes the selected QAA strategy based on the configuration in QAA_instance.
    - SLSQP(self, objective_function, returns): Optimizes the objective function using the SLSQP method.
    - montecarlo(self, objective_function, returns, num_simulations): Optimizes the objective function using the Montecarlo method.
    - gradient_descent(self, objective_function, returns, num_simulations, gradient_function, learning_rate, tolerance=1e-8): Optimizes the objective function using Gradient Descent.
    - min_variance(self, returns): Calculates the portfolio with the minimum variance using the specified optimization model.
    - max_sharpe_ratio(self, returns): Calculates the portfolio with the maximum Sharpe Ratio using the specified optimization model.
    """

    # FORMAT VARIABLES
    DAYS_IN_YEAR = 252
    NUMBER_OF_SIMULATIONS = 10000
    LEARNING_RATE = 0.001
    DATE_FORMAT = '%Y-%m-%d'

# ----------------------------------------------------------------------------------------------------

    # INPUT VARIABLES
    def __init__(self, tickers=None, benchmark=None, rf=None, lower_bound=None,
                 higher_bound=None, start_date=None, end_date=None, optimization_model=None, QAA_strategy=None):
        """
        Constructor of the QAA class.

        Parameters:
        - tickers (list): List of asset tickers in the portfolio (cannot be empty).
        - benchmark (str): Ticker of the benchmark for comparisons.
        - rf (float): Risk-free rate.
        - lower_bound (float): Lower limit for asset weights.
        - higher_bound (float): Upper limit for asset weights.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        - optimization_model (str): Optimization model to use ('SLSQP', 'MONTECARLO', OR 'GRADIENT DESCENT').
        - QAA_strategy (str): QAA strategy to apply ('MIN VARIANCE', 'MAX SHARPE RATIO', ...).
        """

        if not tickers or not isinstance(tickers, list):
            raise ValueError("A non-empty list of tickers is required.")

        if start_date and end_date:
            try:
                self.start_date = datetime.datetime.strptime(start_date, self.DATE_FORMAT)
                self.end_date = datetime.datetime.strptime(end_date, self.DATE_FORMAT)

                if self.start_date >= self.end_date:
                    raise ValueError("The start date must be before the end date.")

            except ValueError:
                raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")

        else:
            self.start_date = self.end_date = None

        if not isinstance(rf, (int, float)):
            raise ValueError("Risk-free rate (rf) must be a numeric value.")

        if not (0 <= lower_bound <= 1) or not (0 <= higher_bound <= 1):
            raise ValueError("Bounds must be between 0 and 1.")

        if optimization_model not in ["SLSQP", "MONTECARLO", "GRADIENT DESCENT"]:
            raise ValueError("Invalid optimization model.")

        if QAA_strategy not in ["MIN VARIANCE", "MAX SHARPE RATIO", "OMEGA RATIO", 'MAX SORTINO RATIO']:
            raise ValueError("Invalid QAA strategy.")

        self.tickers = tickers
        self.benchmark = benchmark
        self.rf = rf
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.optimization_model = optimization_model
        self.QAA_strategy = QAA_strategy

# ----------------------------------------------------------------------------------------------------

    # INITIAL CALCULATIONS
    def assets_metrics(self):
        """
        Loads data, calculates returns, and computes statistical variables.

        Returns:
        - data (pd.DataFrame): DataFrame with loaded data.
        - returns (pd.DataFrame): DataFrame with calculated returns.
        - volatility (pd.Series): Standard deviation.
        - variance (pd.Series): Variance.
        - covariance_matrix (pd.DataFrame): Covariance matrix.
        - correlation_matrix (pd.DataFrame): Correlation matrix.
        """

        try:
            assets = self.tickers + [self.benchmark]

            data = yf.download(assets, start=self.start_date, end=self.end_date)['Adj Close']

            returns = data.pct_change().dropna()

            volatility = returns.std()
            variance = returns.var()
            covariance_matrix = returns.cov()
            correlation_matrix = returns.corr()
            return data, returns, volatility, variance, covariance_matrix, correlation_matrix

        except yf.exceptions.YFinanceError as e:
            raise ValueError(f"Error in data processing: {str(e)}")

        except ValueError as ve:
            raise ValueError(f"Error in date processing: {str(ve)}")

# ----------------------------------------------------------------------------------------------------

    # FINAL CALCULATIONS
    def portfolio_metrics(self, returns):
        """
        Calculates and displays the performance and volatility of the portfolio compared to the benchmark.

        Parameters:
        - returns (pd.DataFrame): Returns of the assets.

        Returns:
        - Graphs of returns and volatility of the assets and rf.
        """

        portfolio_returns = returns.drop(columns=[self.benchmark])
        benchmark_returns = returns.drop(columns=self.tickers)

        portfolio_return = np.dot(portfolio_returns.mean(), self.optimal_weights)
        benchmark_return = benchmark_returns.mean()
        rf_return = (1 + self.rf) ** (1/252) - 1

        portfolio_volatility = np.sqrt(np.dot(self.optimal_weights.T, np.dot(portfolio_returns.cov(), self.optimal_weights)))
        benchmark_volatility = benchmark_returns.std()
        rf_volatility = 0.0

        print("\n---\n")
        print(f"Portfolio Return: {portfolio_return * 100:.2f}%")
        print(f"Benchmark Return ({self.benchmark}): {benchmark_return.iloc[0] * 100:.2f}%")
        print(f"Risk-Free Rate Return: {rf_return * 100:.2f}%")
        print("\n---\n")
        print(f"Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
        print(f"Benchmark Volatility ({self.benchmark}): {benchmark_volatility.iloc[0] * 100:.2f}%")
        print(f"Risk-Free Rate Volatility: {rf_volatility * 100:.2f}%")
        print("\n---\n")

        fig, axs = plt.subplots(1, 2, figsize=(20, 7))

        axs[0].plot(portfolio_returns.index, np.cumprod(1 + portfolio_returns.dot(self.optimal_weights)) - 1, label='Portfolio', color='blue')
        axs[0].plot(benchmark_returns.index, np.cumprod(1 + benchmark_returns) - 1, label=f'Benchmark ({self.benchmark})', color='red')
        axs[0].axhline(y=rf_return, color='green', linestyle='-', label='Risk-Free Rate')

        axs[0].set_title('Portfolio Returns VS Benchmark Returns VS rf', fontsize=18)
        axs[0].set_xlabel('Date', fontsize=14)
        axs[0].set_ylabel('Returns', fontsize=14)
        axs[0].tick_params(axis='both', labelsize=12)
        axs[0].legend(fontsize=12)

        axs[1].plot(portfolio_returns.index, portfolio_returns.dot(self.optimal_weights).rolling(window=20).std(), label='Portfolio', color='blue')
        axs[1].plot(benchmark_returns.index, benchmark_returns.rolling(window=20).std(), label=f'Benchmark ({self.benchmark})', color='red')
        axs[1].axhline(y=rf_volatility, color='green', linestyle='-', label='Risk-Free Rate')

        axs[1].set_title('Portfolio Volatility VS Benchmark Volatility VS rf', fontsize=18)
        axs[1].set_xlabel('Date', fontsize=14)
        axs[1].set_ylabel('Volatility', fontsize=14)
        axs[1].tick_params(axis='both', labelsize=12)
        axs[1].legend(fontsize=12)

        plt.show()

# ----------------------------------------------------------------------------------------------------

    # FIXED PARAMETERS
    def fixed_parameters(self, returns):
        """
        Preprocesses and sets fixed parameters for optimization.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - returns (pd.DataFrame): Processed returns after removing the benchmark.
        - weights (np.array): Initial weights for the assets.
        - bounds (np.array): Bounds for asset weights.
        - constraints (list): List of optimization constraints.
        """

        num_assets = len(returns.columns)

        if num_assets <= 0:
            raise ValueError("Number of assets must be greater than zero.")

        weights = np.ones(num_assets) / num_assets
        bounds = np.array([(self.lower_bound, self.higher_bound)] * num_assets)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        return weights, bounds, constraints

# ----------------------------------------------------------------------------------------------------

    # STRATEGY SELECTION
    def QAA_strategy_selection(self, returns):
        """
        Executes the selected QAA strategy based on the configuration in QAA_instance.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """
        
        if self.QAA_strategy == 'MIN VARIANCE':
            return self.min_variance(returns)
        
        elif self.QAA_strategy == 'MAX SHARPE RATIO':
            return self.max_sharpe_ratio(returns)
        
        elif self.QAA_strategy == 'OMEGA RATIO':
            return self.max_omega_ratio(returns)
        
        elif self.QAA_strategy == 'MAX SORTINO RATIO':
            return self.max_sortino_ratio(returns)
        
        else:
            raise ValueError(f"QAA Strategy '{self.QAA_strategy}' not recognized.")

# ----------------------------------------------------------------------------------------------------

    # 1ST OPTIMIZE METHOD: "SLSQP"
    def SLSQP(self, objective_function, returns):
        """
        Optimizes the objective function using the SLSQP method.

        Parameters:
        - objective_function (function): Objective function to optimize.
        - weights (np.array): Initial weights.
        - bounds (tuple): Weights limits.
        - constraints (list): Optimization constraints.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        """

        weights, bounds, constraints = self.fixed_parameters(returns)

        try:
            result = minimize(objective_function, weights, method="SLSQP", bounds=bounds, constraints=constraints)
            return result

        except Exception as e:
            raise ValueError(f"Error in SLSQP optimization: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 2ND OPTIMIZE METHOD: "MONTECARLO (BFGS)"
    def montecarlo(self, objective_function, returns, num_simulations):
        """
        Optimizes the objective function using the Montecarlo method.

        Parameters:
        - objective_function (function): Objective function to optimize.
        - num_simulations (int): Number of Monte Carlo simulations.
        - bounds (tuple): Weights limits.
        - constraints (list): Optimization constraints.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        """

        weights, bounds, constraints = self.fixed_parameters(returns)

        all_results = []

        try:
            for _ in range(num_simulations):
                random_weights = np.random.uniform(bounds[:, 0], bounds[:, 1])
                random_weights /= np.sum(random_weights)

                result = minimize(objective_function, random_weights, constraints=constraints, bounds=bounds)
                all_results.append(result)

            best_result = min(all_results, key=lambda x: x.fun)

            return best_result

        except Exception as e:
            raise ValueError(f"Error in Montecarlo optimization: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 3RD OPTIMIZE METHOD: "GRADIENT DESCENT"
    def gradient_descent(self, objective_function, returns, num_simulations, gradient_function, learning_rate, tolerance=1e-8):
        """
        Optimizes the objective function using Gradient Descent.

        Parameters:
        - objective_function (function): Objective function to optimize.
        - gradient_function (function): Gradient function of the objective function.
        - weights (np.array): Initial weights.
        - learning_rate (float): Learning rate for the gradient descent.
        - bounds (tuple): Weights limits.
        - constraints (list): Optimization constraints.
        - tolerance (float): Tolerance to stop iterations.
        - num_simulations (int): Maximum number of iterations.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        """

        weights, bounds, constraints = self.fixed_parameters(returns)

        iteration = 0
        while iteration < num_simulations:
            try:
                gradient = gradient_function(weights)
                new_weights = weights - learning_rate * gradient

                new_weights = np.clip(new_weights, bounds[:, 0], bounds[:, 1])

                if constraints:
                    for constraint in constraints:
                        if 'fun' in constraint:
                            constraint_value = constraint['fun'](new_weights)
                            if not np.isclose(constraint_value, 0.0, atol=tolerance):
                                constraint_jac = constraint.get('jac', 0)
                                if np.linalg.norm(constraint_jac) > 0:
                                    new_weights -= constraint_value * constraint_jac / np.dot(constraint_jac, constraint_jac)

                new_weights /= np.sum(new_weights)

                if np.linalg.norm(new_weights - weights) < tolerance:
                    break

                weights = new_weights
                iteration += 1

            except Exception as e:
                raise ValueError(f"Error in gradient descent optimization: {str(e)}")

        result = {'x': weights, 'fun': objective_function(weights), 'nit': iteration, 'success': iteration < num_simulations}
        return result
# ----------------------------------------------------------------------------------------------------

    # 1ST QAA STRATEGY: "MIN VARIANCE"
    def min_variance(self, returns):
        """
        Calculates the portfolio with the minimum variance using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        returns = returns.drop(columns=[self.benchmark])

        objective_function = lambda w: np.dot(w.T, np.dot(returns.cov(), w))

        gradient_function = lambda w: 2 * np.dot(returns.cov(), w)

        if self.optimization_model == "SLSQP":
            result = self.SLSQP(objective_function, returns)
            optimization_model = "SLSQP"

        elif self.optimization_model == "MONTECARLO":
            result = self.montecarlo(objective_function, returns, self.NUMBER_OF_SIMULATIONS)
            optimization_model = "MONTECARLO"

        elif self.optimization_model == "GRADIENT DESCENT":
            result = self.gradient_descent(objective_function, returns, self.NUMBER_OF_SIMULATIONS, gradient_function, self.LEARNING_RATE)
            optimization_model = "GRADIENT DESCENT"

        else:
            raise ValueError(f"Invalid optimization method: {self.optimization_model}")

        self.optimal_weights = result['x']
        weights_series = pd.Series(self.optimal_weights, index=self.tickers, name='Optimal Weights')

        print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
        print(weights_series)

        return weights_series

# ----------------------------------------------------------------------------------------------------

    # 2ND QAA STRATEGY: "MAX SHARPE RATIO"
    def max_sharpe_ratio(self, returns):
        """
        Calculates the portfolio with the maximum Sharpe Ratio using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        returns = returns.drop(columns=[self.benchmark])

        objective_function = lambda w: -((np.dot(returns.mean(), w) - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))))

        gradient_function = lambda w: -((returns.mean() * self.DAYS_IN_YEAR - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov() * self.DAYS_IN_YEAR, w)))) * ((returns.cov() * self.DAYS_IN_YEAR) @ w) / np.power(np.dot(w.T, np.dot(returns.cov() * self.DAYS_IN_YEAR, w)), 1.5)

        if self.optimization_model == "SLSQP":
            result = self.SLSQP(objective_function, returns)
            optimization_model = "SLSQP"

        elif self.optimization_model == "MONTECARLO":
            result = self.montecarlo(objective_function, returns, self.NUMBER_OF_SIMULATIONS)
            optimization_model = "MONTECARLO"

        elif self.optimization_model == "GRADIENT DESCENT":
            result = self.gradient_descent(objective_function, returns, self.NUMBER_OF_SIMULATIONS, gradient_function, self.LEARNING_RATE)
            optimization_model = "GRADIENT DESCENT"

        else:
            raise ValueError(f"Invalid optimization method: {self.optimization_model}")

        self.optimal_weights = result['x']
        weights_series = pd.Series(self.optimal_weights, index=self.tickers, name='Optimal Weights')

        print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
        print(weights_series)

        return weights_series

# ----------------------------------------------------------------------------------------------------

        # 3RD QAA STRATEGY: "OMEGA RATIO"
    def max_omega_ratio(self, returns):
        """
        Calculates the portfolio with the maximum Omega Ratio using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """
        returns = returns.drop(columns=[self.benchmark])
        threshold_return = self.rf / self.DAYS_IN_YEAR  # Adjusted for daily returns if necessary

        # Define the objective function for Omega Ratio
        def objective_function(weights):
            portfolio_returns = np.dot(returns, weights)
            excess_returns = portfolio_returns - threshold_return

            upside_potential = np.sum(excess_returns[excess_returns > 0]) / len(excess_returns)
            downside_risk = -np.sum(excess_returns[excess_returns < 0]) / len(excess_returns) if np.sum(excess_returns < 0) != 0 else 1

            omega_ratio = upside_potential / downside_risk
            return -omega_ratio  # Negating the ratio for minimization

        if self.optimization_model == "SLSQP":
            result = self.SLSQP(objective_function, returns)
            optimization_model = "SLSQP"

        elif self.optimization_model == "MONTECARLO":
            result = self.montecarlo(objective_function, returns, self.NUMBER_OF_SIMULATIONS)
            optimization_model = "MONTECARLO"

        elif self.optimization_model == "GRADIENT DESCENT":
            result = self.gradient_descent(objective_function, returns, self.NUMBER_OF_SIMULATIONS, self.LEARNING_RATE)
            optimization_model = "GRADIENT DESCENT"

        else:
            raise ValueError(f"Invalid optimization method: {self.optimization_model}")

        self.optimal_weights = result['x']
        weights_series = pd.Series(self.optimal_weights, index=self.tickers, name='Optimal Weights')

        print(f"\nOptimal Portfolio Weights for MAX OMEGA RATIO QAA using {optimization_model} optimization:")
        print(weights_series)

        return weights_series
    

# ----------------------------------------------------------------------------------------------------

        # 4TH QAA STRATEGY: "MAX SORTINO RATIO"
    def max_sortino_ratio(self, returns):
        """
        Calculates the portfolio with the maximum Sortino Ratio using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        returns = returns.drop(columns=[self.benchmark])

        downside_returns = np.minimum(returns, 0)

        target_return = self.rf  
        
        objective_function = lambda w: -((np.dot(returns.mean(), w) - target_return) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))))

        downside_deviation = lambda w: np.sqrt(np.dot(w.T, np.dot(downside_returns.cov(), w)))

        gradient_function = lambda w: -((returns.mean() - target_return) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))) * (returns.cov() @ w) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))) - (downside_returns.cov() @ w) / downside_deviation(w)

        if self.optimization_model == "SLSQP":
            result = self.SLSQP(objective_function, returns)
            optimization_model = "SLSQP"

        elif self.optimization_model == "MONTECARLO":
            result = self.montecarlo(objective_function, returns, self.NUMBER_OF_SIMULATIONS)
            optimization_model = "MONTECARLO"
    
        elif self.optimization_model == "GRADIENT DESCENT":
            result = self.gradient_descent(objective_function, returns, self.NUMBER_OF_SIMULATIONS, gradient_function, self.LEARNING_RATE)
            optimization_model = "GRADIENT DESCENT"
    
        else:
            raise ValueError(f"Invalid optimization method: {self.optimization_model}")
    
        self.optimal_weights = result['x']
        weights_series = pd.Series(self.optimal_weights, index=self.tickers, name='Optimal Weights')
    
        print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
        print(weights_series)
    
        return weights_series
    

# ----------------------------------------------------------------------------------------------------
    



# ----------------------------------------------------------------------------------------------------
    




# EXAMPLE
qaa_instance = QAA(
    tickers=['ABBV', 'MET', 'OXY', 'PERI'],
    benchmark='SPY',
    rf=0.50, #modifique rf para hacer pruebas
    lower_bound=0.1,
    higher_bound=0.9,
    start_date='2020-01-02',
    end_date='2023-01-23',
    optimization_model='SLSQP',
    #optimization_model='MONTECARLO',
    #optimization_model='GRADIENT DESCENT',
    #optimization_model='SLSQP',
    #QAA_strategy='MIN VARIANCE',
    #QAA_strategy='MAX SHARPE RATIO',
    #QAA_strategy='OMEGA RATIO',
    #QAA_strategy='MAX SORTINO RATIO'
)

try:
    data, returns, std, var, cov, corr = qaa_instance.assets_metrics()

    optimal_weights = qaa_instance.QAA_strategy_selection(returns)
    
    qaa_instance.portfolio_metrics(returns)

except ValueError as ve:
    print(f"Error: {str(ve)}")
