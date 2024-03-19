
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Quantitative Asset Allocation (QAA)                                                        -- #
# -- script: main.py - Python script with the main functionality                                         -- #
# -- authors: diegotita4 - Antonio-IF - JoAlfonso - J3SVS - Oscar148                                     -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/diegotita4/PAP                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ----------------------------------------------------------------------------------------------------

# LIBRARIES
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

# ----------------------------------------------------------------------------------------------------

# CLASS DEFINITION
class QAA:
    """
    Class QAA: Quantitative Asset Allocation.

    This class provides functionalities for conducting quantitative analysis and asset allocation.

    Class Functions:
    - __init__(self, tickers=None, benchmark=None, rf=None, lower_bound=None, higher_bound=None, start_date=None, end_date=None, optimization_model=None, QAA_strategy=None, expected_returns=None, opinions=None, MAR=None): Constructor of the QAA class.
    - assets_metrics(self): Loads data, calculates returns, and computes statistical variables.
    - portfolio_metrics(self, returns): Calculates and displays the performance and volatility of the portfolio compared to the benchmark.
    - fixed_parameters(self, returns): Preprocesses and sets fixed parameters for optimization.
    - QAA_strategy_selection(self, returns): Executes the selected QAA strategy based on the configuration in QAA_instance.
    - optimization_model_selection(self, returns, objective_function, gradient_function): Executes the selected QAA strategy based on the configuration in QAA_instance.

    - SLSQP(self, returns, objective_function): Optimizes the objective function using the SLSQP method.
    - montecarlo(self, returns, objective_function): Optimizes the objective function using the Montecarlo method.
    - gradient_descent(self, returns, objective_function, gradient_function): Optimizes the objective function using Gradient Descent.

    - min_variance(self, returns): Calculates the portfolio with the minimum variance using the specified optimization model.
    - max_sharpe_ratio(self, returns): Calculates the portfolio with the maximum Sharpe Ratio using the specified optimization model.
    - omega(self, returns): Calculates the portfolio with the Omega using the specified optimization model.
    - semivariance(self, returns): Calculates the portfolio with the Semivariance using the specified optimization model.
    - sortino_ratio(self, returns): Calculates the portfolio with the maximum Sortino Ratio using the specified optimization model.
    - black_litterman(self, returns): Calculates the portfolio with the Black Litterman using the specified optimization model.
    - HRP(self, returns): Calculates the portfolio weights using the HRP (Hierarchical Risk Parity) approach.
    - roy_safety_first_ratio(self, returns): Calculates the portfolio with the maximum Roy Safety First Ratio using the specified optimization model.
    - martingale(self, returns): Martingale strategy for asset allocation.
    - 
    - 
    """

    # FORMAT VARIABLES
    TAU = 0.025
    TOLERANCE = 1e-8
    DAYS_IN_YEAR = 252
    LEARNING_RATE = 0.01
    DATE_FORMAT = "%Y-%m-%d"
    NUMBER_OF_SIMULATIONS = 10000

    # ----------------------------------------------------------------------------------------------------

    def __init__(self, tickers=None, benchmark=None, rf=None, lower_bound=None, higher_bound=None, 
                 start_date=None, end_date=None, optimization_model=None, QAA_strategy=None,
                 expected_returns=None, opinions=None, MAR=None):
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
        - optimization_model (str): Optimization model to use ("SLSQP", "MONTECARLO", OR "GRADIENT DESCENT").
        - QAA_strategy (str): QAA strategy to apply ("MIN VARIANCE", "MAX SHARPE RATIO", "OMEGA", "SEMIVARIANCE",
                                                     "SORTINO RATIO", "BLACK LITTERMAN", "HRP", "ROY SAFETY FISRT RATIO",
                                                     "MARTINGALE", "", or "").
        - expected_returns (numpy array): Investor's expected returns, one for each asset. Default: None.
        - opinions (numpy array): How much you expect each asset to outperform the others. Default: None.
        - MAR (float): the Minimum Acceptable Return (MAR) in Roy's Safety-First Ratio strategy. Default: None.
        """

        # Validate tickers
        if not tickers or not isinstance(tickers, list):
            raise ValueError("A non-empty list of tickers is required.")

        # Validate dates
        if start_date and end_date:
            try:
                self.start_date = datetime.strptime(start_date, self.DATE_FORMAT)
                self.end_date = datetime.strptime(end_date, self.DATE_FORMAT)

                if self.start_date >= self.end_date:
                    raise ValueError("The start date must be before the end date.")

            except ValueError:
                raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")
        else:
            self.start_date = self.end_date = None

        # Validate risk-free rate
        if not isinstance(rf, (int, float)):
            raise ValueError("Risk-free rate (rf) must be a numeric value.")

        # Validate bounds
        if not (0 <= lower_bound <= 1) or not (0 <= higher_bound <= 1):
            raise ValueError("Bounds must be between 0 and 1.")

        # Validate optimization model
        if optimization_model not in ["SLSQP", "MONTECARLO", "GRADIENT DESCENT"]:
            raise ValueError("Invalid optimization model.")

        # Validate QAA strategy
        if QAA_strategy not in ["MIN VARIANCE", "MAX SHARPE RATIO", "OMEGA", "SEMIVARIANCE", "SORTINO RATIO", "BLACK LITTERMAN", "HRP", "ROY SAFETY FIRST RATIO", "MARTINGALE"]:#, "", ""]:
            raise ValueError("Invalid QAA strategy.")

        # Set default values
        self.expected_returns = expected_returns if expected_returns is not None else np.array([0.1] * len(tickers))
        self.opinions = opinions if opinions is not None else np.zeros((len(expected_returns), len(tickers)))
        self.MAR = MAR if MAR is not None else 0.2

        # Assign parameters
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
            data = yf.download(assets, start=self.start_date, end=self.end_date)["Adj Close"]

            if data.empty:
                raise ValueError("No data available for the specified date range.")

            # Calculate returns and statistical variables
            returns = data.pct_change().dropna()

            if returns.empty:
                raise ValueError("Insufficient data to calculate returns.")

            volatility = returns.std()
            variance = returns.var()
            covariance_matrix = returns.cov()
            correlation_matrix = returns.corr()
            return data, returns, volatility, variance, covariance_matrix, correlation_matrix

        except yf.exceptions.YFinanceError as e:
            raise ValueError(f"Error in data retrieval: {str(e)}")

        except ValueError as ve:
            raise ValueError(f"Error in data processing: {str(ve)}")

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

        try:
            # Extract relevant returns
            portfolio_returns = returns.drop(columns=[self.benchmark])
            benchmark_returns = returns.drop(columns=self.tickers)

            # Calculate portfolio metrics
            portfolio_return = np.dot(portfolio_returns.mean(), self.optimal_weights)
            benchmark_return = benchmark_returns.mean()
            rf_return = self.rf / self.DAYS_IN_YEAR

            portfolio_volatility = np.sqrt(np.dot(self.optimal_weights.T, np.dot(portfolio_returns.cov(), self.optimal_weights)))
            benchmark_volatility = benchmark_returns.std()
            rf_volatility = 0.0

            # Display metrics
            print("\n---\n")
            print(f"Portfolio Return: {portfolio_return * 100:.2f}%")
            print(f"Benchmark Return ({self.benchmark}): {benchmark_return.iloc[0] * 100:.2f}%")
            print(f"Risk-Free Rate Return: {rf_return * 100:.2f}%")
            print("\n---\n")
            print(f"Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
            print(f"Benchmark Volatility ({self.benchmark}): {benchmark_volatility.iloc[0] * 100:.2f}%")
            print(f"Risk-Free Rate Volatility: {rf_volatility * 100:.2f}%")
            print("\n---\n")

            # Plot graphs
            fig, axs = plt.subplots(1, 2, figsize=(20, 7))

            axs[0].plot(portfolio_returns.index, np.cumprod(1 + portfolio_returns.dot(self.optimal_weights)) - 1, label="Portfolio", color="blue")
            axs[0].plot(benchmark_returns.index, np.cumprod(1 + benchmark_returns) - 1, label=f"Benchmark ({self.benchmark})", color="red")
            axs[0].axhline(y=rf_return, color="green", linestyle="-", label="Risk-Free Rate")

            axs[0].set_title("Portfolio Returns VS Benchmark Returns VS rf", fontsize=18)
            axs[0].set_xlabel("Date", fontsize=14)
            axs[0].set_ylabel("Returns", fontsize=14)
            axs[0].tick_params(axis="both", labelsize=12)
            axs[0].legend(fontsize=12)

            axs[1].plot(portfolio_returns.index, portfolio_returns.dot(self.optimal_weights).rolling(window=20).std(), label="Portfolio", color="blue")
            axs[1].plot(benchmark_returns.index, benchmark_returns.rolling(window=20).std(), label=f"Benchmark ({self.benchmark})", color="red")
            axs[1].axhline(y=rf_volatility, color="green", linestyle="-", label="Risk-Free Rate")

            axs[1].set_title("Portfolio Volatility VS Benchmark Volatility VS rf", fontsize=18)
            axs[1].set_xlabel("Date", fontsize=14)
            axs[1].set_ylabel("Volatility", fontsize=14)
            axs[1].tick_params(axis="both", labelsize=12)
            axs[1].legend(fontsize=12)

            plt.show()

        except ValueError as ve:
            raise ValueError(f"Error in portfolio metrics calculation: {str(ve)}")

# ----------------------------------------------------------------------------------------------------

    # FIXED PARAMETERS
    def fixed_parameters(self, returns):
        """
        Preprocesses and sets fixed parameters for optimization.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights (np.array): Initial weights for the assets.
        - bounds (np.array): Bounds for asset weights.
        - constraints (list): List of optimization constraints.
        """

        try:
            num_assets = len(returns.columns)

            if num_assets <= 0:
                raise ValueError("Number of assets must be greater than zero.")

            # Set initial weights, bounds, and constraints
            weights = np.ones(num_assets) / num_assets
            bounds = np.array([(self.lower_bound, self.higher_bound)] * num_assets)
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            return weights, bounds, constraints

        except ValueError as ve:
            raise ValueError(f"Error in fixed parameters calculation: {str(ve)}")

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

        try:
            # Select and execute the chosen QAA strategy
            if self.QAA_strategy == "MIN VARIANCE":
                return self.min_variance(returns)

            elif self.QAA_strategy == "MAX SHARPE RATIO":
                return self.max_sharpe_ratio(returns)

            elif self.QAA_strategy == "OMEGA":
                return self.omega(returns)

            elif self.QAA_strategy == "SEMIVARIANCE":
                return self.semivariance(returns)

            elif self.QAA_strategy == "SORTINO RATIO":
                return self.sortino_ratio(returns)

            elif self.QAA_strategy == "BLACK LITTERMAN":
                return self.black_litterman(returns)

            elif self.QAA_strategy == "HRP":
                return self.HRP(returns)

            elif self.QAA_strategy == "ROY SAFETY FIRST RATIO":
                return self.roy_safety_first_ratio(returns)
            
            elif self.QAA_strategy == "MARTINGALE":
                return self.martingale(returns)

            #elif self.QAA_strategy == "":
                #return self.(returns)

            #elif self.QAA_strategy == "":
                #return self.(returns)

            else:
                raise ValueError(f"QAA Strategy '{self.QAA_strategy}' not recognized. Please choose a valid strategy.")

        except Exception as e:
            raise ValueError(f"Error in QAA strategy selection: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # OPTIMIZATION MODEL SELECTION
    def optimization_model_selection(self, returns, objective_function, gradient_function):
        """
        Executes the selected optimization model based on the configuration in QAA_instance.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.
        - objective_function (function): Objective function to optimize.
        - gradient_function (function): Gradient function of the objective function.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        - optimization_model (str): Selected optimization model.
        """

        try:
            # Select and execute the chosen optimization model
            if self.optimization_model == "SLSQP":
                result = self.SLSQP(returns, objective_function)
                optimization_model = "SLSQP"

            elif self.optimization_model == "MONTECARLO":
                result = self.montecarlo(returns, objective_function)
                optimization_model = "MONTECARLO"

            elif self.optimization_model == "GRADIENT DESCENT":
                result = self.gradient_descent(returns, objective_function, gradient_function)
                optimization_model = "GRADIENT DESCENT"

            else:
                raise ValueError(f"Invalid optimization model: {self.optimization_model}")

            return result, optimization_model

        except Exception as e:
            raise ValueError(f"Error in optimization model selection: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 1ST OPTIMIZE MODEL: "SLSQP"
    def SLSQP(self, returns, objective_function):
        """
        Optimizes the objective function using the SLSQP model.

        Parameters:
        - returns (pd.DataFrame): Processed returns after removing the benchmark.
        - objective_function (function): Objective function to optimize.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        """

        # Get initial weights, bounds, and constraints
        weights, bounds, constraints = self.fixed_parameters(returns)

        try:
            # Minimize the objective function using SLSQP model
            result = minimize(objective_function, weights, method="SLSQP", bounds=bounds, constraints=constraints)
            return result

        except Exception as e:
            raise ValueError(f"Error in SLSQP optimization: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 2ND OPTIMIZE MODEL: "MONTECARLO (BFGS)"
    def montecarlo(self, returns, objective_function):
        """
        Optimizes the objective function using the Montecarlo model.

        Parameters:
        - returns (pd.DataFrame): Processed returns after removing the benchmark.
        - objective_function (function): Objective function to optimize.

        Returns:
        - best_result (scipy.optimize.OptimizeResult): Optimization result.
        """

        # Get initial weights, bounds, and constraints
        weights, bounds, constraints = self.fixed_parameters(returns)

        all_results = []

        try:
            # Perform Montecarlo simulations
            for _ in range(self.NUMBER_OF_SIMULATIONS):
                random_weights = np.random.uniform(bounds[:, 0], bounds[:, 1])
                random_weights /= np.sum(random_weights)

                # Optimize with random weights and store results
                result = minimize(objective_function, random_weights, constraints=constraints, bounds=bounds)
                all_results.append(result)

            # Select the best result based on the objective function value
            best_result = min(all_results, key=lambda x: x.fun)
            return best_result

        except Exception as e:
            raise ValueError(f"Error in Montecarlo optimization: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 3RD OPTIMIZE MODEL: "GRADIENT DESCENT"
    def gradient_descent(self, returns, objective_function, gradient_function):
        """
        Optimizes the objective function using Gradient Descent.

        Parameters:
        - returns (pd.DataFrame): Processed returns after removing the benchmark.
        - objective_function (function): Objective function to optimize.
        - gradient_function (function): Gradient function of the objective function.

        Returns:
        - result (scipy.optimize.OptimizeResult): Optimization result.
        """

        # Get initial weights, bounds, and constraints
        weights, bounds, constraints = self.fixed_parameters(returns)

        iteration = 0

        # Perform gradient descent iterations
        while iteration < self.NUMBER_OF_SIMULATIONS:
            try:
                # Calculate gradient and update weights
                gradient = gradient_function(weights)

                #Check it to be a numpy array, it doesn´t work with anything else
                gradient = gradient.values if isinstance(gradient,pd.Series) else gradient
                new_weights = weights - self.LEARNING_RATE * gradient

                # Clip weights to the specified bounds
                new_weights = np.clip(new_weights, bounds[:, 0], bounds[:, 1])

                # Apply constraints
                if constraints:
                    for constraint in constraints:
                        if "fun" in constraint:
                            constraint_value = constraint["fun"](new_weights)
                            if not np.isclose(constraint_value, 0.0, atol=self.TOLERANCE):
                                constraint_jac = constraint.get("jac", 0)
                                if np.linalg.norm(constraint_jac) > 0:
                                    new_weights -= constraint_value * constraint_jac / np.dot(constraint_jac, constraint_jac)

                # Normalize weights to ensure they sum to 1
                new_weights /= np.sum(new_weights)

                # Check convergence
                if np.linalg.norm(new_weights - weights) < self.TOLERANCE:
                    break

                weights = new_weights
                iteration += 1

            except Exception as e:
                raise ValueError(f"Error in gradient descent optimization: {str(e)}")

        # Prepare and return optimization result
        result = {"x": weights, "fun": objective_function(weights), "nit": iteration, "success": iteration < self.NUMBER_OF_SIMULATIONS}
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

        try:
            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Define the objective function and gradient function
            objective_function = lambda w: np.dot(w.T, np.dot(returns.cov(), w))
            gradient_function = lambda w: 2 * np.dot(returns.cov(), w)

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

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

        try:
            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Define the objective function and gradient function for Sharpe Ratio
            objective_function = lambda w: -((np.dot(returns.mean(), w) - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))))
            gradient_function = lambda w: -((returns.mean() * self.DAYS_IN_YEAR - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov() * self.DAYS_IN_YEAR, w)))) * ((returns.cov() * self.DAYS_IN_YEAR) @ w) / (np.power(np.dot(w.T, np.dot(returns.cov() * self.DAYS_IN_YEAR, w)), 1.5) * np.sqrt(np.dot(w.T, np.dot(returns.cov() * self.DAYS_IN_YEAR, w))))

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 3RD QAA STRATEGY: "OMEGA"
    def omega(self, returns):
        """
        Calculates the portfolio with the maximum OMEGA using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Define the threshold return
            threshold_return = self.rf / self.DAYS_IN_YEAR

            # Define the objective function for OMEGA
            def objective_function(weights):
                portfolio_returns = np.dot(returns, weights)
                excess_returns = portfolio_returns - threshold_return

                upside_potential = np.sum(excess_returns[excess_returns > 0]) / len(excess_returns)
                downside_risk = -np.sum(excess_returns[excess_returns < 0]) / len(excess_returns) if np.sum(excess_returns < 0) != 0 else 1

                omega_ratio = upside_potential / downside_risk
                return -omega_ratio

            # Define the gradient function for OMEGA
            def gradient_function(weights):
                eps = 1e-8
                grad = np.zeros(len(weights))

                for i in range(len(weights)):
                    weights_p = np.array(weights)
                    weights_p[i] += eps
                    grad[i] = (objective_function(weights_p) - objective_function(weights)) / eps
                return grad

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 4TH QAA STRATEGY: "SEMIVARIANCE"
    def semivariance(self, returns):
        """
        Calculates the portfolio with the Semivariance using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Identify negative returns for Semivariance
            diff_neg = returns.copy()
            diff_neg[diff_neg > 0] = 0

            # Define the gradient function for Semivariance
            def gradient_function(weights):
                gradient = -1/2 * np.dot(diff_neg.cov(), weights) / np.sqrt(np.dot(weights.T, np.dot(diff_neg.cov(), weights)))
                return gradient

            # Define the objective function for Semivariance
            def objective_function(weights):
                semivariance = np.sqrt(np.dot(weights.T, np.dot(diff_neg.cov(), weights)))
                return semivariance

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 5TH QAA STRATEGY: "SORTINO RATIO"
    def sortino_ratio(self, returns):
        """
        Calculates the portfolio with the maximum Sortino Ratio using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Calculate downside returns
            downside_returns = np.minimum(returns, 0)

            # Define the objective function for Sortino Ratio
            objective_function = lambda w: -((np.dot(returns.mean(), w) - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))))

            # Define the downside deviation function for Sortino Ratio
            downside_deviation = lambda w: np.sqrt(np.dot(w.T, np.dot(downside_returns.cov(), w)))

            # Define the gradient function for Sortino Ratio
            gradient_function = lambda w: -((returns.mean() - self.rf) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))) * (returns.cov() @ w) / np.sqrt(np.dot(w.T, np.dot(returns.cov(), w))) - (downside_returns.cov() @ w) / downside_deviation(w)

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 6TH QAA STRATEGY: "BLACK LITTERMAN"
    def black_litterman(self, returns):
        """
        Calculates the portfolio with the Black Litterman using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Subjective estimates
            E_r = self.expected_returns
            opinions_p = self.opinions
            Omega = np.diag(np.power(E_r, 2))

            # Drop the benchmark column from returns
            returns = returns.drop(columns=[self.benchmark])

            # Input data
            cov = returns.cov()

            # Calculation of Black Litterman model parameters
            post_mu = (returns.mean() + self.TAU * cov.dot(opinions_p.T).dot(np.linalg.inv(opinions_p.dot(self.TAU ** 2 * cov).dot(opinions_p.T) + Omega)).dot(E_r - opinions_p.dot(returns.mean())))
            post_cov = (cov + self.TAU * cov - self.TAU ** 2 * cov).dot(opinions_p.T).dot(np.linalg.inv(opinions_p.dot(self.TAU ** 2 * cov).dot(opinions_p.T) + Omega)).dot(opinions_p.dot(self.TAU ** 2 * cov))
            volatility = np.sqrt(np.diagonal(post_cov))

            # Define the objective function for Black Litterman
            objective_function = lambda weight: -post_mu.dot(weight) + 0.5 * self.TAU * weight.dot(post_cov).dot(weight)

            # Define the gradient function for Black Litterman
            gradient_function = lambda weight: -(post_mu + self.TAU * post_cov.dot(weight))

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 7TH QAA STRATEGY: "HRP (HIERARCHICAL RISK PARITY)"
    def HRP(self, returns):
        """
        Calculates the portfolio weights using the HRP (Hierarchical Risk Parity) approach.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop benchmark column if present
            returns = returns.drop(columns=[self.benchmark])

            euclidean_distance = np.sqrt(0.5 * (1 - returns.corr()))
            linkage_matrix = linkage(euclidean_distance, method="single")
            leaves_order = leaves_list(linkage_matrix)
            names = returns.iloc[:, leaves_order].columns

            def bisection(items):
                new_items = [i[int(j):int(k)] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                return new_items

            def cluster_variation(returns, items):
                cov = returns.cov().copy().loc[items, items]
                inv_cov = np.linalg.inv(cov)
                weights = 1 / np.diag(inv_cov)
                weights /= np.sum(weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov, weights))
                return portfolio_variance

            # Initialize weights
            weights = pd.Series(1, index=names)

            # Initialize clusters with indices given by cluster
            clusters = [names]

            # While there are still clusters...
            while len(clusters) > 0:
                # Apply recursive bisection
                clusters = bisection(clusters)

                # For each cluster from the recursive bisection splitting into two
                for i in range(0, len(clusters), 2):
                    # Get indices i and i+1 of clusters
                    item_0 = clusters[i]
                    item_1 = clusters[i + 1]

                    # Calculate variance of i and i+1 cluster, using cluster_variation
                    v_0 = cluster_variation(returns, item_0)
                    v_1 = cluster_variation(returns, item_1)

                    # Calculate alpha or adjustment factor for weights
                    alpha = 1 - v_0 / (v_0 + v_1)

                    # Adjust weights with alpha
                    weights[item_0] *= alpha
                    weights[item_1] *= 1 - alpha

            # Define the objective function for HRP
            def objective_function(weights):
                portfolio_variance = 0

                # Calculate the variance of each cluster and sum them up
                for items in clusters:
                    cov = returns.cov().copy().loc[items, items]
                    inv_cov = np.linalg.inv(cov)
                    weights_cluster = weights[items]
                    portfolio_variance += np.dot(weights_cluster.T, np.dot(cov, weights_cluster))

                return portfolio_variance

            # Define the gradient function for HRP
            def gradient_function(weights):
                grad = np.zeros(len(weights))

                for items in clusters:
                    cov = returns.cov().copy().loc[items, items]
                    inv_cov = np.linalg.inv(cov)
                    weights_cluster = weights[items]
                    grad[items] = -2 * np.dot(cov, weights_cluster)

                return grad

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 8TH QAA STRATEGY: "ROY SAFETY FIRST RATIO"
    def roy_safety_first_ratio(self, returns):
        """
        Calculates the portfolio with the maximum Roy Safety First Ratio using the specified optimization model.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop benchmark column if present
            returns = returns.drop(columns=[self.benchmark])

            def objective_function(w):
                E_rp = np.dot(w, returns.mean())
                sigma_p = np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))
                return -(E_rp - self.MAR) / sigma_p

            def gradient_function(w):
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                portfolio_return = np.dot(w, mean_returns)
                portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

                grad_expected_return = mean_returns
                grad_volatility = np.dot(cov_matrix, w) / portfolio_volatility

                gradient = -(grad_expected_return * portfolio_volatility - (portfolio_return - self.MAR) * grad_volatility) / (portfolio_volatility ** 2)
                return gradient

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 9TH QAA STRATEGY: "MARTINGALE"
    def martingale(self, returns):
        """
        Martingale strategy for asset allocation.

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop benchmark column if present
            returns = returns.drop(columns=[self.benchmark])

            def martingale_objective_function(weights, returns):
                portfolio_return = np.sum(returns.mean() * weights)
                return -portfolio_return
            
            def martingale_gradient_function(weights, returns):
                mean_returns = returns.mean()
                gradient = -mean_returns
                return gradient

            # Define the objective function for Martingale
            objective_function = lambda w: martingale_objective_function(w, returns)

            # Define the gradient function for Martingale
            gradient_function = lambda w: martingale_gradient_function(w, returns)

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 10TH QAA STRATEGY: ""
    def ten(self, returns):
        """
        

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop benchmark column if present
            returns = returns.drop(columns=[self.benchmark])



            # Define the objective function and gradient function
            objective_function = a
            gradient_function = a

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")

# ----------------------------------------------------------------------------------------------------

    # 11TH QAA STRATEGY: ""
    def eleven(self, returns):
        """
        

        Parameters:
        - returns (pd.DataFrame): Historical returns of the assets.

        Returns:
        - weights_series (pd.Series): Optimal weights of the portfolio.
        """

        try:
            # Drop benchmark column if present
            returns = returns.drop(columns=[self.benchmark])



            # Define the objective function and gradient function
            objective_function = a
            gradient_function = a

            # Get the optimization result using the selected method
            result, optimization_model = self.optimization_model_selection(returns, objective_function, gradient_function)

            # Extract optimal weights from the result
            self.optimal_weights = result["x"]

            # Create a pandas Series for optimal weights
            weights_series = pd.Series(self.optimal_weights, index=self.tickers, name="Optimal Weights")

            # Display optimal weights
            print(f"\nOptimal Portfolio Weights for {self.QAA_strategy} QAA using {optimization_model} optimization:")
            print(weights_series)
            return weights_series

        except Exception as e:
            raise ValueError(f"Error in HRP strategy: {str(e)}")
