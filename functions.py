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

# LIBRARIES / WARNINGS
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import optuna
import warnings
import logging
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientCVaR
import pandas_datareader.data as web
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
from pypfopt.hierarchical_portfolio import HRPOpt
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', message='A new study created in memory with name:')
warnings.filterwarnings('ignore', message='Method COBYLA cannot handle bounds.')
import empyrical # pip install empyrical

# ----------------------------------------------------------------------------------------------------


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

class HierarchicalRiskParity:
    
    def __init__(self, returns):
        """
        Initialize the HierarchicalRiskParity object with returns data.

        :param returns: Historical returns of assets.
        :type returns: pd.DataFrame
        """
        self.names = returns.columns
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
        
        self.link = None
        self.sort_ix = None
        self.weights = None

    def get_quasi_diagonalization(self, link):
        """
        Perform quasi-diagonalization ordering on a linkage matrix.

        :param link: Linkage matrix from hierarchical clustering.
        :type link: np.ndarray
        :return: Ordered indices based on quasi-diagonalization.
        :rtype: list
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()
   
    def cluster_variation(self, cov, c_items):
        """
        Calculate the variance of a cluster.

        :param cov: Covariance matrix of returns.
        :type cov: pd.DataFrame
        :param c_items: Indices of items in the cluster.
        :type c_items: list
        :return: Variance of the cluster.
        :rtype: float
        """
        c_cov = cov.iloc[c_items, c_items]
        
        ivp_weights = 1. / np.diag(c_cov)
        ivp_weights /= ivp_weights.sum()
        ivp_weights = ivp_weights.reshape(-1, 1)
        
        cluster_variance = np.dot(np.dot(ivp_weights.T, c_cov), ivp_weights)[0, 0]
        return cluster_variance

    @staticmethod
    def bisection(items):
        """
        Recursively split clusters into subclusters.

        :param items: List of cluster indices.
        :type items: list
        :return: List of subcluster indices after bisection.
        :rtype: list
        """
        new_items = [i[int(j):int(k)] for i in items for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        return new_items

    def HRP(self, cov, sort_ix):
        """
        Calculate asset weights using hierarchical risk parity.

        :param cov: Covariance matrix of returns.
        :type cov: pd.DataFrame
        :param sort_ix: Ordered indices from quasi-diagonalization.
        :type sort_ix: list
        :return: Asset weights based on HRP.
        :rtype: pd.Series
        """
        weights = pd.Series(1, index=sort_ix)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters = self.bisection(clusters)

            for i in range(0, len(clusters), 2):
                cluster_0 = clusters[i]
                cluster_1 = clusters[i + 1]

                c_var0 = self.cluster_variation(cov, cluster_0)
                c_var1 = self.cluster_variation(cov, cluster_1)

                alpha = 1 - c_var0 / (c_var0 + c_var1)

                weights[cluster_0] *= alpha
                weights[cluster_1] *= 1 - alpha

        return weights

    def optimize_hrp(self, linkage_method='single'):
        """
        Optimize asset weights using the specified linkage method.

        :param linkage_method: Method used for hierarchical clustering.
        :type linkage_method: str
        :return: Optimized asset weights based on HRP.
        :rtype: pd.Series
        """
        self.link = linkage(squareform(1 - self.corr), method=linkage_method)
        self.sort_ix = self.get_quasi_diagonalization(self.link)
        sorted_weights = self.HRP(self.cov, self.sort_ix)
        self.weights = pd.Series(index=self.names, dtype=float)

        for i, ix in enumerate(self.sort_ix):
            self.weights.iloc[ix] = sorted_weights.iloc[i]
        self.weights.name = "HRP"
        
        return self.weights


# ----------------------------------------------------------------------------------------------------

# CLASS DEFINITION
class QAA:
    """
    Class OptimizedStrategy: Quantitative Asset Allocation.

    This class provides functionalities for conducting quantitative analysis and asset allocation.

    Attributes:
    - tickers (list): List of asset tickers.
    - benchmark_ticker (str): Ticker of the benchmark asset.
    - rf (float): Risk-free rate.
    - lower_bound (float): Lower bound for asset weights.
    - higher_bound (float): Higher bound for asset weights.
    - start_date (str): Start date for data retrieval.
    - end_date (str): End date for data retrieval.
    - optimization_strategy (str): Selected optimization strategy.
    - optimization_model (str): Selected optimization model.

    Methods:
    - _init_: Constructor of the OptimizedStrategy class.
    - set_optimization_strategy: Sets the optimization strategy.
    - set_optimization_model: Sets the optimization model.
    - load_data: Loads historical data for the assets.
    - calculate_returns: Calculates daily returns for the assets.
    - validate_returns_empyrical: Validates the returns of each ticker using empyrical.
    - optimize_slsqp: Optimizes the objective function using the SLSQP method.
    - optimize_montecarlo: Optimizes the objective function using the Montecarlo method.
    - optimize_cobyla: Optimizes the objective function using the COBYLA method.
    - minimum_variance: Calculates the portfolio with the minimum variance.
    - omega_ratio: Calculates the portfolio with the Omega ratio.
    - semivariance: Calculates the portfolio with the Semivariance.
    - roy_safety_first_ratio: Calculates the portfolio with the Roy Safety First Ratio.
    - cvar: Calculates the portfolio with the CVaR (Conditional Value at Risk).
    - sortino_ratio: Calculates the portfolio with the Sortino ratio.
    - optimize: Executes the selected optimization strategy and model.
    """

    def __init__(self, tickers=None, benchmark_ticker='SPY', rf=None, lower_bound=0.10, higher_bound=0.99, start_date=None, end_date=None):
        """
        Initializes the QAA class.

        Parameters:
        - tickers (list, optional): List of asset tickers.
        - benchmark_ticker (str, optional): Ticker of the benchmark asset. Defaults to 'SPY'.
        - rf (float, optional): Risk-free rate. Defaults to None.
        - lower_bound (float, optional): Lower bound for asset weights. Defaults to 0.10.
        - higher_bound (float, optional): Higher bound for asset weights. Defaults to 0.99.
        - start_date (str, optional): Start date for data retrieval. Defaults to None.
        - end_date (str, optional): End date for data retrieval. Defaults to None.
        """
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.rf = rf
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.start_date = start_date
        self.end_date = end_date
        self.data, self.benchmark_data = self.load_data()
        self.returns = self.calculate_returns()
        self.benchmark_returns = self.calculate_benchmark_returns()
        self.optimal_weights = None
        self.optimization_strategy = None
        self.optimization_model = None
        self.ff_data, self.ff_returns = self.load_ff_data()

    def calculate_benchmark_returns(self):
        """Calculates daily returns for the benchmark asset."""
        if self.benchmark_data is not None:
            return self.benchmark_data.pct_change().dropna()
        else:
            raise ValueError("Benchmark data not found.")

    def set_optimization_strategy(self, strategy):
        """Sets the optimization strategy."""
        self.optimization_strategy = strategy

    def set_optimization_model(self, model):
        """Sets the optimization model."""
        self.optimization_model = model
        
    def calculate_portfolio_return(self):
        """Calculates the expected portfolio return based on current optimal weights."""
        if self.optimal_weights is not None and self.returns is not None:
            portfolio_return = np.dot(self.optimal_weights, self.returns.mean()) * 252  # annualizing the return
            return portfolio_return
        else:
            return 0  # Return 0 or handle appropriately if weights or returns are not defined


    def calculate_portfolio_volatility(self):
        """Calculates the portfolio volatility based on current optimal weights."""
        if self.optimal_weights is not None and self.returns is not None:
            return np.sqrt(np.dot(self.optimal_weights.T, np.dot(self.returns.cov() * 252, self.optimal_weights)))  # annualized volatility
        return None
    
    def calculate_portfolio_metrics(qaa_instance):
        # Check if the necessary methods are available in the QAA instance
        if hasattr(qaa_instance, 'calculate_portfolio_return') and hasattr(qaa_instance, 'calculate_portfolio_volatility'):
            portfolio_return = qaa_instance.calculate_portfolio_return()
            portfolio_volatility = qaa_instance.calculate_portfolio_volatility()
            return portfolio_return * 100, portfolio_volatility * 100
        else:
            raise AttributeError("QAA instance is missing required methods.")


    def load_data(self):
        """Loads historical data for the assets and benchmark."""
        if not self.tickers or self.benchmark_ticker is None:
            raise ValueError("You must provide a list of tickers and a benchmark ticker.")
        tickers_with_benchmark = self.tickers + [self.benchmark_ticker] if self.benchmark_ticker not in self.tickers else self.tickers
        data = yf.download(tickers_with_benchmark, start=self.start_date, end=self.end_date)['Adj Close']
        benchmark_data = data.pop(self.benchmark_ticker) if self.benchmark_ticker in data else None
        return data, benchmark_data
    
    def load_ff_data(self):
        ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=self.start_date, end=self.end_date)[0]
        ff_returns = ff_data[['Mkt-RF', 'SMB', 'HML']].loc[self.returns.index]
        return ff_data, ff_returns

    def calculate_returns(self):
        """Calculates daily returns for the assets."""
        return self.data.pct_change().dropna()

    def validate_returns_empyrical(self):
        """Validates the returns of each ticker using empyrical."""
        results = {}

        for ticker in self.tickers:
            # Assuming self.data already has the adjusted closing prices of each ticker
            prices = self.data[ticker]

            # Calculate daily returns
            daily_returns = prices.pct_change().dropna()

            # Calculate annualized return and Sharpe ratio
            annualized_return = empyrical.annual_return(daily_returns)
            sharpe_ratio = empyrical.sharpe_ratio(daily_returns, risk_free=self.rf / 252)

            # Store the results
            results[ticker] = {
                'Annualized Return': annualized_return,
                'Sharpe Ratio': sharpe_ratio
            }

        return results
# ----------------------------------------------------------------------------------------------------

    # OPTIMIZATION MODEL SELECTION
     # 1ST OPTIMIZE MODEL: "SLSQP"
    def optimize_slsqp(self):
        """Optimizes the objective function using the SLSQP method."""
        if self.optimization_strategy == 'Minimum Variance':
            objective = self.minimum_variance
        elif self.optimization_strategy == 'Omega Ratio':
            objective = lambda weights: -self.omega_ratio(weights, self.rf)
        elif self.optimization_strategy == 'Semivariance':
            objective = self.semivariance()
        elif self.optimization_strategy == 'Roy Safety First Ratio':
             objective = self.roy_safety_first_ratio
        elif self.optimization_strategy == 'Sortino Ratio':
             objective = self.sortino_ratio
        elif self.optimization_strategy == 'Fama French':
             objective = self.fama_french
        elif self.optimization_strategy == 'CVaR':
            objective = lambda weights: self.cvar(weights, alpha=0.05)
        elif self.optimization_strategy == 'HRP':
            hrp_instance = HierarchicalRiskParity(self.returns)
            hrp_weights = hrp_instance.optimize_hrp()
            self.optimization_strategy = hrp_weights
        elif self.optimization_strategy == 'Sharpe Ratio':
             objective = self.sharpe_ratio
        elif self.optimization_strategy == 'Black Litterman':
             objective = self.black_litterman
        elif self.optimization_strategy == 'Total Return':
             objective = self.Total_return
        else:
            raise ValueError("Invalid optimization strategy.")

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(self.lower_bound, self.higher_bound) for _ in self.tickers]

        result = minimize(objective, np.ones(len(self.tickers)) / len(self.tickers), method='SLSQP', bounds=bounds, constraints=constraints)
        self.optimal_weights = result.x if result.success else None
# ----------------------------------------------------------------------------------------------------

     # 2ND OPTIMIZE MODEL: "MONTECARLO (w/optuna)"

    def optimize_montecarlo(self):
        """Optimizes the objective function using the Montecarlo method."""
        def objective(trial):
            # Suppress specific warnings from Optuna about suggest_uniform
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)

                # Generate tentative weights with suggest_float
                weights = np.array([trial.suggest_float(f"weight_{i}", 0, 1) for i in range(len(self.tickers))])

            # Normalize the weights to sum to 1
            weights /= np.sum(weights)

            # Apply a penalty if the weights do not respect the lower bound
            penalty = 0
            for weight in weights:
                if weight < self.lower_bound:
                    penalty += 1e6 * (self.lower_bound - weight) ** 2

            # Calculate the objective value based on the selected strategy, including the penalty
            if self.optimization_strategy == 'Minimum Variance':
                objective_value = self.minimum_variance(weights) + penalty
            elif self.optimization_strategy == 'Omega Ratio':
                objective_value = -self.omega_ratio(weights, self.rf) + penalty
            elif self.optimization_strategy == 'Semivariance':
                objective_function = self.semivariance()  # Get the lambda function
                objective_value = objective_function(weights) + penalty  # Invoke it with the weights
            elif self.optimization_strategy == 'Roy Safety First Ratio':
                objective_value = self.roy_safety_first_ratio(weights) + penalty
            elif self.optimization_strategy == 'Sortino Ratio':  
                objective_value = self.sortino_ratio(weights) + penalty
            elif self.optimization_strategy == 'Fama French':
                objective_value = self.fama_french(weights) + penalty
            elif self.optimization_strategy == 'CVaR':
                portfolio_returns = np.dot(self.returns, weights)
                VaR = np.percentile(portfolio_returns, 100 * 0.05)
                CVaR = np.mean(portfolio_returns[portfolio_returns <= VaR])
                return -CVaR + penalty  # Minimize the negative CVaR (maximize CVaR)
                # objective_value = -self.cvar(weights, alpha=0.05) + penalty
            elif self.optimization_strategy == 'Sharpe Ratio':
                objective_value = self.sharpe_ratio(weights) + penalty
            elif self.optimization_strategy == 'Hierarchical Risk Parity':
                hrp_instance = HierarchicalRiskParity(self.returns)
                hrp_weights = hrp_instance.optimize_hrp()
                self.optimal_weights = hrp_weights
                return objective_value + penalty
            elif self.optimization_strategy == 'Black Litterman':
                objective_value = self.black_litterman(weights) + penalty
            elif self.optimization_strategy == 'Total Return':
                objective_value = self.Total_return(weights) + penalty
            else:
                raise ValueError("Invalid optimization strategy.")

            return objective_value

        # Create an Optuna study and find the optimal weights
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')

        num_trials = 500  # Adjust the number of trials as necessary
        study.optimize(objective, n_trials=num_trials, n_jobs=-1)  # n_jobs=-1 uses all available cores

        # Get and save the optimal weights
        optimal_weights = np.array([study.best_params[f"weight_{i}"] for i in range(len(self.tickers))])
        optimal_weights /= np.sum(optimal_weights)  # Normalize the weights to sum to 1
        self.optimal_weights = optimal_weights

    # ----------------------------------------------------------------------------------------------------

     # 3RD OPTIMIZE MODEL: "COBYLA"

    def optimize_cobyla(self):
        """Optimizes the objective function using the COBYLA method."""
        if self.optimization_strategy == 'Minimum Variance':
            objective = self.minimum_variance
        elif self.optimization_strategy == 'Omega Ratio':
            objective = lambda weights: -self.omega_ratio(weights, self.rf)
        elif self.optimization_strategy == 'Semivariance':
            objective = self.semivariance()
        elif self.optimization_strategy == 'Roy Safety First Ratio':
            objective = self.roy_safety_first_ratio
        elif self.optimization_strategy == 'Sortino Ratio':
             objective = self.sortino_ratio
        elif self.optimization_strategy == 'Fama French':
            objective = self.fama_french
        elif self.optimization_strategy == 'CVaR':
            objective = lambda weights: self.cvar(weights, alpha=0.05)
        elif self.optimization_strategy == 'Sharpe Ratio':
             objective = self.sharpe_ratio
        elif self.optimization_strategy == 'Hierarchical Risk Parity':
            hrp_instance = HierarchicalRiskParity(self.returns)
            hrp_weights = hrp_instance.optimize_hrp()
            self.optimal_weights = hrp_weights
            objetive = self.hrp_weights
        elif self.optimization_strategy == 'Black Litterman':
             objective = self.black_litterman
        elif self.optimization_strategy == 'Total Return':
             objective = self.Total_return
        else:
            raise ValueError("Invalid optimization strategy.")

        initial_weights = np.ones(len(self.tickers)) / len(self.tickers)

        # Define inequality constraints
        constraints = [{'type': 'ineq', 'fun': lambda weights, i=i: weights[i] - 0.10} for i in range(len(self.tickers))]
        constraints += [{'type': 'ineq', 'fun': lambda weights: 1 - np.sum(weights)},
                        {'type': 'ineq', 'fun': lambda weights: np.sum(weights) - 0.99}]  # Ensure sum close to 1

        result = minimize(objective, initial_weights, method='COBYLA', constraints=constraints, options={'maxiter': 10000, 'tol': 0.0001})
        if result.success:
            self.optimal_weights = result.x / np.sum(result.x)
        else:
            self.optimal_weights = None
    # ----------------------------------------------------------------------------------------------------  

    # 1ST QAA STRATEGY: "MIN VARIANCE"
    def minimum_variance(self, weights):
        """Minimum variance strategy."""
        return np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
    
    # ----------------------------------------------------------------------------------------------------  

    # 2ND QAA STRATEGY: "OMEGA"
    def omega_ratio(self, weights, threshold=0.0):
        """Strategy based on the Omega Ratio."""
        portfolio_returns = np.dot(self.returns, weights)
        excess_returns = portfolio_returns - threshold

        gain = np.sum(excess_returns[excess_returns > 0])
        loss = -np.sum(excess_returns[excess_returns < 0])

        if loss == 0:
            return np.inf  # Avoid division by zero
        return gain / loss
    # ----------------------------------------------------------------------------------------------------  

    # 3RD QAA STRATEGY: "SEMIVARIANCE"
    def semivariance(self):
        diff = self.returns.subtract(self.benchmark_returns, axis=0)
        negative_diff = diff.copy()
        negative_diff[negative_diff > 0] = 0
        downside_risk = negative_diff.std()

        downside_risk_df = downside_risk.to_frame()
        downside_risk_transposed = downside_risk_df.T
        mmult = np.dot(downside_risk_df, downside_risk_transposed)
        correlation = self.returns.corr()
        semi_var_matrix = (mmult * correlation) * 100

        # Define the objective function to minimize the total semivariance of the portfolio
        semivariance = lambda w: np.dot(w.T, np.dot(semi_var_matrix, w))

        return semivariance
    # ----------------------------------------------------------------------------------------------------  

    # 4TH QAA STRATEGY: "Black Litterman"
    def black_litterman(self, weights, expected_returns=None, opinions_p=None, tau=0.025):
        # Asumir retornos históricos incrementados si no se especifican retornos esperados
        if expected_returns is None:
            expected_returns = self.returns.mean() * 1.05

        # Utilizar una matriz de identidad si no se proporcionan opiniones específicas
        if opinions_p is None:
            opinions_p = np.eye(self.returns.shape[1])  # Asegurar que tiene la misma dimensión que el número de activos

        # Convertir expected_returns a un array numpy si aún no lo es
        expected_returns = np.array(expected_returns)

        # Asegurarse de que las dimensiones son compatibles
        if expected_returns.shape[0] != opinions_p.shape[1]:
            raise ValueError("Dimension mismatch between 'expected_returns' and the number of columns in 'opinions_p'.")

        # Calcular Omega y otros parámetros necesarios
        Omega = np.diag(np.full(opinions_p.shape[0], 0.1))  # Matriz diagonal para la incertidumbre en las opiniones

        # Datos de entrada
        cov = self.returns.cov().values  # Convertir a numpy array si es necesario
        tau_cov = tau * cov

        # Calculando la inversa necesaria para el modelo
        inv = np.linalg.inv(opinions_p.dot(tau_cov).dot(opinions_p.T) + Omega)
        
        # Ajustar returns.mean() a un vector fila
        mean_returns = self.returns.mean().values.reshape(-1, 1)  # Reshape para asegurar dimensiones correctas

        # Calcular posterior_mu según Black-Litterman
        adjusted_returns = expected_returns.reshape(-1, 1) - opinions_p.dot(mean_returns)
        posterior_mu = mean_returns.flatten() + tau_cov.dot(opinions_p.T).dot(inv).dot(adjusted_returns).flatten()

        # Asegurar que posterior_mu tiene las dimensiones correctas para el cálculo final
        if posterior_mu.shape[0] != weights.shape[0]:
            raise ValueError("Dimension mismatch in final calculation.")

        posterior_cov = cov + tau_cov - tau_cov.dot(opinions_p.T).dot(inv).dot(opinions_p).dot(tau_cov)

        # Función objetivo
        objective_function = -weights.dot(posterior_mu) + 0.5 * tau * weights.dot(posterior_cov).dot(weights)

        return objective_function




    # ----------------------------------------------------------------------------------------------------

    # 5TH QAA STRATEGY: "ROY SAFETY FIRST RATIO"
    def roy_safety_first_ratio(self, weights):
        """Roy's Safety-First Ratio strategy."""
        expected_return = np.dot(self.returns.mean() * 252, weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        return -(expected_return - self.rf) / volatility

    # ----------------------------------------------------------------------------------------------------  

    # 6TH QAA STRATEGY: "SORTINO RATIO"
    def sortino_ratio(self, weights):
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        downside_returns = self.returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = np.sqrt(np.sum((downside_returns.std() * weights)**2) * 252)
        sortino_ratio = (portfolio_return - self.rf) / downside_std
        return -sortino_ratio  

    # ----------------------------------------------------------------------------------------------------  
    # 7TH QAA STRATEGY: "FAMA FRENCH"
    def fama_french(self, weights):
        """Optimizes the objective function using Fama-French factors."""
        all_returns = pd.concat([self.returns, self.ff_returns], axis=1)
        risk_free_rate = self.ff_data['RF'].mean()
        ff_weights = np.append(weights, np.zeros(3))  
        portfolio_returns = np.dot(all_returns, ff_weights)
        ff_covariance = np.cov(all_returns.T)
        portfolio_volatility = np.sqrt(np.dot(ff_weights.T, np.dot(ff_covariance, ff_weights)))
        ff_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / portfolio_volatility
        return -ff_ratio
    
    # ---------------------------------------------------------------------------------------------------- 

    # 8TH QAA STRATEGY: "CVAR"
    def cvar(self, weights, alpha=0.05):
        """CVaR (Conditional Value at Risk) strategy."""
        portfolio_returns = np.dot(self.returns, weights)
        VaR = np.percentile(portfolio_returns, alpha * 100)
        CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
        return -CVaR  # Negative because we want to minimize CVaR


   # ---------------------------------------------------------------------------------------------------- 

    # 9TH QAA STRATEGY: "HIERARCHICAL RISK PARITY"
    def optimize_hrp(self):
        """Optimizes using HRP."""
        hrp = HierarchicalRiskParity(self.returns)
        hrp_weights = hrp.optimize_hrp()
        return hrp_weights
    # ----------------------------------------------------------------------------------------------------  

    # 10TH QAA STRATEGY: "SHARPE RATIO"
    def sharpe_ratio(self,weights):
        """Strategy based on the Sortino Ratio."""
        portfolio_return = np.sum(self.returns.mean() * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.rf/100) * 252 / portfolio_volatility
        return -sharpe_ratio


    # ----------------------------------------------------------------------------------------------------  


    # 11TH QAA STRATEGY: "Total Return AA"
    def Total_return(self, weights, lambda_a=1):
            """
            Calcula el ratio de Sharpe modificado para una asignación de pesos dada.

            Parameters:
            - weights (numpy.array): Pesos de asignación de cartera.

            Returns:
            - float: Valor de los pesos de Total Return AA.
            """
            # Calcula los rendimientos del portafolio
            portfolio_returns = np.dot(self.returns, weights)

            # Calcula la volatilidad del portafolio
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))

            # Calcula el rendimiento esperado de la cartera (tau)
            portfolio_expected_return = np.dot(weights, self.returns.mean())

            rf = self.rf
            benchmark_returns = self.benchmark_returns.mean()

            # Calcula el ratio de Sharpe modificado utilizando la fórmula
            objective_function = (np.dot(weights, self.returns.mean()) - rf) / (
                        benchmark_returns - rf + lambda_a * np.sqrt(
                    np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))))

            return objective_function

    # ----------------------------------------------------------------------------------------------------


    # FINAL OPTIMIZE FUNCTION
    def optimize(self):
        """Executes the selected optimization strategy and model."""
        # Define the objective function based on the chosen strategy
        if self.optimization_strategy == 'Minimum Variance':
            self.objective_function = self.minimum_variance
        elif self.optimization_strategy == 'Omega Ratio':
            # Here it is assumed that Omega Ratio is maximized, so we minimize its negative
            self.objective_function = lambda weights: -self.omega_ratio(weights, self.rf)
        elif self.optimization_strategy == 'Semivariance':
            self.objective_function = self.semivariance()
        elif self.optimization_strategy == 'Roy Safety First Ratio':
            self.objective_function = self.roy_safety_first_ratio
        elif self.optimization_strategy == 'Sortino Ratio':
             self.objective_function = self.sortino_ratio
        elif self.optimization_strategy == 'Fama French':
            self.objective_function = self.fama_french     
        elif self.optimization_strategy == 'CVaR':
            self.objective_function = lambda weights: self.cvar(weights, alpha=0.05)
        elif self.optimization_strategy == 'HRP': 
            self.optimal_weights = self.optimize_hrp()
            return
        elif self.optimization_strategy == 'Sharpe Ratio':
             self.objective_function = self.sharpe_ratio
        elif self.optimization_strategy == 'Black Litterman':
            self.objective_function = self.black_litterman
        elif self.optimization_strategy == 'Total Return':
            self.objective_function = self.Total_return
        else:
            raise ValueError("Invalid optimization strategy.")

        # Execute the selected optimization model
        if self.optimization_model == 'SLSQP':
            self.optimize_slsqp()
        elif self.optimization_model == 'Monte Carlo':
            self.optimize_montecarlo()
        elif self.optimization_model == 'COBYLA':
            self.optimize_cobyla()
        else:
            raise ValueError("Invalid optimization model.")