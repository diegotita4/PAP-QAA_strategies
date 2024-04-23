
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
import empyrical

# ----------------------------------------------------------------------------------------------------



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

        self.names = returns.columns
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
        self.link = None
        self.sort_ix = None
        self.weights = None

    def get_quasi_diagonalization(self, link):

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

        c_cov = cov.iloc[c_items, c_items]
        ivp_weights = 1. / np.diag(c_cov)
        ivp_weights /= ivp_weights.sum()
        ivp_weights = ivp_weights.reshape(-1, 1)
        cluster_variance = np.dot(np.dot(ivp_weights.T, c_cov), ivp_weights)[0, 0]
        return cluster_variance

    @staticmethod
    def bisection(items):

        new_items = [i[int(j):int(k)] for i in items for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        return new_items

    def HRP(self, cov, sort_ix):

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

    # ----------------------------------------------------------------------------------------------------

    def __init__(self, tickers=None, benchmark_ticker='SPY', rf=None, lower_bound=0.10, higher_bound=0.99, start_date=None, end_date=None):

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

# ----------------------------------------------------------------------------------------------------

    def calculate_benchmark_returns(self):

        if self.benchmark_data is not None:
            return self.benchmark_data.pct_change().dropna()

        else:
            raise ValueError("Benchmark data not found.")

# ----------------------------------------------------------------------------------------------------

    def set_optimization_strategy(self, strategy):

        self.optimization_strategy = strategy

# ----------------------------------------------------------------------------------------------------

    def set_optimization_model(self, model):

        self.optimization_model = model

# ----------------------------------------------------------------------------------------------------

    def load_data(self):

        if not self.tickers or self.benchmark_ticker is None:
            raise ValueError("You must provide a list of tickers and a benchmark ticker.")

        tickers_with_benchmark = self.tickers + [self.benchmark_ticker] if self.benchmark_ticker not in self.tickers else self.tickers
        data = yf.download(tickers_with_benchmark, start=self.start_date, end=self.end_date)['Adj Close']
        benchmark_data = data.pop(self.benchmark_ticker) if self.benchmark_ticker in data else None
        return data, benchmark_data

# ----------------------------------------------------------------------------------------------------

    def load_ff_data(self):

        ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=self.start_date, end=self.end_date)[0]
        ff_returns = ff_data[['Mkt-RF', 'SMB', 'HML']].loc[self.returns.index]
        return ff_data, ff_returns

# ----------------------------------------------------------------------------------------------------

    def calculate_returns(self):

        return self.data.pct_change().dropna()

# ----------------------------------------------------------------------------------------------------

    def validate_returns_empyrical(self):

        results = {}
        for ticker in self.tickers:
            prices = self.data[ticker]
            daily_returns = prices.pct_change().dropna()
            annualized_return = empyrical.annual_return(daily_returns)
            sharpe_ratio = empyrical.sharpe_ratio(daily_returns, risk_free=self.rf / 252)
            results[ticker] = {
                'Annualized Return': annualized_return,
                'Sharpe Ratio': sharpe_ratio
            }
        return results

# ----------------------------------------------------------------------------------------------------

# 1ST OPTIMIZE MODEL: "SLSQP"

    def optimize_slsqp(self):

        if self.optimization_strategy == 'Minimum Variance':
            objective = self.minimum_variance

        elif self.optimization_strategy == 'Omega Ratio':
            objective = lambda weights: -self.omega_ratio(weights, self.rf)

        elif self.optimization_strategy == 'Semivariance':
            objective = self.semivariance()

        elif self.optimization_strategy == 'Martingale':
            objective = self.martingale

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

# 2ND OPTIMIZE MODEL: "MONTECARLO (w/optuna)""

    def optimize_montecarlo(self):

        def objective(trial):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                weights = np.array([trial.suggest_float(f"weight_{i}", 0, 1) for i in range(len(self.tickers))])

            weights /= np.sum(weights)

            penalty = 0
            for weight in weights:
                if weight < self.lower_bound:
                    penalty += 1e6 * (self.lower_bound - weight) ** 2

            if self.optimization_strategy == 'Minimum Variance':
                objective_value = self.minimum_variance(weights) + penalty

            elif self.optimization_strategy == 'Omega Ratio':
                objective_value = -self.omega_ratio(weights, self.rf) + penalty

            elif self.optimization_strategy == 'Semivariance':
                objective_function = self.semivariance()
                objective_value = objective_function(weights) + penalty

            elif self.optimization_strategy == 'Martingale':
                objective_value = self.martingale(weights) + penalty

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
                return -CVaR + penalty

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

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
        num_trials = 500
        study.optimize(objective, n_trials=num_trials, n_jobs=-1)
        optimal_weights = np.array([study.best_params[f"weight_{i}"] for i in range(len(self.tickers))])
        optimal_weights /= np.sum(optimal_weights)
        self.optimal_weights = optimal_weights

# ----------------------------------------------------------------------------------------------------

# 3RD OPTIMIZE MODEL: "COBYLA"

    def optimize_cobyla(self):

        if self.optimization_strategy == 'Minimum Variance':
            objective = self.minimum_variance

        elif self.optimization_strategy == 'Omega Ratio':
            objective = lambda weights: -self.omega_ratio(weights, self.rf)

        elif self.optimization_strategy == 'Semivariance':
            objective = self.semivariance()

        elif self.optimization_strategy == 'Martingale':
            objective = self.martingale

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

        elif self.optimization_strategy == 'Black Litterman':
             objective = self.black_litterman

        elif self.optimization_strategy == 'Total Return':
             objective = self.Total_return

        else:
            raise ValueError("Invalid optimization strategy.")

        initial_weights = np.ones(len(self.tickers)) / len(self.tickers)
        constraints = [{'type': 'ineq', 'fun': lambda weights, i=i: weights[i] - 0.10} for i in range(len(self.tickers))]
        constraints += [{'type': 'ineq', 'fun': lambda weights: 1 - np.sum(weights)},
                        {'type': 'ineq', 'fun': lambda weights: np.sum(weights) - 0.99}]
        result = minimize(objective, initial_weights, method='COBYLA', constraints=constraints, options={'maxiter': 10000, 'tol': 0.0001})

        if result.success:
            self.optimal_weights = result.x / np.sum(result.x)

        else:
            self.optimal_weights = None

# ----------------------------------------------------------------------------------------------------  

# 1ST QAA STRATEGY: "MIN VARIANCE"

    def minimum_variance(self, weights):

        return np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
    
# ----------------------------------------------------------------------------------------------------  

# 2ND QAA STRATEGY: "OMEGA"

    def omega_ratio(self, weights, threshold=0.0):

        portfolio_returns = np.dot(self.returns, weights)
        excess_returns = portfolio_returns - threshold
        gain = np.sum(excess_returns[excess_returns > 0])
        loss = -np.sum(excess_returns[excess_returns < 0])

        if loss == 0:
            return np.inf
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
        semivariance = lambda w: np.dot(w.T, np.dot(semi_var_matrix, w))
        return semivariance

# ----------------------------------------------------------------------------------------------------  

# 4TH QAA STRATEGY: "Black Litterman"

    def black_litterman(self, weight, expected_returns=np.array([.15, .2, .25, .30]), opinions_p=np.array([[1, 0, 0, 0], [0, 1, -3, 0], [0, 0, 1, -1], [0, 0, 0, 0]]), tau=0.025):

        E_r = expected_returns
        opinions_p = opinions_p
        Omega = np.diag(np.power(E_r, 2))
        returns = self.returns
        cov = returns.cov()
        tau = tau
        posterior_mu = (returns.mean() + tau * cov.dot(opinions_p.T).dot(np.linalg.inv(opinions_p.dot(tau ** 2 * cov).dot(opinions_p.T) + Omega)).dot(E_r - opinions_p.dot(returns.mean())))
        posterior_cov = (cov + tau * cov - tau ** 2 * cov).dot(opinions_p.T).dot(np.linalg.inv(opinions_p.dot(tau ** 2 * cov).dot(opinions_p.T) + Omega)).dot(opinions_p.dot(tau ** 2 * cov))
        objective_function = -posterior_mu.dot(weight) + 0.5 * tau * weight.dot(posterior_cov).dot(weight)
        return objective_function

# ----------------------------------------------------------------------------------------------------

# 5TH QAA STRATEGY: "ROY SAFETY FIRST RATIO"

    def roy_safety_first_ratio(self, weights):

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

        portfolio_returns = np.dot(self.returns, weights)
        VaR = np.percentile(portfolio_returns, alpha * 100)
        CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
        return -CVaR

# ---------------------------------------------------------------------------------------------------- 

# 9TH QAA STRATEGY: "HIERARCHICAL RISK PARITY"

    def optimize_hrp(self):

        hrp = HierarchicalRiskParity(self.returns)
        hrp_weights = hrp.optimize_hrp()
        return hrp_weights

# ----------------------------------------------------------------------------------------------------  

# 10TH QAA STRATEGY: "SHARPE RATIO"

    def sharpe_ratio(self,weights):

        portfolio_return = np.sum(self.returns.mean() * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.rf/100) * 252 / portfolio_volatility
        return -sharpe_ratio

# ----------------------------------------------------------------------------------------------------  

# 11TH QAA STRATEGY: "Total Return AA"

    def Total_return(self, weights, lambda_a=1):

            rf = self.rf
            benchmark_returns = self.benchmark_returns.mean()
            objective_function = (np.dot(weights, self.returns.mean()) - rf) / (benchmark_returns - rf + lambda_a * np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))))
            return objective_function

# ----------------------------------------------------------------------------------------------------

# FINAL OPTIMIZE FUNCTION

    def optimize(self):

        if self.optimization_strategy == 'Minimum Variance':
            self.objective_function = self.minimum_variance

        elif self.optimization_strategy == 'Omega Ratio':
            self.objective_function = lambda weights: -self.omega_ratio(weights, self.rf)

        elif self.optimization_strategy == 'Semivariance':
            self.objective_function = self.semivariance()

        elif self.optimization_strategy == 'Martingale':
            self.objective_function = self.martingale

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

        elif self.optimization_strategy == 'Sharpe Ratio':
             self.objective_function = self.sharpe_ratio

        elif self.optimization_strategy == 'Black Litterman':
            self.objective_function = self.black_litterman

        elif self.optimization_strategy == 'Total Return':
            self.objective_function = self.Total_return

        else:
            raise ValueError("Invalid optimization strategy.")

# -----------------------------------

        if self.optimization_model == 'SLSQP':
            self.optimize_slsqp()

        elif self.optimization_model == 'Monte Carlo':
            self.optimize_montecarlo()

        elif self.optimization_model == 'COBYLA':
            self.optimize_cobyla()

        else:
            raise ValueError("Invalid optimization model.")
