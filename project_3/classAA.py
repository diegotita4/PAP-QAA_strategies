# Librerias

import numpy as np
from scipy.optimize import minimize, BFGS
import numpy as np
import pandas as pd
import yfinance as yf


# Clase AA
class PortfolioOptimizer:
    def __init__(self, tickers, risk_free_rate=0.0, bounds=None):
        """
        Initializes the Portfolio Optimizer object with given parameters.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols for assets in the portfolio.
        risk_free_rate : float, optional
            Risk-free rate of return, default is 0.0.
        bounds : list of tuple, optional
            Bounds for asset weights in the optimization, each tuple is (min, max) for an asset.
        """
        self.tickers = tickers
        self.rf = risk_free_rate
        self.bounds = bounds if bounds is not None else [(0.10, 1.0) for _ in range(len(tickers))]
        self.returns, self.cov_matrix, self.num_assets = self.get_portfolio_data(tickers)

    def get_portfolio_data(self, tickers):
        """
        Fetches and prepares portfolio data from Yahoo Finance.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols for assets in the portfolio.

        Returns
        -------
        tuple
            Returns, covariance matrix, and number of assets in the portfolio.
        """
        data = yf.download(tickers, start="2020-01-01", end="2023-01-23")['Adj Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov().to_numpy()
        num_assets = len(tickers)
        return returns, cov_matrix, num_assets

    def _calculate_portfolio_metrics(self, weights):
        """
        Calculates portfolio metrics: return, volatility, and Sharpe ratio.

        Parameters
        ----------
        weights : ndarray
            Asset weights in the portfolio.

        Returns
        -------
        tuple
            Portfolio return, volatility, and Sharpe ratio.
        """
        port_return = np.dot(weights, self.returns.mean()) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (port_return - self.rf) / port_volatility
        return port_return, port_volatility, sharpe_ratio

    def _calculate_omega_ratio(self, weights, threshold_return=0.0):
        """
        Calculates the Omega Ratio for a given portfolio.

        Parameters
        ----------
        weights : ndarray
            Asset weights in the portfolio.
        threshold_return : float, optional
            The threshold return for calculating the Omega Ratio, default is 0.0.

        Returns
        -------
        float
            The Omega Ratio of the portfolio.
        """
        port_return, port_volatility, _ = self._calculate_portfolio_metrics(weights)
        excess_return = port_return - threshold_return
        if excess_return > 0:
            omega_ratio = excess_return / port_volatility
        else:
            omega_ratio = -port_volatility
        return omega_ratio



    def optimize_portfolio(self, optimization_method='SLSQP', optimization_type='sharpe', threshold_return=0.0):
        """
        Optimizes the portfolio using the specified method and optimization type.

        Parameters
        ----------
        optimization_method : str
            Optimization method to use ('SLSQP', 'MonteCarlo', 'GradientDescent').
        optimization_type : str
            Type of optimization ('sharpe', 'variance', 'omega').
        threshold_return : float, optional
            Threshold return for Omega Ratio calculation, relevant if optimization_type is 'omega'.

        Returns
        -------
        ndarray
            Optimized asset weights for the portfolio.
        """
        if optimization_method == 'SLSQP':
            return self._slsqp_optimization(optimization_type, threshold_return)
            raise ValueError("Unsupported optimization method")

    def _slsqp_optimization(self, optimization_type, threshold_return=0.0):
        """
        Performs SLSQP optimization for the portfolio.

        Parameters
        ----------
        optimization_type : str
            Type of optimization ('sharpe', 'variance', 'omega').
        threshold_return : float
            Threshold return for Omega Ratio calculation, relevant if optimization_type is 'omega'.

        Returns
        -------
        ndarray
            Optimized asset weights for the portfolio.
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights must equal 1
        ]

        if optimization_type == 'sharpe':
            objective = lambda x: -self._calculate_portfolio_metrics(x)[2]
        elif optimization_type == 'variance':
            objective = lambda x: self._calculate_portfolio_metrics(x)[1]
        elif optimization_type == 'omega':
            objective = lambda x: -self._calculate_omega_ratio(x, threshold_return)

        result = minimize(
            objective,
            x0=np.array([1.0/self.num_assets] * self.num_assets),  # Initial guess
            method='SLSQP',
            bounds=self.bounds,
            constraints=constraints
        )

        return result.x




# Ejemplo de uso
    
if __name__ == "__main__":
    # Definir los tickers y parámetros iniciales
    tickers = ['ABBV', 'MET', 'OXY', 'PERI']
    risk_free_rate = 0.055 / 252  # Tasa libre de riesgo ajustada diariamente
    optimizer = PortfolioOptimizer(tickers, risk_free_rate=risk_free_rate, bounds=[(0.1, 1.0) for _ in tickers])

    # Función para imprimir los resultados
    def print_optimized_weights(tickers, weights, optimization_type, method):
        print(f"Optimized Weights for {optimization_type} using {method}:")
        for ticker, weight in zip(tickers, weights):
            print(f"{ticker}: {weight*100:.2f}%")
        print("\n")  # Agrega una línea en blanco para separar los resultados

    # Lista de métodos de optimización y tipos de optimización
    optimization_methods = ['SLSQP'] #'MonteCarlo', 'GradientDescent'
    optimization_types = ['sharpe', 'variance', 'omega']

    # Ejecutar la optimización para cada combinación de método y tipo
    for method in optimization_methods:
        for opt_type in optimization_types:
            optimized_weights = optimizer.optimize_portfolio(optimization_method=method, optimization_type=opt_type)
            print_optimized_weights(tickers, optimized_weights, opt_type, method)

    # Nota: Asegúrate de tener implementaciones completas para monte_carlo_optimization y gradient_descent_optimization
