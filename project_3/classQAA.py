
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- authors: YOUR GITHUB USER NAME                                                                      -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#Librerias 
import numpy as np
from scipy.optimize import minimize, BFGS
import numpy as np
import pandas as pd
import yfinance as yf

class QAA:
    def __init__(self, tickers, start, end, risk_free_rate=0.0, bounds=None, initial_cash=1000000):
        """
        Initializes the Portfolio Optimizer object with given parameters.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols for assets in the portfolio.
        start : str
            Start date for the data retrieval in the format 'YYYY-MM-DD', 
            indicating the beginning of the period over which the asset values are considered.
        end : str
            End date for the data retrieval in the format 'YYYY-MM-DD', 
            marking the end of the period over which the asset values are considered.
        risk_free_rate : float, optional
            Risk-free rate of return, default is 0.0.
        bounds : list of tuple, optional
            Bounds for asset weights in the optimization, each tuple is (min, max) for an asset.
        initial_cash : int, optional
            Amount of cash to start the portafolio
        """

        self.tickers = tickers
        self.rf = risk_free_rate
        self.start = start
        self.end = end
        self.bounds = bounds if bounds is not None else [(0.10, 1.0) for _ in range(len(tickers))]
        self.initial_cash = initial_cash
        self.returns, self.cov_matrix, self.num_assets = self.get_portfolio_data(tickers, start, end)
        self.asset_allocations = []  # Inicializar la lista para guardar las asignaciones de efectivo para cada activo


    def get_portfolio_data(self, tickers, start, end):
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

        data = yf.download(tickers, start=start, end=end)['Adj Close']
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
        return port_return, port_volatility



    def calculate_sharpe_ratio(self, port_return, port_volatility):
        """
        Calcula la ratio de Sharpe para el portafolio dado.

        Parameters
        ----------
        port_return : float
            Retorno esperado anualizado del portafolio.
        port_volatility : float
            Volatilidad anualizada del portafolio.

        Returns
        -------
        float
            La ratio de Sharpe del portafolio.
        """
        sharpe_ratio = (port_return - self.rf) / port_volatility
        return sharpe_ratio

    def negative_sharpe_ratio(weights, returns, cov_matrix, rf):
        port_return = np.dot(weights, returns.mean()) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (port_return - rf) / port_volatility
        return -sharpe_ratio  # Negativo para minimización (Maximizamos sharpe sacando el negativo del minimo) #Revisar esta logica.....



    def _calculate_omega_ratio(self, weights, threshold_return=0.0):
        """
        Calcula la ratio de Omega para un portafolio dado.

        Parameters
        ----------
        weights : ndarray
            Pesos de los activos en el portafolio.
        threshold_return : float, opcional
            El retorno umbral para calcular la ratio de Omega, por defecto es 0.0.

        Returns
        -------
        float
            La ratio de Omega del portafolio.
        """
        portfolio_returns = np.dot(self.returns, weights)
        gains_above_threshold = portfolio_returns[portfolio_returns > threshold_return] - threshold_return
        losses_below_threshold = threshold_return - portfolio_returns[portfolio_returns <= threshold_return]
        
        if losses_below_threshold.sum() == 0:
            return float('inf')  # Retornar infinito si no hay pérdidas, indicando una ratio de Omega muy favorable
        
        omega_ratio = gains_above_threshold.sum() / losses_below_threshold.sum() if losses_below_threshold.sum() != 0 else np.nan
        
        # Minimizar la inversa de la ratio de Omega para la optimización
        return -omega_ratio



    def monte_carlo_optimization(self, num_portfolios=10000, optimization_type='sharpe', threshold_return=0.0):
        """
        Performs Monte Carlo optimization for the portfolio.

        Parameters
        ----------
        num_portfolios : int, optional
            Number of portfolios to simulate, default is 10000.
        optimization_type : str, optional
            Type of optimization ('sharpe', 'variance', 'omega'), default is 'sharpe'.
        threshold_return : float, optional
            Threshold return for Omega Ratio calculation, relevant if optimization_type is 'omega'.

        Returns
        -------
        ndarray
            Optimized asset weights for the portfolio.
        """

        results = np.zeros((4, num_portfolios))
        all_weights = np.zeros((num_portfolios, self.num_assets))

        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            port_return, port_volatility, sharpe_ratio = self._calculate_portfolio_metrics(weights)
            omega_ratio = self._calculate_omega_ratio(weights, threshold_return)

            all_weights[i, :] = weights
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = sharpe_ratio
            results[3, i] = omega_ratio

        if optimization_type == 'sharpe':
            max_sharpe_idx = np.argmax(results[2])
            return all_weights[max_sharpe_idx]
        elif optimization_type == 'variance':
            min_variance_idx = np.argmin(results[1])
            return all_weights[min_variance_idx]
        elif optimization_type == 'omega':
            max_omega_idx = np.argmax(results[3])
            return all_weights[max_omega_idx]
        else:
            raise ValueError("Unsupported optimization type")
        return weights
    

    def gradient_descent_optimization(self, optimization_type='sharpe', learning_rate=0.01, iterations=100):
            """
            Performs a basic gradient descent optimization for the portfolio.

            Parameters
            ----------
            optimization_type : str, optional
                Type of optimization, default is 'sharpe'.
            learning_rate : float, optional
                Learning rate for the optimization, default is 0.01.
            iterations : int, optional
                Number of iterations to perform, default is 100.

            Returns
            -------
            ndarray
                Optimized asset weights for the portfolio.
            """

            # Inicializar pesos aleatorios
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)

            for _ in range(iterations):
                # Computar el gradiente aquí (esto es un placeholder, necesitas implementar tu propio gradiente)
                gradient = np.zeros(self.num_assets)  # Esto es un placeholder
                # Actualizar pesos basados en el gradiente
                weights -= learning_rate * gradient
                # Mantener los pesos dentro de los límites
                weights = np.clip(weights, 0, 1)
                # Normalizar los pesos para que sumen 1
                weights /= np.sum(weights)
            return weights    


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
        elif optimization_method == 'MonteCarlo':
            return self.monte_carlo_optimization(num_portfolios=10000, optimization_type=optimization_type)
        elif optimization_method == 'GradientDescent':
            return self.gradient_descent_optimization(optimization_type=optimization_type, learning_rate=0.01, iterations=100)
        else:
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
    

    def allocate_cash_to_assets(self, weights):
        """
        Allocates initial cash to assets based on optimized weights and saves the allocations.
        
        Parameters
        - weights: np.array
            the optimized asset weights for the portafolio 

        Returns
        -------
        ndarray
            Weights for the portfolio in cash.
        """

        # Amount in each asset
        investment_amounts = self.initial_cash * weights
        
        # Save the amount 
        self.asset_allocations = [{'ticker': ticker, 'allocation': amount} for ticker, amount in zip(self.tickers, investment_amounts)]
        
        # Opcionalmente, imprimir las asignaciones para verificación
        for allocation in self.asset_allocations:
            print(f"{allocation['ticker']}: ${allocation['allocation']:.2f}")
    


    


if __name__ == "__main__":
    # Definir los tickers y parámetros iniciales
    tickers = ['ABBV', 'MET', 'OXY', 'PERI']
    date_start = "2020-01-01"
    date_end = "2023-01-01"
    # Bounds (restricciones)
    lower_bounds = 0.1  # Minimo en cada activo cubra un 10% de participacion en el peso
    upper_bounds = 1.0  
    risk_free_rate = 0.055 / 252  # Tasa libre de riesgo ajustada diariamente
    optimizer = QAA(tickers, date_start, date_end, risk_free_rate=risk_free_rate, bounds=[(lower_bounds, upper_bounds) for _ in tickers])

    # Función para imprimir los resultados
    def print_optimized_weights(tickers, weights, optimization_type):
        print(f"Optimized Weights for {optimization_type}:")
        for ticker, weight in zip(tickers, weights):
            print(f"{ticker}: {weight*100:.2f}%")
        print("\n")  # Agrega una línea en blanco para separar los resultados

    def allocate_cash_to_assets(self, weights):
        """
        Allocates initial cash to assets based on optimized weights and saves the allocations.
        
        Parameters
        - weights: np.array
            the optimized asset weights for the portafolio 

        Returns
        -------
        ndarray
            Weights for the portfolio in cash.
        """

        # Calcular los montos de inversión para cada activo
        investment_amounts = self.initial_cash * weights
        
        # Guardar las asignaciones en una estructura
        self.asset_allocations = [{'ticker': ticker, 'allocation': amount} for ticker, amount in zip(self.tickers, investment_amounts)]
        
        # print (at least for this moment)
        for allocation in self.asset_allocations:
            print(f"{allocation['ticker']}: ${allocation['allocation']:.2f}")


    # Optimización para el máximo ratio de Sharpe usando SLSQP
    optimized_weights_sharpe = optimizer.optimize_portfolio(optimization_method='SLSQP', optimization_type='sharpe')
    optimizer.allocate_cash_to_assets(optimized_weights_sharpe)
    print_optimized_weights(tickers, optimized_weights_sharpe, "Maximum Sharpe Ratio (SLSQP)")

    # Optimización para la mínima varianza usando SLSQP
    optimized_weights_variance = optimizer.optimize_portfolio(optimization_method='SLSQP', optimization_type='variance')
    optimizer.allocate_cash_to_assets(optimized_weights_variance)
    print_optimized_weights(tickers, optimized_weights_variance, "Minimum Variance (SLSQP)")

    # Optimización para el máximo ratio Omega usando SLSQP
    optimized_weights_omega = optimizer.optimize_portfolio(optimization_method='SLSQP', optimization_type='omega')
    optimizer.allocate_cash_to_assets(optimized_weights_omega)
    print_optimized_weights(tickers, optimized_weights_omega, "Maximum Omega Ratio (SLSQP)")

     # Optimización para el máximo ratio de Sharpe usando MonteCarlo
    optimized_weights_sharpe = optimizer.optimize_portfolio(optimization_method='MonteCarlo', optimization_type='sharpe')
    optimizer.allocate_cash_to_assets(optimized_weights_sharpe)
    print_optimized_weights(tickers, optimized_weights_sharpe, "Maximum Sharpe Ratio (MonteCarlo)")

    # Optimización para la mínima varianza usando MonteCarlo
    optimized_weights_variance = optimizer.optimize_portfolio(optimization_method='MonteCarlo', optimization_type='variance')
    optimizer.allocate_cash_to_assets(optimized_weights_variance)
    print_optimized_weights(tickers, optimized_weights_variance, "Minimum Variance (MonteCarlo)")

    # Optimización para el máximo ratio Omega usando MonteCarlo
    optimized_weights_omega = optimizer.optimize_portfolio(optimization_method='MonteCarlo', optimization_type='omega')
    optimizer.allocate_cash_to_assets(optimized_weights_omega)
    print_optimized_weights(tickers, optimized_weights_omega, "Maximum Omega Ratio (MonteCarlo)")

      # Optimización para el máximo ratio de Sharpe usando SLSQP
    optimized_weights_sharpe = optimizer.optimize_portfolio(optimization_method='GradientDescent', optimization_type='sharpe')
    optimizer.allocate_cash_to_assets(optimized_weights_sharpe)
    print_optimized_weights(tickers, optimized_weights_sharpe, "Maximum Sharpe Ratio (GradientDescent)")

    # Optimización para la mínima varianza usando SLSQP
    optimized_weights_variance = optimizer.optimize_portfolio(optimization_method='GradientDescent', optimization_type='variance')
    optimizer.allocate_cash_to_assets(optimized_weights_variance)
    print_optimized_weights(tickers, optimized_weights_variance, "Minimum Variance (GradientDescent)")

    # Optimización para el máximo ratio Omega usando SLSQP
    optimized_weights_omega = optimizer.optimize_portfolio(optimization_method='GradientDescent', optimization_type='omega')
    optimizer.allocate_cash_to_assets(optimized_weights_omega)
    print_optimized_weights(tickers, optimized_weights_omega, "Maximum Omega Ratio (GradientDescent)")
