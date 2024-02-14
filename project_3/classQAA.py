
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
import warnings
# quitar warnings....
warnings.simplefilter(action='ignore', category=FutureWarning)


class QAA:
    def __init__(self, tickers, start, end, benchmark='SPY', risk_free_rate=0.0, bounds=None, initial_cash=1000000):
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
        
        # Regresa la metrica omega en base a los modelos de optimizacion vistos en el código
        return omega_ratio
    

    # Metodo de semivarianza , parte 1 calcular el downside
    def calculate_downside_risks(self):
        benchmark_data = yf.download(self.benchmark, start=self.start, end=self.end)['Adj Close']
        benchmark_returns = benchmark_data.pct_change().dropna()
        diff = self.returns.subtract(benchmark_returns, axis=0)
        diff_neg = diff.copy()
        diff_neg[diff_neg > 0] = 0
        downside_r = diff_neg.std()
        return downside_r
    
    # Matiz de semivarianza
    def semi_variance_matrix(self):
        downside_r_df = self.downside_r.to_frame()
        downside_r_transposed = downside_r_df.T
        mmult = np.dot(downside_r_df, downside_r_transposed)
        mmult_df = pd.DataFrame(mmult, self.tickers, self.tickers)
        correlacion = self.returns.corr()
        semi_var = (mmult_df * correlacion) * 100
        return semi_var
    

    # Nuevo método para maximizar la ratio de Sharpe
    def max_sharpe(self, optimization_method='SLSQP'):
        if optimization_method == 'SLSQP':
            return self._slsqp_optimization(optimization_type='sharpe')
        elif optimization_method == 'MonteCarlo':
            return self.monte_carlo_optimization(optimization_type='sharpe')
        # Puedes añadir más métodos de optimización aquí
        else:
            raise ValueError("Unsupported optimization method")
    
    # Nuevo método para minimizar la varianza
    def min_variance(self, optimization_method='SLSQP'):
        if optimization_method == 'SLSQP':
            return self._slsqp_optimization(optimization_type='variance')
        elif optimization_method == 'MonteCarlo':
            return self.monte_carlo_optimization(optimization_type='variance')
        # Puedes añadir más métodos de optimización aquí
        else:
            raise ValueError("Unsupported optimization method")
        
    def calculate_sortino_ratio(self, weights):
        port_return = np.dot(weights, self.returns.mean()) * 252
        downside_risk = self.returns[self.returns<0].std()*np.sqrt(252)
        sortino_ratio = (port_return - self.rf) / downside_risk
        return sortino_ratio

    def monte_carlo_optimization(self, optimization_goal='sharpe', num_portfolios=10000):
        """
        Realiza la optimización de Monte Carlo para el portafolio.

        Parameters
        ----------
        optimization_goal : str, optional
            Objetivo de la optimización ('sharpe', 'min_variance', 'omega_ratio'), por defecto es 'sharpe'.
        num_portfolios : int, optional
            Número de portafolios a simular, por defecto es 10000.

        Returns
        -------
        ndarray
            Pesos optimizados para el portafolio.
        """

        results = np.zeros((3, num_portfolios))
        all_weights = np.zeros((num_portfolios, self.num_assets))

        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            port_return, port_volatility = self._calculate_portfolio_metrics(weights)

            if optimization_goal == 'sharpe':
                sharpe_ratio = self.calculate_sharpe_ratio(port_return, port_volatility)
                metric = sharpe_ratio
            elif optimization_goal == 'min_variance':
                metric = port_volatility
            elif optimization_goal == 'omega_ratio':
                # Asume que tienes una función para calcular la ratio de Omega
                omega_ratio = self._calculate_omega_ratio(weights, threshold_return=0.0)  # Ajusta threshold_return si necesario
                metric = omega_ratio
            else:
                raise ValueError(f"Unsupported optimization goal: {optimization_goal}")

            # Almacenar resultados
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = metric
            all_weights[i, :] = weights

        # Encontrar el índice del mejor resultado basado en el objetivo de optimización
        if optimization_goal in ['sharpe', 'omega_ratio']:
            best_idx = np.argmax(results[2])
        elif optimization_goal == 'min_variance':
            best_idx = np.argmin(results[2])

        return all_weights[best_idx]

    

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


    def optimize_portfolio(self, optimization_goal='sharpe', optimization_method='SLSQP'):
        """
        Optimiza el portafolio basado en el objetivo y el método de optimización seleccionados.

        Parameters
        ----------
        optimization_goal : str
            Objetivo de la optimización ('omega_ratio', 'sharpe', 'min_variance').
        optimization_method : str
            Método de optimización a utilizar ('MonteCarlo', 'SLSQP', 'GradientDescent').
        """
        if optimization_method == 'SLSQP':
            optimized_weights = self._slsqp_optimization(optimization_goal)
        elif optimization_method == 'MonteCarlo':
            optimized_weights = self.monte_carlo_optimization(optimization_goal)
        elif optimization_method == 'GradientDescent':
            optimized_weights = self.gradient_descent_optimization(optimization_goal)
        else:
            raise ValueError("Unsupported optimization method")

        return optimized_weights


    def _slsqp_optimization(self, optimization_goal, threshold_return=0.0, user_defined_bounds=None):
        """
        Realiza la optimización SLSQP para el portafolio.

        Parameters
        ----------
        optimization_type : str
            Tipo de optimización ('sharpe', 'variance', 'omega').
        threshold_return : float
            Retorno umbral para el cálculo de la ratio de Omega, relevante si optimization_type es 'omega'.
        user_defined_bounds : list of tuple, optional
            Límites definidos por el usuario para los pesos de los activos en el portafolio. Cada tupla es (min, max).

        Returns
        -------
        ndarray
            Pesos optimizados para el portafolio.
        """

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # La suma de los pesos debe ser igual a 1

        # Define la función objetivo basada en el tipo de optimización
        if optimization_goal == 'sharpe':
            objective = lambda x: -self.calculate_sharpe_ratio(*self._calculate_portfolio_metrics(x))
        elif optimization_goal == 'min_variance':
            objective = lambda x: self._calculate_portfolio_metrics(x)[1]
        elif optimization_goal == 'omega_ratio':
            objective = lambda x: -self._calculate_omega_ratio(x, threshold_return)
        #elif optimization_goal == 'sortino':
        elif optimization_goal == 'semi_variance':
            # objective = lambda x: self.calulate_semi_varinace
            pass
        else:
            raise ValueError(f"Optimization goal '{optimization_goal}' not supported.")

        # Utiliza los límites definidos por el usuario si están disponibles, de lo contrario utiliza los límites predeterminados
        bounds_to_use = user_defined_bounds if user_defined_bounds is not None else self.bounds

        result = minimize(
            objective,
            x0=np.array([1.0 / self.num_assets] * self.num_assets),
            method='SLSQP',
            bounds=bounds_to_use,
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
    date_end = "2020-12-31"
    # Bounds (restricciones)
    lower_bounds = 0.1  # Minimo en cada activo cubra un 10% de participacion en el peso
    upper_bounds = 1.0  
    risk_free_rate = 0  # Tasa libre de riesgo ajustada diariamente
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
optimized_weights_sharpe = optimizer.optimize_portfolio(optimization_goal='sharpe', optimization_method='SLSQP')
optimizer.allocate_cash_to_assets(optimized_weights_sharpe)
print_optimized_weights(tickers, optimized_weights_sharpe, "Maximum Sharpe Ratio (SLSQP)")

# Optimización para la mínima varianza usando SLSQP
optimized_weights_variance = optimizer.optimize_portfolio(optimization_goal='min_variance', optimization_method='SLSQP')
optimizer.allocate_cash_to_assets(optimized_weights_variance)
print_optimized_weights(tickers, optimized_weights_variance, "Minimum Variance (SLSQP)")

# Optimización para el máximo ratio Omega usando SLSQP
optimized_weights_omega = optimizer.optimize_portfolio(optimization_goal='omega_ratio', optimization_method='SLSQP')
optimizer.allocate_cash_to_assets(optimized_weights_omega)
print_optimized_weights(tickers, optimized_weights_omega, "Omega Ratio (SLSQP)")

# Optimización para el máximo ratio de Sharpe usando MonteCarlo
optimized_weights_sharpe_mc = optimizer.optimize_portfolio(optimization_goal='sharpe', optimization_method='MonteCarlo')
optimizer.allocate_cash_to_assets(optimized_weights_sharpe_mc)
print_optimized_weights(tickers, optimized_weights_sharpe_mc, "Maximum Sharpe Ratio (MonteCarlo)")

# Optimización para la mínima varianza usando MonteCarlo
optimized_weights_variance_mc = optimizer.optimize_portfolio(optimization_goal='min_variance', optimization_method='MonteCarlo')
optimizer.allocate_cash_to_assets(optimized_weights_variance_mc)
print_optimized_weights(tickers, optimized_weights_variance_mc, "Minimum Variance (MonteCarlo)")

# Optimización para el máximo ratio Omega usando MonteCarlo
optimized_weights_omega_mc = optimizer.optimize_portfolio(optimization_goal='omega_ratio', optimization_method='MonteCarlo')
optimizer.allocate_cash_to_assets(optimized_weights_omega_mc)
print_optimized_weights(tickers, optimized_weights_omega_mc, "Omega Ratio (MonteCarlo)")

# Optimización para el máximo ratio de Sharpe usando GradientDescent
optimized_weights_sharpe_gd = optimizer.optimize_portfolio(optimization_goal='sharpe', optimization_method='GradientDescent')
optimizer.allocate_cash_to_assets(optimized_weights_sharpe_gd)
print_optimized_weights(tickers, optimized_weights_sharpe_gd, "Maximum Sharpe Ratio (GradientDescent)")

# Optimización para la mínima varianza usando GradientDescent
optimized_weights_variance_gd = optimizer.optimize_portfolio(optimization_goal='min_variance', optimization_method='GradientDescent')
optimizer.allocate_cash_to_assets(optimized_weights_variance_gd)
print_optimized_weights(tickers, optimized_weights_variance_gd, "Minimum Variance (GradientDescent)")

# Optimización para el máximo ratio Omega usando GradientDescent
optimized_weights_omega_gd = optimizer.optimize_portfolio(optimization_goal='omega_ratio', optimization_method='GradientDescent')
optimizer.allocate_cash_to_assets(optimized_weights_omega_gd)
print_optimized_weights(tickers, optimized_weights_omega_gd, "Omega Ratio (GradientDescent)")

# Optimización para el máximo ratio Sortino usando SLSQP
#optimized_weights_sortino = optimizer.optimize_portfolio(optimization_goal='sortino', optimization_method='SLSQP')
#optimizer.allocate_cash_to_assets(optimized_weights_sortino)
#print_optimized_weights(tickers, optimized_weights_sortino, "Sortino Ratio (SLSQP)")

# Optimización para el máximo ratio Sortino usando GradientDescent
#optimized_weights_sortino_gd = optimizer.optimize_portfolio(optimization_goal='sortino', optimization_method='GradientDescent')
#optimizer.allocate_cash_to_assets(optimized_weights_sortino_gd)
#print_optimized_weights(tickers, optimized_weights_sortino_gd, "Sortino Ratio (GradientDescent)")

# Optimización para el máximo ratio de Sortino usando MonteCarlo
#optimized_weights_sortino_mc = optimizer.optimize_portfolio(optimization_goal='sortino', optimization_method='MonteCarlo')
#optimizer.allocate_cash_to_assets(optimized_weights_sortino_mc)
#print_optimized_weights(tickers, optimized_weights_sortino_mc, "Maximum Sortino Ratio (MonteCarlo)")
