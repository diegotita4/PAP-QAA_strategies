import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd  # Agregado para usar pd.to_timedelta

class PortfolioOptimizer:
    def __init__(self, tickers):
        self.returns = self.get_returns(tickers)
        self.cov_matrix = self.get_covariance_matrix(tickers)
        self.num_assets = len(tickers)
        
    def get_returns(self, tickers):
        # Utiliza yfinance para obtener los rendimientos históricos
        data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
        returns = data.pct_change().dropna()
        return returns
    
    def get_covariance_matrix(self, tickers):
        # Utiliza yfinance para obtener la matriz de covarianza
        data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov()
        return cov_matrix.to_numpy()
        
    def _constraints(self, weights):
        # Restricciones para que la suma de los pesos sea 1 y cada activo sea >= 10%
        return [np.sum(weights) - 1, *weights - 0.1]
    
    def _calculate_portfolio_metrics(self, weights):
        # Función auxiliar para calcular la volatilidad del portafolio
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return port_volatility
    
    def minimum_variance_portfolio(self):
        # Función para calcular la Mínima Varianza
        initial_weights = np.ones(self.num_assets) / self.num_assets
        result = minimize(self._calculate_portfolio_metrics, initial_weights,
                          method=None, constraints={'type': 'eq', 'fun': self._constraints})
        return result.x
    
    def maximum_sharpe_ratio_portfolio(self):
        # Función para calcular el Máximo Ratio de Sharpe
        initial_weights = np.ones(self.num_assets) / self.num_assets
        result = minimize(lambda weights: -self._calculate_portfolio_metrics(weights),
                          initial_weights, method=None, constraints={'type': 'eq', 'fun': self._constraints})
        return result.x
    
    def model_fit(self, x, y):
        def obj(coef, r_fondo, r_estilos):
            alpha = coef[0]
            w = coef[1:]
            modelo = alpha + r_estilos.dot(w)
            residuales = r_fondo - modelo
            return (residuales**2).mean()

        coef0 = [0, 0.25, 0.25, 0.25, 0.25]
        cons = {"type": "eq",
                "fun": lambda coef: coef[1:].sum() - 1}
        bnds = ((None, None), ) + ((0, None), ) * 4
        style = minimize(fun=obj,
                         x0=coef0,
                         args=(y, x),
                         bounds=bnds,
                         constraints=cons,
                         tol=1e-10)

        return style


# Ejemplo de uso con tickers
ticker_symbols = ['AAPL', 'AMZN']  # Puedes cambiar estos tickers según tus preferencias
portfolio_optimizer = PortfolioOptimizer(ticker_symbols)

weights_min_variance = portfolio_optimizer.minimum_variance_portfolio()
weights_max_sharpe = portfolio_optimizer.maximum_sharpe_ratio_portfolio()

print("Pesos para Mínima Varianza:", weights_min_variance)
print("Pesos para Máximo Ratio de Sharpe:", weights_max_sharpe)

# Ejemplo de uso de model_fit
x = np.random.rand(100, 5)  # Ejemplo de datos aleatorios para x
y = np.random.rand(100)  # Ejemplo de datos aleatorios para y
result_model_fit = portfolio_optimizer.model_fit(x, y)
print("Resultados de model_fit:", result_model_fit)
