import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def _init_(self, returns, cov_matrix):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.num_assets = len(returns)
        
    def _constraints(self, weights):
        # Restricciones para que la suma de los pesos sea 1 y cada activo sea >= 10%
        return [np.sum(weights) - 1, *weights - 0.1]
    
    def _calculate_portfolio_metrics(self, weights):
        # Función auxiliar para calcular métricas del portafolio
        port_return = np.dot(weights, self.returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return port_return, port_volatility
    
    def minimum_variance_portfolio(self):
        # Función para calcular la Mínima Varianza
        initial_weights = np.ones(self.num_assets) / self.num_assets
        result = minimize(self._calculate_portfolio_metrics, initial_weights,
                          method='SLSQP', constraints={'type': 'eq', 'fun': self._constraints})
        return result.x
    
    def maximum_sharpe_ratio_portfolio(self):
        # Función para calcular el Máximo Ratio de Sharpe
        initial_weights = np.ones(self.num_assets) / self.num_assets
        result = minimize(lambda weights: -self._calculate_portfolio_metrics(weights)[0] /
                          self._calculate_portfolio_metrics(weights)[1],
                          initial_weights, method='SLSQP', constraints={'type': 'eq', 'fun': self._constraints})
        return result.x

# Ejemplo de uso
returns = np.array([0.05, 0.08, 0.12])  # Rendimientos esperados de los activos
cov_matrix = np.array([[0.1, 0.03, 0.05],
                      [0.03, 0.12, 0.07],
                      [0.05, 0.07, 0.15]])  # Matriz de covarianza

# Crear instancia de la clase PortfolioOptimizer
portfolio_optimizer = PortfolioOptimizer(returns, cov_matrix)

# Calcular la Mínima Varianza y el Máximo Ratio de Sharpe
weights_min_variance = portfolio_optimizer.minimum_variance_portfolio()
weights_max_sharpe = portfolio_optimizer.maximum_sharpe_ratio_portfolio()

print("Pesos para Mínima Varianza:", weights_min_variance)
print("Pesos para Máximo Ratio de Sharpe:", weights_max_sharpe)