## Librerias
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
## Explicacion codigo
# El código proporcionado define una clase `PortfolioOptimizer` en Python diseñada para optimizar carteras de inversión utilizando datos históricos de precios de acciones. La clase utiliza las bibliotecas `numpy` para cálculos numéricos, `yfinance` para descargar datos financieros, y `scipy` para optimización. A continuación, se detalla la funcionalidad de cada parte del código:

### Clase `PortfolioOptimizer`

#- **`__init__` (Método inicializador):** Este método se llama automáticamente al crear una instancia de `PortfolioOptimizer`. Inicializa la instancia con los tickers (símbolos de las acciones) proporcionados, un tipo de interés libre de riesgo, y obtiene los datos de retorno y la matriz de covarianza de los activos mediante el método `get_portfolio_data`.

#- **`get_portfolio_data` (Obtener datos del portafolio):** Descarga los precios ajustados al cierre de los tickers especificados entre las fechas de inicio y fin. Calcula los retornos diarios y la matriz de covarianza de estos retornos. Devuelve los retornos, la matriz de covarianza y el número de activos.

#- **`_calculate_portfolio_metrics` (Calcular métricas del portafolio):** Calcula y devuelve el retorno esperado y la volatilidad (riesgo) del portafolio basado en los pesos actuales de los activos, donde los retornos se anualizan multiplicándolos por 252 (días hábiles en un año).

### Métodos de Optimización

#- **`minimum_variance_portfolio` (Portafolio de mínima varianza):** Utiliza la función `minimize` de SciPy para encontrar los pesos de los activos que minimizan la volatilidad del portafolio. Se definen límites para asegurar que cada peso esté entre 0.10 y 1, y una restricción para que la suma de los pesos sea igual a 1.

#- **`maximum_sharpe_ratio_portfolio` (Portafolio de máximo ratio de Sharpe):** Encuentra los pesos que maximizan el ratio de Sharpe del portafolio (retorno del portafolio sobre su volatilidad, ajustado por el tipo libre de riesgo), también asegurando que cada peso esté entre 0.10 y 1 y que la suma de pesos sea 1.

### Métodos Auxiliares

#- **`_variance_objective` (Objetivo de varianza):** Una función objetivo usada por el optimizador para calcular la volatilidad del portafolio, basada en la matriz de covarianza `sigma` y los pesos actuales.

#- **`_sharpe_objective` (Objetivo del ratio de Sharpe):** Calcula el negativo del ratio de Sharpe para ser utilizado en la optimización. Se busca maximizar el ratio de Sharpe, lo que se logra minimizando su negativo.

### Ejemplo de Uso

#El código final crea una instancia de `PortfolioOptimizer` con un conjunto de tickers y un tipo de interés libre de riesgo. Luego, calcula y muestra los pesos de los activos para el portafolio de mínima varianza y el portafolio de máximo ratio de Sharpe, aplicando las restricciones y límites mencionados.

### Importancia de Cada Parte

#- **Optimización de Portafolios:** Este proceso es fundamental en la gestión financiera y la inversión. Ayuda a los inversores a determinar la mejor distribución de sus activos para maximizar el retorno ajustado al riesgo.
  
#- **Ratio de Sharpe:** Es una medida del rendimiento ajustado al riesgo. Un ratio más alto indica un mejor rendimiento ajustado al riesgo.

#- **Mínima Varianza:** Buscar el portafolio de mínima varianza ayuda a los inversores a minimizar el riesgo.

# Este enfoque cuantitativo para la selección de portafolios permite a los inversores tomar decisiones basadas en datos y estadísticas, en lugar de intuición o especulación.

class PortfolioOptimizer:
    def __init__(self, tickers, risk_free_rate=0.0):
        self.returns, self.cov_matrix, self.num_assets = self.get_portfolio_data(tickers)
        self.rf = risk_free_rate

    def get_portfolio_data(self, tickers):
        data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov().to_numpy()
        num_assets = len(tickers)
        return returns, cov_matrix, num_assets

    def _calculate_portfolio_metrics(self, weights):
        port_return = np.dot(weights, self.returns.mean()) * 252  # Annualize returns
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)  # Annualize volatility
        return port_return, port_volatility

    def minimum_variance_portfolio(self):
        initial_weights = np.ones(self.num_assets) / self.num_assets
        # Adjusting bounds to ensure each weight is >= 0.10
        bounds = tuple((0.10, 1) for _ in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        result = minimize(lambda x: self._calculate_portfolio_metrics(x)[1], initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def maximum_sharpe_ratio_portfolio(self):
        initial_weights = np.ones(self.num_assets) / self.num_assets
        # Adjusting bounds to ensure each weight is >= 0.10
        bounds = tuple((0.10, 1) for _ in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        result = minimize(lambda x: -self._sharpe_objective(x), initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def _variance_objective(self, weights, sigma):
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return port_volatility

    def _sharpe_objective(self, weights):
        port_return, port_volatility = self._calculate_portfolio_metrics(weights)
        sharpe_ratio = (port_return - self.rf) / port_volatility
        return -sharpe_ratio

# Ejemplo de uso:
portfolio_optimizer = PortfolioOptimizer(['AAPL', 'AMZN', 'GOOGL', 'MSFT'], 0.02)

weights_min_variance = portfolio_optimizer.minimum_variance_portfolio()
weights_max_sharpe = portfolio_optimizer.maximum_sharpe_ratio_portfolio()

print("Pesos del Portafolio de Mínima Varianza:", weights_min_variance)
print("Pesos del Portafolio de Máximo Ratio de Sharpe:", weights_max_sharpe)

### Proceso pero con diferentes metodos de optimizacion

import numpy as np
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, tickers, risk_free_rate=0.0):
        self.returns, self.cov_matrix, self.num_assets = self.get_portfolio_data(tickers)
        self.rf = risk_free_rate

    def get_portfolio_data(self, tickers):
        data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov().to_numpy()
        num_assets = len(tickers)
        return returns, cov_matrix, num_assets

    def _calculate_portfolio_metrics(self, weights):
        port_return = np.dot(weights, self.returns.mean()) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        return port_return, port_volatility

    def _sharpe_objective(self, weights):
        port_return, port_volatility = self._calculate_portfolio_metrics(weights)
        sharpe_ratio = (port_return - self.rf) / port_volatility
        return -sharpe_ratio

    def optimize_portfolio(self, objective_function, method):
        initial_weights = np.ones(self.num_assets) / self.num_assets
        bounds = tuple((0.10, 1) for _ in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        result = minimize(objective_function, initial_weights, method=method,
                          bounds=bounds, constraints=constraints)
        return result.x

    def minimum_variance_portfolio(self, method='SLSQP'):
        return self.optimize_portfolio(lambda x: self._calculate_portfolio_metrics(x)[1], method)

    def maximum_sharpe_ratio_portfolio(self, method='SLSQP'):
        return self.optimize_portfolio(lambda x: -self._sharpe_objective(x), method)

# Ejemplo de uso:
portfolio_optimizer = PortfolioOptimizer(['AAPL', 'AMZN', 'GOOGL', 'MSFT'], 0.02)

methods = ['SLSQP', 'TNC', 'L-BFGS-B']
for method in methods:
    weights_min_variance = portfolio_optimizer.minimum_variance_portfolio(method)
    weights_max_sharpe = portfolio_optimizer.maximum_sharpe_ratio_portfolio(method)
    print(f"Metodo: {method}")
    print("Pesos del Portafolio de Mínima Varianza:", weights_min_variance)
    print("Pesos del Portafolio de Máximo Ratio de Sharpe:", weights_max_sharpe)
    print("---")


###  Codigo con las 3 funciones de optimizacion quitando las restricciones para TNC Y L-BFGS-B
    
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, tickers, risk_free_rate=0.0):
        self.returns, self.cov_matrix, self.num_assets = self.get_portfolio_data(tickers)
        self.rf = risk_free_rate

    def get_portfolio_data(self, tickers):
        data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov().to_numpy()
        num_assets = len(tickers)
        return returns, cov_matrix, num_assets

    def _calculate_portfolio_metrics(self, weights):
        port_return = np.dot(weights, self.returns.mean()) * 252  # Annualize returns
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)  # Annualize volatility
        return port_return, port_volatility

    def _sharpe_objective(self, weights):
        port_return, port_volatility = self._calculate_portfolio_metrics(weights)
        sharpe_ratio = (port_return - self.rf) / port_volatility
        return -sharpe_ratio

    def optimize_portfolio(self, objective_function, method):
        initial_weights = np.ones(self.num_assets) / self.num_assets
        bounds = tuple((0.10, 1) for _ in range(self.num_assets))
        
        # Aplicar restricciones solo para SLSQP
        if method == 'SLSQP':
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            result = minimize(objective_function, initial_weights, method=method, bounds=bounds, constraints=constraints)
        else:
            # Para TNC y L-BFGS-B, omitir restricciones de igualdad
            result = minimize(objective_function, initial_weights, method=method, bounds=bounds)
            
        return result.x

    def minimum_variance_portfolio(self, method='SLSQP'):
        return self.optimize_portfolio(lambda x: self._calculate_portfolio_metrics(x)[1], method)

    def maximum_sharpe_ratio_portfolio(self, method='SLSQP'):
        return self.optimize_portfolio(lambda x: -self._sharpe_objective(x), method)

# Ejemplo de uso:
portfolio_optimizer = PortfolioOptimizer(['AAPL', 'AMZN', 'GOOGL', 'MSFT'], 0.02)

methods = ['SLSQP', 'TNC', 'L-BFGS-B']
for method in methods:
    weights_min_variance = portfolio_optimizer.minimum_variance_portfolio(method)
    weights_max_sharpe = portfolio_optimizer.maximum_sharpe_ratio_portfolio(method)
    print(f"Método: {method}")
    print("Pesos del Portafolio de Mínima Varianza:", weights_min_variance)
    print("Pesos del Portafolio de Máximo Ratio de Sharpe:", weights_max_sharpe)
    print("---")



# PONCHO INTENTO IMPLEMENTARLO DE MANERA DIINAMICA 1
    
def optimize_sharpe_ratio(tickers, start_date, end_date, risk_free_rate=0.0):
    # Cargar datos
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Funciones auxiliares
    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        return -sharpe_ratio  # Negativo porque minimizamos

    # Restricciones y límites
    bounds = tuple((0.1, 1) for _ in tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)

    # Optimización
    initial_guess = np.ones(len(tickers)) / len(tickers)
    result = minimize(portfolio_performance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError('No se pudo encontrar una solución óptima.')
    

def show():
    st.title("Optimizador de Portafolio - Ratio de Sharpe")

    # Entrada de usuario para tickers
    user_input_tickers = st.text_input("Ingrese los tickers separados por coma", "AAPL, MSFT, GOOGL, AMZN")
    tickers = [ticker.strip() for ticker in user_input_tickers.split(',')]

    # Entrada de usuario para fechas
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2021-01-01"))

    # Entrada de usuario para la tasa libre de riesgo
    risk_free_rate = st.number_input("Tasa Libre de Riesgo", value=0.0, format="%.2f")

    if st.button("Optimizar"):
        weights = optimize_sharpe_ratio(tickers, start_date, end_date, risk_free_rate)
        weights_rounded = np.round(weights * 100, 2)
        
        # Mostrar los pesos óptimos
        st.subheader("Pesos Óptimos del Portafolio")
        for ticker, weight in zip(tickers, weights_rounded):
            st.write(f"{ticker}: {weight}%")
        
        # Cargar datos nuevamente para el período seleccionado
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        returns = data.pct_change().dropna()
        
        # Calcular el rendimiento diario del portafolio
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calcular el rendimiento acumulado
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Graficar el rendimiento acumulado
        plt.figure(figsize=(10, 6))
        cumulative_returns.plot()
        plt.title('Rendimiento Acumulado del Portafolio')
        plt.xlabel('Fecha')
        plt.ylabel('Rendimiento Acumulado')
        plt.grid(True)
        st.pyplot(plt)
