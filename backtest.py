import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
# pip install optuna
import optuna
import warnings
from matplotlib.lines import Line2D
import logging
import matplotlib.pyplot as plt
from functions import QAA


# Class aun no funciona perfecto
class BacktestingDinamico:
    def __init__(self, tickers, start_date, start_backtesting, end_date, frecuencias_rebalanceo_meses, rf, optimization_strategy, optimization_model, valor_portafolio_inicial):
        self.tickers = tickers
        self.start_date = start_date
        self.start_backtesting = start_backtesting
        self.end_date = end_date
        self.frecuencias_rebalanceo_meses = frecuencias_rebalanceo_meses
        self.rf = rf
        self.optimization_strategy = optimization_strategy
        self.optimization_model = optimization_model
        self.valor_portafolio_inicial = valor_portafolio_inicial
        self.resultados = []
        self.data = None
        self.benchmark_data = None

    def run_backtesting(self):
        fecha_inicio = pd.to_datetime(self.start_backtesting)
        fecha_fin = pd.to_datetime(self.end_date)
        valor_portafolio = self.valor_portafolio_inicial
        num_acciones_anteriores = pd.Series(0, index=self.tickers)

        # Proceso inicial en start_backtesting
        fecha_actual = fecha_inicio
        fecha_rebalanceo_fin = fecha_actual   
        if fecha_rebalanceo_fin > fecha_fin:
            fecha_rebalanceo_fin = fecha_fin
        
        estrategia = QAA(
            tickers=self.tickers,
            start_date=self.start_date,  # Datos históricos desde start_date
            end_date=fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
            rf=self.rf
        )
        estrategia.set_optimization_strategy(self.optimization_strategy)
        estrategia.set_optimization_model(self.optimization_model)
        estrategia.load_data()
        estrategia.optimize()
        
        precios_actuales = estrategia.data.iloc[-1]
        optimal_weights = estrategia.optimal_weights
        
        valor_inversion_por_ticker = valor_portafolio * optimal_weights
        num_acciones = (valor_inversion_por_ticker / precios_actuales).apply(np.floor)
        valor_invertido = num_acciones * precios_actuales
        cash_sobrante = valor_portafolio - valor_invertido.sum()
        
        num_acciones_anteriores = num_acciones.copy()
        valor_portafolio = valor_invertido.sum() + cash_sobrante
        
        fila_resultado = {
            'fecha_data_origen': self.start_date,
            'fecha_fin': fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
            **{f'peso_{ticker}': optimal_weights[i] for i, ticker in enumerate(self.tickers)},
            **{f'acciones_{ticker}': num_acciones[ticker] for ticker in self.tickers},
            **{f'valor_{ticker}': valor_invertido[ticker] for ticker in self.tickers},
            'cash_sobrante': cash_sobrante,
            'valor_total_cartera': valor_portafolio
        }
        
        self.resultados.append(fila_resultado)
        
        # Continuación del proceso de rebalanceo después del inicio
        for frecuencia_rebalanceo_meses in self.frecuencias_rebalanceo_meses:
            fecha_actual = fecha_rebalanceo_fin
            while fecha_actual <= fecha_fin:
                fecha_rebalanceo_fin = fecha_actual + relativedelta(months=frecuencia_rebalanceo_meses)
                if fecha_rebalanceo_fin > fecha_fin:
                    fecha_rebalanceo_fin = fecha_fin
                
                estrategia = QAA(
                    tickers=self.tickers, 
                    start_date=self.start_date, 
                    end_date=fecha_rebalanceo_fin.strftime('%Y-%m-%d'), 
                    rf=self.rf
                )
                estrategia.set_optimization_strategy(self.optimization_strategy)
                estrategia.set_optimization_model(self.optimization_model)
                estrategia.load_data()
                estrategia.optimize()
                
                
                precios_actuales = estrategia.data.iloc[-1]
                optimal_weights = estrategia.optimal_weights
                
                if not num_acciones_anteriores.equals(pd.Series(0, index=self.tickers)):
                    valor_portafolio = (num_acciones_anteriores * precios_actuales).sum()
                
                valor_inversion_por_ticker = valor_portafolio * optimal_weights
                num_acciones = (valor_inversion_por_ticker / precios_actuales).apply(np.floor)
                valor_invertido = num_acciones * precios_actuales
                cash_sobrante = valor_portafolio - valor_invertido.sum()
                
                diff_acciones = num_acciones - num_acciones_anteriores
                
                num_acciones_anteriores = num_acciones.copy()
                valor_portafolio = valor_invertido.sum() + cash_sobrante
                
                fila_resultado = {
                    'fecha_data_origen': self.start_date,
                    'fecha_fin': fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
                    **{f'peso_{ticker}': optimal_weights[i] for i, ticker in enumerate(self.tickers)},
                    **{f'acciones_{ticker}': num_acciones[ticker] for ticker in self.tickers},
                    **{f'diff_{ticker}': diff_acciones[ticker] for ticker in self.tickers},
                    **{f'valor_{ticker}': valor_invertido[ticker] for ticker in self.tickers},
                    'cash_sobrante': cash_sobrante,
                    'valor_total_cartera': valor_portafolio
                }
                
                self.resultados.append(fila_resultado)
                fecha_actual = fecha_rebalanceo_fin
                if fecha_rebalanceo_fin == fecha_fin:
                    break
        
        return pd.DataFrame(self.resultados)
    
# DEF que funciona mejor

def backtesting_dinamico(tickers, start_date, start_backtesting, end_date, frecuencias_rebalanceo_meses, rf, optimization_strategy, optimization_model, valor_portafolio_inicial):
    fecha_inicio = pd.to_datetime(start_backtesting)
    fecha_fin = pd.to_datetime(end_date)
    valor_portafolio = valor_portafolio_inicial
    num_acciones_anteriores = pd.Series(0, index=tickers)
    resultados = []
    
    # Proceso inicial en start_backtesting
    fecha_actual = fecha_inicio
    fecha_rebalanceo_fin = fecha_actual   
    if fecha_rebalanceo_fin > fecha_fin:
        fecha_rebalanceo_fin = fecha_fin
    
    estrategia = QAA(
        tickers=tickers,
        start_date=start_date,  # Datos históricos desde start_date
        end_date=fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
        rf=rf
    )
    estrategia.set_optimization_strategy(optimization_strategy)
    estrategia.set_optimization_model(optimization_model)
    estrategia.load_data()
    estrategia.optimize()
    
    precios_actuales = estrategia.data.iloc[-1]
    optimal_weights = estrategia.optimal_weights
    
    valor_inversion_por_ticker = valor_portafolio * optimal_weights
    num_acciones = (valor_inversion_por_ticker / precios_actuales).apply(np.floor)
    valor_invertido = num_acciones * precios_actuales
    cash_sobrante = valor_portafolio - valor_invertido.sum()
    
    num_acciones_anteriores = num_acciones.copy()
    valor_portafolio = valor_invertido.sum() + cash_sobrante
    
    fila_resultado = {
        'fecha_data_origen': start_date,
        'fecha_fin': fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
        **{f'peso_{ticker}': optimal_weights[i] for i, ticker in enumerate(tickers)},
        **{f'acciones_{ticker}': num_acciones[ticker] for ticker in tickers},
        **{f'valor_{ticker}': valor_invertido[ticker] for ticker in tickers},
        'cash_sobrante': cash_sobrante,
        'valor_total_cartera': valor_portafolio
    }
    
    resultados.append(fila_resultado)
    
    # Continuación del proceso de rebalanceo después del inicio
    for frecuencia_rebalanceo_meses in frecuencias_rebalanceo_meses:
        fecha_actual = fecha_rebalanceo_fin
        while fecha_actual <= fecha_fin:
            fecha_rebalanceo_fin = fecha_actual + relativedelta(months=frecuencia_rebalanceo_meses)
            if fecha_rebalanceo_fin > fecha_fin:
                fecha_rebalanceo_fin = fecha_fin
            
            estrategia = QAA(
                tickers=tickers, 
                start_date=start_date, 
                end_date=fecha_rebalanceo_fin.strftime('%Y-%m-%d'), 
                rf=rf
            )
            estrategia.set_optimization_strategy(optimization_strategy)
            estrategia.set_optimization_model(optimization_model)
            estrategia.load_data()
            estrategia.optimize()
            
            precios_actuales = estrategia.data.iloc[-1]
            optimal_weights = estrategia.optimal_weights
            
            if not num_acciones_anteriores.equals(pd.Series(0, index=tickers)):
                valor_portafolio = (num_acciones_anteriores * precios_actuales).sum()
            
            valor_inversion_por_ticker = valor_portafolio * optimal_weights
            num_acciones = (valor_inversion_por_ticker / precios_actuales).apply(np.floor)
            valor_invertido = num_acciones * precios_actuales
            cash_sobrante = valor_portafolio - valor_invertido.sum()
            
            diff_acciones = num_acciones - num_acciones_anteriores
            
            num_acciones_anteriores = num_acciones.copy()
            valor_portafolio = valor_invertido.sum() + cash_sobrante
            
            fila_resultado = {
                'fecha_data_origen': start_date,
                'fecha_fin': fecha_rebalanceo_fin.strftime('%Y-%m-%d'),
                **{f'peso_{ticker}': optimal_weights[i] for i, ticker in enumerate(tickers)},
                **{f'acciones_{ticker}': num_acciones[ticker] for ticker in tickers},
                **{f'diff_{ticker}': diff_acciones[ticker] for ticker in tickers},
                **{f'valor_{ticker}': valor_invertido[ticker] for ticker in tickers},
                'cash_sobrante': cash_sobrante,
                'valor_total_cartera': valor_portafolio
            }
            
            resultados.append(fila_resultado)
            fecha_actual = fecha_rebalanceo_fin
            if fecha_rebalanceo_fin == fecha_fin:
                break
    
    return pd.DataFrame(resultados)


def plot_portfolio_value(resultados_backtesting, tickers):
    # Prepare the dataframe with dynamic column names for each ticker
    df_columns = ['fecha_fin'] + [f'acciones_{ticker}' for ticker in tickers] + ['cash_sobrante']
    df = resultados_backtesting[df_columns].copy()
    df['fecha_fin'] = pd.to_datetime(df['fecha_fin'])
    df.set_index('fecha_fin', inplace=True)

    # Generate a date range for the daily data
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a new dataframe with a record for each day, forward-filling the rebalancing information
    daily_data = pd.DataFrame(index=date_range)
    daily_data = daily_data.join(df, how='left').ffill().reset_index().rename(columns={'index': 'fecha_fin'})

    # Fetch historical stock data using yfinance for the entire range of daily_data
    stock_data = yf.download(tickers, start=daily_data['fecha_fin'].min(), end=daily_data['fecha_fin'].max())
    prices = stock_data['Adj Close']

    # Update the index of daily_data to match the dates from the stock data
    daily_data.set_index('fecha_fin', inplace=True)
    daily_data = daily_data.reindex(prices.index).ffill()

    # Calculate the daily portfolio value
    portfolio_value = daily_data['cash_sobrante']
    for ticker in tickers:
        portfolio_value += daily_data[f'acciones_{ticker}'] * prices[ticker]

    # Plot the portfolio value over time
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', color='green')

    # Add vertical lines for rebalance dates
    for date in df.index:
        plt.axvline(x=date, color='red', linestyle='--')

    # Create custom legends
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='Portfolio Value'),
                       Line2D([0], [0], color='red', linestyle='--', label='Rebalance Date')]
    plt.legend(handles=legend_elements)

    plt.title('Daily Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value in $')
    plt.grid(True)
    plt.show()