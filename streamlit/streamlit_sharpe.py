
import streamlit as st



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

    st.markdown(f"{texto}", unsafe_allow_html=True)

texto = """
<div style="text-align: justify;">

# Estrategia de Inversión del Máximo de Sharpe

La estrategia de inversión del Máximo de Sharpe, también conocida como el ratio de Sharpe, es una técnica fundamental en la gestión de carteras de inversión. Desarrollada por William F. Sharpe en 1966, esta estrategia se utiliza para evaluar el rendimiento ajustado al riesgo de un portafolio y para optimizar la asignación de activos.

## Conceptos Clave

**1. Ratio de Sharpe:**
   - Es una medida que evalúa el rendimiento excedente de un portafolio por encima de la tasa de rendimiento libre de riesgo, ajustado por su volatilidad (desviación estándar de los rendimientos del portafolio). Se calcula como la diferencia entre el rendimiento del portafolio y el rendimiento libre de riesgo, dividido por la volatilidad del portafolio.

**2. Rendimiento Libre de Riesgo:**
   - Es el rendimiento que se espera de una inversión considerada "sin riesgo", como los bonos del tesoro de EE.UU.

**3. Volatilidad:**
   - Se refiere a la variabilidad de los rendimientos de un portafolio y es una medida común de riesgo.

## Funcionamiento de la Estrategia

El objetivo de la estrategia del Máximo de Sharpe es maximizar el ratio de Sharpe, es decir, obtener la máxima recompensa (rendimiento excedente) por cada unidad de riesgo asumido.

**1. Selección de Activos y Rendimientos Esperados:**
   - Se eligen activos para incluir en el portafolio y se estiman sus rendimientos esperados, así como la volatilidad y correlaciones entre ellos.

**2. Tasa de Rendimiento Libre de Riesgo:**
   - Se establece una tasa de rendimiento libre de riesgo para comparar con los rendimientos esperados del portafolio.

**3. Optimización del Portafolio:**
   - Utilizando técnicas de optimización, se calculan las ponderaciones de los activos en el portafolio que maximizan el ratio de Sharpe. Esto implica encontrar la combinación de activos que ofrece el mayor exceso de rendimiento por unidad de riesgo.

## Ejemplo Práctico

Imagina que estás considerando invertir en dos activos: Acciones y Bonos. Supón que las acciones tienen un rendimiento esperado del 10% con una volatilidad del 15%, y los bonos tienen un rendimiento esperado del 4% con una volatilidad del 5%. La tasa de rendimiento libre de riesgo es del 2%.

**Cálculo del Ratio de Sharpe para Cada Activo:**
   - **Acciones:** (10% - 2%) / 15% = 0.53
   - **Bonos:** (4% - 2%) / 5% = 0.40

Aquí, las acciones tienen un ratio de Sharpe más alto, lo que indica un mejor rendimiento ajustado por riesgo en comparación con los bonos.

**Construcción del Portafolio:**
   - La estrategia sería entonces encontrar la combinación óptima de acciones y bonos que maximice el ratio de Sharpe para el portafolio completo. Esto podría resultar, por ejemplo, en un 70% en acciones y un 30% en bonos.

## Ventajas y Desventajas

**Ventajas:**
   - **Rendimiento Ajustado al Riesgo:** Proporciona una comparación estandarizada del rendimiento de diferentes inversiones teniendo en cuenta su riesgo.
   - **Optimización del Portafolio:** Ayuda a los inversores a alcanzar la mejor combinación posible de activos para maximizar los rendimientos ajustados al riesgo.

**Desventajas:**
   - **Basado en Rendimientos Pasados:** El ratio de Sharpe se calcula utilizando datos históricos, lo que no siempre es un indicador confiable de rendimientos futuros.
   - **Suposición de Normalidad:** La efectividad del ratio de Sharpe se basa en la suposición de que los rendimientos de los activos se distribuyen normalmente, lo cual no siempre es el caso.

## Consideraciones Finales

La estrategia del Máximo de Sharpe es una herramienta valiosa para los inversores al buscar optimizar su portafolio para el mejor rendimiento ajustado al riesgo. Sin embargo, como con cualquier estrategia de inversión, es importante ser consciente de sus limitaciones y considerar una variedad de factores, incluyendo los objetivos de inversión, el horizonte de tiempo y la tolerancia al riesgo. Además, es crucial realizar un seguimiento y un reajuste periódico del portafolio para reflejar los cambios en el mercado y en las condiciones económicas.
</div>
"""

def show():
    st.title("Página de Ejemplo")
    st.write("¡Bienvenido a la página de ejemplo!")
    