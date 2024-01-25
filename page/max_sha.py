

import streamlit as st
             
def show():
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