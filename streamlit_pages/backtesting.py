import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import dynamic_backtesting  
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
from streamlit_extras.badges import badge
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stoggle import stoggle

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def show_backtesting():
    st.title(':violet[Backtesting individual]')
    st.write("Selecciona los parámetros con el objetivo de realizar el backtesting para las estrategias QAA.")

    stoggle("Estrategias QAA usadas",
            """Mínima Varianza, Máximo Ratio de Sharpe, Semivarianza, Omega, Hierarchical Risk Parity (HRP), Conditional Value ar Risk (CVaR), Black Litterman, Famma French, Total Return AA, Roy Safety First Ratio, Sortino Ratio.""")
    
    add_vertical_space(5)

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        add_vertical_space(2)

        with col1:
            tickers = st.text_input("TICKERS", "A, B")
            start_date_data = st.text_input("FECHA INICIAL (aaaa-mm-dd)", "2020-01-02")
            end_date = st.text_input("FECHA FINAL (aaaa-mm-dd)", "2024-01-23")
            rf = st.number_input("TASA LIBRE DE RIESGO (%)", value=0.0000)
        
        with col2:
            start_backtesting = st.text_input("FECHA INICIAL DE REBALANCEO (aaaa-mm-dd)", "2023-01-23")
            rebalance_frequency_months = st.slider("REBALANCEO (meses)", 1, 12, 1)
            lower_bound= st.number_input("LÍMITE INFERIOR", value=0.01)
            higher_bound= st.number_input("LÍMITE SUPERIOR", value=0.99)
        
        with col3:
            optimization_strategy = st.selectbox("ESTRATEGIA QAA", 
                                                 ['Minimum Variance', 'Omega Ratio', 'Semivariance',
                                                  'Roy Safety First Ratio', 'Sortino Ratio', 'Fama French', 'CVaR', 
                                                  'HRP', 'Sharpe Ratio', 'Black Litterman', 'Total Return'])
            optimization_model = st.selectbox("MODELO DE OPTIMIZACIÓN", ['SLSQP', 'Monte Carlo', 'COBYLA'])            
            initial_portfolio_value = st.text_input("VALOR INICIAL DEL PORTAFOLIO ($)", value='0')
            commission = st.number_input("COMISIÓN (%)", value=0.0000, format="%.4f")
  
        submitted = st.form_submit_button(":violet[CALCULA BACKTESTING]", type="secondary", use_container_width=True)

    add_vertical_space(5)

    if submitted:
        if not (validate_date(start_date_data) and validate_date(start_backtesting) and validate_date(end_date)):
            st.error("Por favor, ingrese fechas válidas en el formato YYYY-MM-DD.")
            return

        initial_portfolio_value = float(initial_portfolio_value.replace(',', ''))
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]

        resultados_backtesting, daily_data, portfolio_values = dynamic_backtesting(
            tickers=tickers_list,
            start_date_data=start_date_data,
            start_backtesting=start_backtesting,
            end_date=end_date,
            lower_bound=lower_bound,
            higher_bound=higher_bound,
            rebalance_frequency_months=rebalance_frequency_months, 
            rf=rf, 
            optimization_strategy=optimization_strategy, 
            optimization_model=optimization_model,
            initial_portfolio_value=initial_portfolio_value,
            commission=commission
        )

        st.session_state['resultados_backtesting'] = resultados_backtesting
        st.session_state['daily_data'] = daily_data
        st.session_state['portfolio_values'] = portfolio_values

    if 'resultados_backtesting' in st.session_state:
    
        if 'daily_data' in st.session_state and 'portfolio_values' in st.session_state:
            rebalance_dates = st.session_state['resultados_backtesting']['end_date'].unique()
            st.subheader(f":violet[{optimization_strategy}]", divider="violet")
            st.dataframe(st.session_state['resultados_backtesting'])

            # Plot the portfolio value with optional rebalance lines
            plot_portfolio_value(st.session_state['daily_data'], st.session_state['portfolio_values'], rebalance_dates)

        # Always plot the last asset weights pie chart
        plot_asset_weights_pie_chart(st.session_state['resultados_backtesting'], tickers_list)

    add_vertical_space(5)

    badge(type="github", name="diegotita4/PAP")

def plot_asset_weights_pie_chart(resultados_backtesting, tickers):
    last_weights = resultados_backtesting.iloc[-1]
    weights = [last_weights[f'weight_{ticker}'] for ticker in tickers]
    labels = tickers
    
    fig = px.pie(values=weights, names=labels, title='DISTRIBUCIÓN DE PESOS DEL PORTAFOLIO', 
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_value(daily_data, portfolio_values, rebalance_dates=None, show_rebalance_lines=False):
    fig = px.line(daily_data, x=daily_data.index, y=portfolio_values, labels={'y': 'Valor del portafolio ($)', 'x': 'Fecha'},
                  title='VALOR DEL PORTAFOLIO EN EL TIEMPO')
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Valor del portafolio ($)', legend_title='Leyenda')
    fig.add_scatter(x=daily_data.index, y=portfolio_values, mode='lines', name='Valor del portafolio', line=dict(color='blue'))

    if show_rebalance_lines and rebalance_dates is not None:
        for date in rebalance_dates:
            fig.add_vline(x=date, line_width=2, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)