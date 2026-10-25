# Asset Allocation Backtesting

Interactive Streamlit dashboard that builds and backtests 11 quantitative asset-allocation strategies with three optimization methods, all implemented from scratch in Python.

Professional Application Project (PAP I) · ITESO · Jan – May 2024 · Team of 5
📄 [Full report (ITESO institutional repository)](https://rei.iteso.mx/items/4d803520-dc49-4ade-93cb-4175e064d78d)

## What it does

- **Builds portfolios** for any set of tickers with 11 strategies: minimum variance, maximum Sharpe, semivariance, Omega, Hierarchical Risk Parity (HRP), CVaR, Black-Litterman, Fama-French, Total Return, Roy's Safety First and Sortino.
- **Optimizes weights** with three methods: SLSQP, Monte Carlo (with Optuna) and COBYLA, with weight bounds of 10%–99% per asset.
- **Backtests dynamically** with periodic rebalancing and a 0.25% commission per trade, compared against a benchmark (SPY by default).
- **Validates the implementation** against reference libraries (PyPortfolioOpt and empyrical) in `validations.ipynb`.

## Dashboard

Three pages, in Spanish:

1. **Cálculo de estrategias QAA:** optimal weights for each strategy.
2. **Backtesting individual:** one strategy and one optimizer over time.
3. **Backtesting general:** all strategies compared.

## Tech stack

Python · pandas · NumPy · SciPy · Optuna · Streamlit · Plotly · matplotlib · yfinance · pandas-datareader (Fama-French factors)

## How to run

```bash
git clone https://github.com/diegotita4/PAP-QAA_strategies.git
cd PAP-QAA_strategies
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
app.py               Streamlit entry point
functions.py         QAA class: strategies and optimizers
backtest.py          dynamic backtesting with rebalancing and commissions
streamlit_pages/     dashboard pages
validations.ipynb    validation against PyPortfolioOpt and empyrical
```

The report, documentation and presentation are included as PDF, DOCX and PPTX.

## Team

Diego Emilio Enriquez Nares ([@diegotita4](https://github.com/diegotita4)) · Juan Antonio Mugica Liparoli ([@Antonio-IF](https://github.com/Antonio-IF)) · José Alfonso Martínez Ramírez ([@JoAlfonso](https://github.com/JoAlfonso)) · Jesús Emmanuel Palomera Gaytán ([@J3SVS](https://github.com/J3SVS)) · Óscar Uriel Alvarado Garnica ([@Oscar148](https://github.com/Oscar148))

## License

GNU General Public License v3.0
