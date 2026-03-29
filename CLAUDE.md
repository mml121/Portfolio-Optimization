# CLAUDE.md

## Portfolio Optimization using Machine Learning (Student Project Version)

---

## Project Overview

This project implements a **portfolio optimization system** using financial data and machine learning techniques. The goal is to construct an optimal portfolio that maximizes return while minimizing risk using both traditional finance methods and ML-based improvements.

This project is intentionally written in a **student-style implementation** (simple structure, readable code, minimal over-engineering), but still aims for **accurate results and proper methodology**.

---

## Objectives

* Collect historical stock price data
* Calculate returns and risk metrics
* Implement Markowitz Mean-Variance Optimization
* Improve risk estimation using ML techniques
* Compare performance with a baseline portfolio
* Evaluate using financial metrics like Sharpe Ratio

---

## Required Libraries

Install the following before starting:

```
pip install pandas numpy matplotlib yfinance scikit-learn PyPortfolioOpt
```

---

## Data Sources

Use the following sources:

* Stock price data:
  https://finance.yahoo.com

* Python API wrapper:
  https://pypi.org/project/yfinance/

* Portfolio optimization library:
  https://pyportfolioopt.readthedocs.io

---

## Project Structure

```
portfolio-optimization/
│
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── optimization.py
│   ├── evaluation.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Step 1: Data Collection

Use `yfinance` to download stock data.

Example stocks:

* AAPL
* MSFT
* GOOGL
* AMZN
* TSLA

Example code:

```python
import yfinance as yf

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
data = yf.download(stocks, start="2020-01-01", end="2024-01-01")["Adj Close"]
```

---

## Step 2: Feature Engineering

### Compute Daily Returns

```python
returns = data.pct_change().dropna()
```

### Compute Expected Returns

```python
mean_returns = returns.mean()
```

### Compute Covariance Matrix

```python
cov_matrix = returns.cov()
```

---

## Step 3: Portfolio Optimization (Markowitz)

Use PyPortfolioOpt:

```python
from pypfopt.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(mean_returns, cov_matrix)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
```

---

## Step 4: ML-Based Risk Improvement

Instead of using raw covariance, improve estimation.

### Option 1: Shrinkage (Recommended)

```python
from pypfopt import risk_models

cov_matrix = risk_models.CovarianceShrinkage(data).ledoit_wolf()
```

### Option 2: Volatility Prediction (ML)

Use Random Forest:

```python
from sklearn.ensemble import RandomForestRegressor
```

Train model on rolling window volatility.

---

## Step 5: Portfolio Performance

```python
ef.portfolio_performance(verbose=True)
```

Metrics:

* Expected Return
* Volatility
* Sharpe Ratio

---

## Step 6: Baseline Comparison

Compare against:

* Equal-weight portfolio
* Random weights

Example:

```python
import numpy as np

weights = np.ones(len(stocks)) / len(stocks)
```

---

## Step 7: Backtesting

Split data:

* Train: 2020–2022
* Test: 2023–2024

Apply weights on test data:

```python
portfolio_returns = (returns_test * weights).sum(axis=1)
```

---

## Step 8: Evaluation Metrics

### Sharpe Ratio

```python
sharpe = portfolio_returns.mean() / portfolio_returns.std()
```

### Maximum Drawdown

```python
cum_returns = (1 + portfolio_returns).cumprod()
drawdown = cum_returns / cum_returns.cummax() - 1
```

---

## Step 9: Visualization

Plot:

* Portfolio value over time
* Efficient frontier
* Drawdowns

```python
import matplotlib.pyplot as plt

cum_returns.plot()
plt.title("Portfolio Performance")
plt.show()
```

---

## Important Notes (Common Mistakes to Avoid)

* Do NOT randomly split time-series data
* Do NOT ignore transaction costs (even small ones matter)
* Do NOT assume higher returns = better model
* Always compare with a baseline
* Avoid overfitting (especially with ML models)

---

## Possible Extensions

* Add transaction costs
* Use more assets (Nifty 50 stocks for Indian context)
* Try LSTM for volatility prediction
* Build a simple dashboard using Streamlit
* Add rebalancing strategy

---

## What Makes This Project Strong

* Combines finance + ML
* Uses real-world data
* Includes evaluation beyond accuracy
* Demonstrates understanding of risk

---

## Expected Output

* Optimized portfolio weights
* Sharpe ratio comparison
* Performance plots
* Simple explanation of results

---

## Final Note

Keep the code clean but not overcomplicated. Focus on:

* Correct methodology
* Clear logic
* Honest evaluation

A simple implementation done correctly is better than a complex one done poorly.

---
