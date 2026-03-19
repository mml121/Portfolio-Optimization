import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from sklearn.ensemble import RandomForestRegressor


def compute_returns(prices):
    """Compute daily percentage returns."""
    returns = prices.pct_change().dropna()
    return returns


def compute_expected_returns(prices):
    """Compute annualized mean historical returns using PyPortfolioOpt."""
    mu = expected_returns.mean_historical_return(prices)
    return mu


def compute_covariance_raw(returns):
    """Compute annualized covariance matrix from daily returns."""
    cov_matrix = returns.cov() * 252
    return cov_matrix


def compute_covariance_shrinkage(prices):
    """Compute Ledoit-Wolf shrinkage covariance matrix."""
    cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    return cov_matrix


def compute_rolling_volatility(returns, window=21):
    """Compute rolling standard deviation (volatility)."""
    return returns.rolling(window=window).std()


def train_volatility_model(returns_train):
    """
    Train a Random Forest model per stock to predict next-period volatility.

    Features: rolling 5-day return, 21-day volatility, 63-day volatility
    Target: next 21-day realized volatility
    """
    models = {}

    for ticker in returns_train.columns:
        series = returns_train[ticker]

        # Build features
        feat = pd.DataFrame(index=series.index)
        feat["ret_5d"] = series.rolling(5).mean()
        feat["vol_21d"] = series.rolling(21).std()
        feat["vol_63d"] = series.rolling(63).std()

        # Target: forward 21-day realized volatility
        feat["target"] = series.rolling(21).std().shift(-21)

        # Drop NaN rows
        feat = feat.dropna()

        if len(feat) < 50:
            print(f"Warning: not enough data to train model for {ticker}, skipping.")
            continue

        X = feat[["ret_5d", "vol_21d", "vol_63d"]].values
        y = feat["target"].values

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        models[ticker] = model
        print(f"  Trained volatility model for {ticker} (R² = {model.score(X, y):.3f})")

    return models


def predict_covariance_ml(models, returns, prices):
    """
    Adjust the shrinkage covariance matrix using RF-predicted volatilities.

    Scale each asset's row/column by (predicted_vol / historical_vol).
    """
    # Get shrinkage covariance as base
    cov_shrink = compute_covariance_shrinkage(prices)

    tickers = returns.columns.tolist()
    adjustments = {}

    for ticker in tickers:
        if ticker not in models:
            adjustments[ticker] = 1.0
            continue

        series = returns[ticker]

        # Build the same features as training
        feat = pd.DataFrame(index=series.index)
        feat["ret_5d"] = series.rolling(5).mean()
        feat["vol_21d"] = series.rolling(21).std()
        feat["vol_63d"] = series.rolling(63).std()
        feat = feat.dropna()

        if len(feat) == 0:
            adjustments[ticker] = 1.0
            continue

        # Use the most recent observation to predict forward volatility
        X_last = feat.iloc[[-1]][["ret_5d", "vol_21d", "vol_63d"]].values
        predicted_vol = models[ticker].predict(X_last)[0]
        predicted_vol = max(predicted_vol, 1e-6)  # clip to positive

        historical_vol = series.std()
        if historical_vol > 0:
            adjustments[ticker] = predicted_vol / historical_vol
        else:
            adjustments[ticker] = 1.0

    # Build adjustment matrix and apply to covariance
    adj_array = np.array([adjustments[t] for t in tickers])
    adj_matrix = np.outer(adj_array, adj_array)

    cov_ml = pd.DataFrame(
        cov_shrink.values * adj_matrix,
        index=cov_shrink.index,
        columns=cov_shrink.columns
    )

    print("  ML-adjusted covariance matrix computed.")
    return cov_ml
