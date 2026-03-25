"""
Portfolio Optimization using Machine Learning
==============================================
Compares Markowitz optimization with different risk estimation methods:
- Raw covariance
- Ledoit-Wolf shrinkage
- ML-adjusted (Random Forest volatility prediction)
- Equal-weight baseline
- Random-weight baseline

Tested across multiple rolling backtest periods for robustness.
"""

import os
import numpy as np
import pandas as pd
from src.data_loader import download_data, split_data, save_data
from src.feature_engineering import (
    compute_returns,
    compute_expected_returns,
    compute_covariance_raw,
    compute_covariance_shrinkage,
    train_volatility_model,
    predict_covariance_ml,
)
from src.optimization import (
    optimize_max_sharpe,
    equal_weight_portfolio,
    random_weight_portfolio,
    compute_efficient_frontier,
)
from src.evaluation import (
    backtest_portfolio,
    compare_portfolios,
    plot_performance,
    plot_efficient_frontier,
    plot_weights,
    rolling_backtest,
    plot_rolling_backtest,
)


# ── Configuration ──────────────────────────────────────────────

# Diversified across sectors: Tech, Healthcare, Finance, Consumer, Energy, Industrials
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Finance
    "JPM", "V", "BAC",
    # Consumer
    "AMZN", "PG", "KO",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "HON",
    # Tesla (high-volatility)
    "TSLA",
]

START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
TRAIN_END = "2022-12-31"
TRANSACTION_COST = 0.001  # 10 basis points


def print_analysis(summary, all_weights):
    """Print analysis of results explaining why certain strategies performed as they did."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Why Do Results Look This Way?")
    print("=" * 60)

    # Count how many assets each Markowitz strategy actually uses
    for name, weights in all_weights.items():
        if "Markowitz" in name:
            active = {k: v for k, v in weights.items() if v > 0.01}
            n_active = len(active)
            top = max(active, key=active.get)
            print(f"\n{name}:")
            print(f"  Active positions: {n_active} out of {len(weights)} assets")
            print(f"  Largest position: {top} ({active[top]:.1%})")
            if n_active <= 3:
                print(f"  -> CONCENTRATED: only {n_active} stocks. High risk from single-stock moves.")

    print("\n--- Key Takeaways ---")
    print("""
1. CONCENTRATION RISK: Markowitz optimization tends to concentrate heavily in
   a few assets that had the best historical risk-adjusted returns. With only
   training data from 2019-2022, the optimizer doesn't know which stocks will
   perform best in 2023. A concentrated bet can pay off (high total return)
   but comes with higher volatility and drawdowns.

2. DIVERSIFICATION BENEFIT: Equal-weight spreads risk across all 18 stocks
   and multiple sectors. This reduces the impact of any single stock crashing,
   leading to lower drawdowns and often better Sharpe ratios — especially when
   the concentrated Markowitz bets don't perfectly predict the future.

3. ML IMPROVEMENT: The ML-adjusted strategy uses Random Forest to predict
   forward volatility, which helps reduce allocation to stocks the model
   expects to be more volatile. This typically improves Sharpe ratio vs. raw
   Markowitz by producing more moderate, diversified allocations.

4. SHRINKAGE helps when assets are correlated (like tech stocks) by pulling
   extreme covariance estimates toward a structured target. The improvement
   over raw covariance is most visible with a larger asset universe.

5. HONEST ASSESSMENT: In many real-world scenarios, simple equal-weight
   portfolios are hard to beat after transaction costs. This is a well-known
   result in finance (the "1/N puzzle"). The value of optimization shows
   more clearly over longer periods and with proper rebalancing.
""")


def main():
    os.makedirs("data", exist_ok=True)

    # ── Step 1: Data Collection ────────────────────────────────
    print("=" * 60)
    print("STEP 1: Downloading Data")
    print("=" * 60)
    prices = download_data(TICKERS, START_DATE, END_DATE)
    save_data(prices, "data/prices.csv")
    print(f"\nAsset universe: {len(TICKERS)} stocks across 6 sectors")

    # ── Step 2: Train/Test Split ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Data")
    print("=" * 60)
    prices_train, prices_test = split_data(prices, TRAIN_END)

    returns_train = compute_returns(prices_train)
    returns_test = compute_returns(prices_test)

    # ── Step 3: Feature Engineering ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    print("\nComputing expected returns (from training data)...")
    mu = compute_expected_returns(prices_train)
    print(mu.sort_values(ascending=False).to_string())

    print("\nComputing raw covariance matrix...")
    cov_raw = compute_covariance_raw(returns_train)

    print("\nComputing Ledoit-Wolf shrinkage covariance...")
    cov_shrink = compute_covariance_shrinkage(prices_train)

    print("\nTraining ML volatility models...")
    vol_models = train_volatility_model(returns_train)

    print("\nPredicting ML-adjusted covariance...")
    cov_ml = predict_covariance_ml(vol_models, returns_train, prices_train)

    # ── Step 4: Portfolio Optimization ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Portfolio Optimization")
    print("=" * 60)

    print("\n--- Max Sharpe (Raw Covariance) ---")
    w_raw, perf_raw = optimize_max_sharpe(mu, cov_raw)
    print({k: v for k, v in w_raw.items() if v > 0})

    print("\n--- Max Sharpe (Shrinkage Covariance) ---")
    w_shrink, perf_shrink = optimize_max_sharpe(mu, cov_shrink)
    print({k: v for k, v in w_shrink.items() if v > 0})

    print("\n--- Max Sharpe (ML-Adjusted Covariance) ---")
    w_ml, perf_ml = optimize_max_sharpe(mu, cov_ml)
    print({k: v for k, v in w_ml.items() if v > 0})

    print("\n--- Equal Weight ---")
    w_equal = equal_weight_portfolio(TICKERS)
    print(f"  {len(TICKERS)} stocks, each at {1/len(TICKERS):.1%}")

    print("\n--- Random Weight ---")
    w_random = random_weight_portfolio(TICKERS)
    top3 = sorted(w_random.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3: {', '.join(f'{t} ({w:.1%})' for t, w in top3)}")

    # ── Step 5: Single-Period Backtesting ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Backtesting on Test Data (2023)")
    print("=" * 60)

    all_weights = {
        "Markowitz (Raw)": w_raw,
        "Markowitz (Shrinkage)": w_shrink,
        "Markowitz (ML)": w_ml,
        "Equal Weight": w_equal,
        "Random Weight": w_random,
    }

    results = {}
    for name, weights in all_weights.items():
        ret = backtest_portfolio(weights, returns_test, TRANSACTION_COST)
        results[name] = ret

    # ── Step 6: Evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Single-Period Performance Comparison")
    print("=" * 60)

    summary = compare_portfolios(results)
    print("\n" + summary.to_string())

    sharpes = {name: float(row["Sharpe Ratio"]) for name, row in summary.iterrows()}
    best = max(sharpes, key=sharpes.get)
    print(f"\nBest Sharpe Ratio: {best} ({sharpes[best]:.2f})")

    # ── Step 7: Rolling Backtest ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Rolling Backtest (2-year train, 6-month test windows)")
    print("=" * 60)

    all_returns = compute_returns(prices)

    def make_strategy(strategy_name):
        """Create a strategy function for rolling backtest."""
        def strategy_fn(prices_window):
            mu_w = compute_expected_returns(prices_window)
            if strategy_name == "raw":
                cov_w = compute_covariance_raw(compute_returns(prices_window))
                w, _ = optimize_max_sharpe(mu_w, cov_w)
            elif strategy_name == "shrinkage":
                cov_w = compute_covariance_shrinkage(prices_window)
                w, _ = optimize_max_sharpe(mu_w, cov_w)
            elif strategy_name == "ml":
                returns_w = compute_returns(prices_window)
                models = train_volatility_model(returns_w)
                cov_w = predict_covariance_ml(models, returns_w, prices_window)
                w, _ = optimize_max_sharpe(mu_w, cov_w)
            elif strategy_name == "equal":
                w = equal_weight_portfolio(prices_window.columns.tolist())
            return w
        return strategy_fn

    rolling_results = {}
    for name, key in [
        ("Markowitz (Shrinkage)", "shrinkage"),
        ("Markowitz (ML)", "ml"),
        ("Equal Weight", "equal"),
    ]:
        print(f"\nRolling backtest: {name}")
        rolling_results[name] = rolling_backtest(
            prices, all_returns, make_strategy(key),
            window_years=2, step_months=6
        )

    # Print rolling backtest summary
    print("\n--- Rolling Backtest Summary ---")
    for name, periods in rolling_results.items():
        avg_sharpe = np.mean([p["sharpe"] for p in periods.values()])
        avg_dd = np.mean([p["max_dd"] for p in periods.values()])
        n_positive = sum(1 for p in periods.values() if p["sharpe"] > 0)
        print(f"{name}:")
        print(f"  Avg Sharpe: {avg_sharpe:.2f} | Avg Max DD: {avg_dd:.2%} | Positive periods: {n_positive}/{len(periods)}")

    # ── Step 8: Analysis ───────────────────────────────────────
    print_analysis(summary, all_weights)

    # ── Step 9: Visualization ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Generating Plots")
    print("=" * 60)

    # Performance plot
    plot_performance(results)

    # Efficient frontier (using shrinkage covariance)
    print("\nComputing efficient frontier...")
    f_vols, f_rets = compute_efficient_frontier(mu, cov_shrink)

    portfolio_points = {}
    for name, perf in [
        ("Markowitz (Raw)", perf_raw),
        ("Markowitz (Shrinkage)", perf_shrink),
        ("Markowitz (ML)", perf_ml),
    ]:
        portfolio_points[name] = (perf[1], perf[0])

    for name in ["Equal Weight", "Random Weight"]:
        ret_series = results[name]
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * np.sqrt(252)
        portfolio_points[name] = (ann_vol, ann_ret)

    plot_efficient_frontier(f_vols, f_rets, portfolio_points)

    # Weights comparison (only show non-zero for readability)
    plot_weights(all_weights)

    # Rolling backtest plot
    plot_rolling_backtest(rolling_results)

    print("\n" + "=" * 60)
    print("DONE! Check the data/ folder for saved plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
