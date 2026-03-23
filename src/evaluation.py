import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def backtest_portfolio(weights, returns_test, transaction_cost=0.001):
    """
    Backtest a portfolio on test data.

    Applies fixed weights to daily returns and subtracts a one-time
    transaction cost at entry.
    """
    tickers = returns_test.columns.tolist()
    weight_array = np.array([weights.get(t, 0.0) for t in tickers])

    portfolio_returns = (returns_test.values * weight_array).sum(axis=1)
    portfolio_returns = pd.Series(portfolio_returns, index=returns_test.index)

    # Apply one-time transaction cost on day 1
    portfolio_returns.iloc[0] -= transaction_cost

    return portfolio_returns


def compute_sharpe_ratio(portfolio_returns, risk_free_rate=0.02):
    """Compute annualized Sharpe ratio."""
    daily_rf = risk_free_rate / 252
    excess_returns = portfolio_returns - daily_rf
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe


def compute_max_drawdown(portfolio_returns):
    """Compute maximum drawdown and return the drawdown time series."""
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_dd = drawdown.min()
    return max_dd, drawdown


def compute_cumulative_returns(portfolio_returns):
    """Compute cumulative returns (starting at $1)."""
    return (1 + portfolio_returns).cumprod()


def compute_annualized_volatility(portfolio_returns):
    """Compute annualized volatility."""
    return portfolio_returns.std() * np.sqrt(252)


def compare_portfolios(results_dict):
    """
    Compare multiple portfolios and return a summary DataFrame.

    results_dict: {name: portfolio_returns_series}
    """
    rows = []
    for name, returns in results_dict.items():
        sharpe = compute_sharpe_ratio(returns)
        max_dd, _ = compute_max_drawdown(returns)
        cum_ret = compute_cumulative_returns(returns)
        total_return = cum_ret.iloc[-1] - 1
        ann_vol = compute_annualized_volatility(returns)

        rows.append({
            "Strategy": name,
            "Total Return": f"{total_return:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
        })

    summary = pd.DataFrame(rows).set_index("Strategy")
    return summary


def rolling_backtest(prices, returns, strategy_fn, window_years=2, step_months=6):
    """
    Perform rolling window backtesting.

    Trains on `window_years` of data, tests on the next `step_months`,
    then rolls forward. Returns a dict of {period_label: portfolio_returns}.
    """
    all_periods = {}
    dates = prices.index

    train_days = window_years * 252
    step_days = step_months * 21  # approx trading days per month

    start = 0
    while start + train_days + step_days <= len(dates):
        train_end_idx = start + train_days
        test_end_idx = min(train_end_idx + step_days, len(dates))

        prices_train = prices.iloc[start:train_end_idx]
        returns_test = returns.iloc[train_end_idx:test_end_idx]

        if len(returns_test) < 10:
            break

        weights = strategy_fn(prices_train)
        period_returns = backtest_portfolio(weights, returns_test)

        period_label = f"{returns_test.index[0].strftime('%Y-%m')} to {returns_test.index[-1].strftime('%Y-%m')}"
        all_periods[period_label] = {
            "returns": period_returns,
            "sharpe": compute_sharpe_ratio(period_returns),
            "max_dd": compute_max_drawdown(period_returns)[0],
            "total_return": compute_cumulative_returns(period_returns).iloc[-1] - 1,
        }

        start += step_days

    return all_periods


def plot_rolling_backtest(rolling_results_dict, save_path="data/rolling_backtest.png"):
    """Plot rolling backtest Sharpe ratios across periods for each strategy."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Sharpe per period
    for name, periods in rolling_results_dict.items():
        labels = list(periods.keys())
        sharpes = [periods[p]["sharpe"] for p in labels]
        axes[0].plot(range(len(labels)), sharpes, marker="o", label=name)

    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Rolling Backtest: Sharpe Ratio by Period")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)

    # Cumulative returns across all periods (stitched together)
    for name, periods in rolling_results_dict.items():
        all_returns = pd.concat([periods[p]["returns"] for p in periods])
        cum = compute_cumulative_returns(all_returns)
        axes[1].plot(cum.index, cum.values, label=name)

    axes[1].set_title("Rolling Backtest: Stitched Cumulative Returns")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Rolling backtest plot saved to {save_path}")


def plot_performance(results_dict, save_path="data/performance.png"):
    """Plot cumulative returns, drawdowns, and Sharpe comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # 1. Cumulative returns
    for name, returns in results_dict.items():
        cum = compute_cumulative_returns(returns)
        axes[0].plot(cum.index, cum.values, label=name)
    axes[0].set_title("Portfolio Cumulative Returns")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Drawdowns
    for name, returns in results_dict.items():
        _, dd = compute_max_drawdown(returns)
        axes[1].plot(dd.index, dd.values, label=name)
    axes[1].set_title("Portfolio Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Sharpe ratio bar chart
    names = list(results_dict.keys())
    sharpes = [compute_sharpe_ratio(results_dict[n]) for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    axes[2].bar(names, sharpes, color=colors)
    axes[2].set_title("Sharpe Ratio Comparison")
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Performance plot saved to {save_path}")


def plot_efficient_frontier(frontier_vols, frontier_rets, portfolio_points, save_path="data/efficient_frontier.png"):
    """
    Plot the efficient frontier with portfolio positions marked.

    portfolio_points: dict of {name: (volatility, return)}
    """
    plt.figure(figsize=(10, 7))
    plt.plot(frontier_vols, frontier_rets, "b-", linewidth=2, label="Efficient Frontier")

    colors = plt.cm.Set1(np.linspace(0, 1, len(portfolio_points)))
    for (name, (vol, ret)), color in zip(portfolio_points.items(), colors):
        plt.scatter(vol, ret, s=100, zorder=5, color=color, label=name)

    plt.title("Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Efficient frontier plot saved to {save_path}")


def plot_weights(weights_dict, save_path="data/weights.png"):
    """Plot portfolio weight allocations as horizontal bar charts."""
    n = len(weights_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, weights) in zip(axes, weights_dict.items()):
        tickers = list(weights.keys())
        vals = list(weights.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
        ax.barh(tickers, vals, color=colors)
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Weight")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Weights plot saved to {save_path}")
