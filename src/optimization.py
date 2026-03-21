import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier


def optimize_max_sharpe(expected_returns, cov_matrix):
    """Find the portfolio that maximizes the Sharpe ratio."""
    ef = EfficientFrontier(expected_returns, cov_matrix)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()

    perf = ef.portfolio_performance(verbose=False)
    print(f"  Expected Return: {perf[0]:.2%}")
    print(f"  Volatility:      {perf[1]:.2%}")
    print(f"  Sharpe Ratio:    {perf[2]:.2f}")

    return cleaned, perf


def equal_weight_portfolio(tickers):
    """Create an equal-weight portfolio."""
    n = len(tickers)
    weights = {t: 1.0 / n for t in tickers}
    return weights


def random_weight_portfolio(tickers, seed=42):
    """Create a random-weight portfolio using Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(len(tickers)))
    weights = {t: w for t, w in zip(tickers, raw)}
    return weights


def compute_efficient_frontier(expected_returns, cov_matrix, n_points=50):
    """Generate points along the efficient frontier."""
    vols = []
    rets = []

    # Find the range of feasible returns
    ef_min = EfficientFrontier(expected_returns, cov_matrix)
    ef_min.min_volatility()
    min_ret = ef_min.portfolio_performance(verbose=False)[0]

    max_ret = expected_returns.max()

    target_returns = np.linspace(min_ret, max_ret, n_points)

    for target in target_returns:
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            ef.efficient_return(target)
            perf = ef.portfolio_performance(verbose=False)
            vols.append(perf[1])
            rets.append(perf[0])
        except Exception:
            continue

    return np.array(vols), np.array(rets)
