"""
risk_engine.py
--------------
Institutional-grade risk metrics engine.
Implements VaR (Historical, Parametric, Monte Carlo), CVaR, Sharpe,
Sortino, Calmar, Beta, Max Drawdown, Rolling Volatility, and more.

Aligned with FRM Part 1 & 2 curriculum (GARP).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from typing import Union
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252
RISK_FREE    = 0.0525   # ~current Fed Funds rate approximation


# ── Value at Risk ─────────────────────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95,
                   horizon: int = 1) -> float:
    """
    Historical VaR — non-parametric, uses empirical return distribution.
    FRM: Market Risk · Basel III standard approach.
    """
    sorted_r = np.sort(returns.dropna())
    idx      = int(np.floor((1 - confidence) * len(sorted_r)))
    daily_var = abs(sorted_r[idx])
    return daily_var * np.sqrt(horizon)


def var_parametric(returns: pd.Series, confidence: float = 0.95,
                   horizon: int = 1) -> float:
    """
    Parametric (Variance-Covariance) VaR — assumes normal distribution.
    FRM: RiskMetrics methodology.
    """
    mu    = returns.mean()
    sigma = returns.std()
    z     = norm.ppf(1 - confidence)
    daily_var = abs(mu + z * sigma)
    return daily_var * np.sqrt(horizon)


def var_monte_carlo(returns: pd.Series, confidence: float = 0.95,
                    horizon: int = 1, simulations: int = 10000) -> float:
    """
    Monte Carlo VaR — simulates future return distribution.
    FRM: Simulation-based risk estimation.
    """
    mu      = returns.mean()
    sigma   = returns.std()
    np.random.seed(42)
    sim_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), simulations)
    return abs(np.percentile(sim_returns, (1 - confidence) * 100))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR) — average loss beyond VaR threshold.
    FRM: Coherent risk measure, preferred over VaR under Basel III.
    """
    sorted_r  = np.sort(returns.dropna())
    idx       = int(np.floor((1 - confidence) * len(sorted_r)))
    tail      = sorted_r[:idx]
    return abs(tail.mean()) if len(tail) > 0 else 0.0


def var_summary(returns: pd.Series, confidence: float = 0.95,
                horizon: int = 1, portfolio_value: float = 1_000_000) -> dict:
    """Return all three VaR methods + CVaR as a dictionary."""
    h_var  = var_historical(returns, confidence, horizon)
    p_var  = var_parametric(returns, confidence, horizon)
    mc_var = var_monte_carlo(returns, confidence, horizon)
    cvar   = cvar_historical(returns, confidence)

    return {
        "confidence"          : confidence,
        "horizon_days"        : horizon,
        "var_historical"      : round(h_var, 6),
        "var_parametric"      : round(p_var, 6),
        "var_monte_carlo"     : round(mc_var, 6),
        "cvar_historical"     : round(cvar, 6),
        "var_hist_dollar"     : round(h_var * portfolio_value, 2),
        "var_param_dollar"    : round(p_var * portfolio_value, 2),
        "var_mc_dollar"       : round(mc_var * portfolio_value, 2),
        "cvar_dollar"         : round(cvar * portfolio_value, 2),
    }


# ── Performance & Risk Ratios ─────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = RISK_FREE) -> float:
    """Annualized Sharpe Ratio. FRM: Performance attribution."""
    excess = returns - risk_free / TRADING_DAYS
    if returns.std() == 0:
        return 0.0
    return float((excess.mean() / returns.std()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = RISK_FREE) -> float:
    """
    Sortino Ratio — penalizes only downside volatility.
    FRM: Superior to Sharpe for non-normal distributions.
    """
    excess      = returns - risk_free / TRADING_DAYS
    downside    = returns[returns < 0]
    down_std    = downside.std() * np.sqrt(TRADING_DAYS)
    if down_std == 0:
        return 0.0
    return float(excess.mean() * TRADING_DAYS / down_std)


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar Ratio — annualized return / max drawdown."""
    ann_return = returns.mean() * TRADING_DAYS
    mdd        = max_drawdown(returns)
    return float(ann_return / abs(mdd)) if mdd != 0 else 0.0


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Information Ratio — active return / tracking error."""
    active      = returns - benchmark
    tracking_er = active.std() * np.sqrt(TRADING_DAYS)
    if tracking_er == 0:
        return 0.0
    return float(active.mean() * TRADING_DAYS / tracking_er)


# ── Drawdown Analysis ─────────────────────────────────────────────────────────

def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def drawdown_stats(returns: pd.Series) -> dict:
    """Max drawdown, duration, and recovery stats."""
    dd   = drawdown_series(returns)
    mdd  = dd.min()
    # Duration of worst drawdown
    trough_idx = dd.idxmin()
    peak_before = returns[:trough_idx].index[
        (1 + returns[:trough_idx]).cumprod().values ==
        (1 + returns[:trough_idx]).cumprod().max()
    ]
    duration = len(returns[peak_before[-1]:trough_idx]) if len(peak_before) > 0 else 0

    return {
        "max_drawdown"     : round(float(mdd), 6),
        "max_drawdown_pct" : round(float(mdd) * 100, 2),
        "trough_date"      : str(trough_idx.date()) if hasattr(trough_idx, 'date') else str(trough_idx),
        "drawdown_duration": duration,
    }


# ── Market Risk ───────────────────────────────────────────────────────────────

def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    CAPM Beta — systematic market risk.
    FRM: Equity risk factor exposure.
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    cov    = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    b      = cov[0, 1] / cov[1, 1]
    return round(float(b), 4)


def alpha(returns: pd.Series, benchmark: pd.Series,
          risk_free: float = RISK_FREE) -> float:
    """Jensen's Alpha — excess return above CAPM prediction."""
    b         = beta(returns, benchmark)
    ann_ret   = returns.mean() * TRADING_DAYS
    bench_ret = benchmark.mean() * TRADING_DAYS
    return round(ann_ret - (risk_free + b * (bench_ret - risk_free)), 6)


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_var(returns: pd.Series, window: int = 63,
                confidence: float = 0.95) -> pd.Series:
    """Rolling historical VaR."""
    return returns.rolling(window).apply(
        lambda x: var_historical(pd.Series(x), confidence), raw=False
    )


# ── Volatility Models ─────────────────────────────────────────────────────────

def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    EWMA (Exponentially Weighted Moving Average) Volatility.
    FRM: RiskMetrics volatility estimation model.
    λ = 0.94 is the RiskMetrics standard for daily data.
    """
    sq_returns = returns ** 2
    ewma_var   = sq_returns.ewm(alpha=1 - lambda_, adjust=False).mean()
    return np.sqrt(ewma_var) * np.sqrt(TRADING_DAYS)


def annualized_volatility(returns: pd.Series) -> float:
    """Simple annualized volatility."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


# ── Distribution Analysis ─────────────────────────────────────────────────────

def return_distribution(returns: pd.Series) -> dict:
    """Skewness, kurtosis, and normality test."""
    clean = returns.dropna()
    _, jb_pvalue = stats.jarque_bera(clean)
    _, sw_pvalue = stats.shapiro(clean[:500] if len(clean) > 500 else clean)

    return {
        "mean"           : round(float(clean.mean() * TRADING_DAYS), 6),
        "std_annual"     : round(float(clean.std() * np.sqrt(TRADING_DAYS)), 6),
        "skewness"       : round(float(stats.skew(clean)), 4),
        "excess_kurtosis": round(float(stats.kurtosis(clean)), 4),
        "jarque_bera_p"  : round(float(jb_pvalue), 6),
        "is_normal"      : jb_pvalue > 0.05,
        "min_return"     : round(float(clean.min()), 6),
        "max_return"     : round(float(clean.max()), 6),
    }


# ── Master Risk Report ────────────────────────────────────────────────────────

def full_risk_report(returns: pd.Series, benchmark: pd.Series = None,
                     confidence: float = 0.95,
                     portfolio_value: float = 1_000_000,
                     ticker: str = "Asset") -> dict:
    """Generate a complete risk report for a single asset."""

    report = {
        "ticker"          : ticker,
        "observations"    : len(returns),
        "start_date"      : str(returns.index[0].date()),
        "end_date"        : str(returns.index[-1].date()),
        "annualized_return": round(float(returns.mean() * TRADING_DAYS * 100), 2),
        "annualized_vol"  : round(annualized_volatility(returns) * 100, 2),
        "sharpe_ratio"    : round(sharpe_ratio(returns), 4),
        "sortino_ratio"   : round(sortino_ratio(returns), 4),
        "calmar_ratio"    : round(calmar_ratio(returns), 4),
        **var_summary(returns, confidence, 1, portfolio_value),
        **drawdown_stats(returns),
        **return_distribution(returns),
    }

    if benchmark is not None:
        report["beta"]  = beta(returns, benchmark)
        report["alpha"] = round(alpha(returns, benchmark) * 100, 4)
        report["information_ratio"] = round(information_ratio(returns, benchmark), 4)

    return report


if __name__ == "__main__":
    from data_fetcher import fetch_returns, fetch_benchmark, get_date_range

    start, end = get_date_range("2y")
    returns   = fetch_returns(["AAPL"], start, end)["AAPL"]
    benchmark = fetch_benchmark(start, end)

    report = full_risk_report(returns, benchmark, ticker="AAPL")
    print("\n── AAPL Full Risk Report ──")
    for k, v in report.items():
        print(f"  {k:30s}: {v}")
