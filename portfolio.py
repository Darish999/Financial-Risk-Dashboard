"""
portfolio.py
------------
Multi-asset portfolio construction, optimization, and risk aggregation.
Implements correlation matrix, portfolio VaR, efficient frontier simulation,
and weight optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk_engine import (
    var_historical, cvar_historical, sharpe_ratio,
    max_drawdown, annualized_volatility, TRADING_DAYS, RISK_FREE
)


# ── Portfolio Construction ────────────────────────────────────────────────────

class Portfolio:
    """
    Multi-asset portfolio with risk analytics.

    Usage:
        p = Portfolio(returns_df, weights=[0.4, 0.3, 0.3])
        report = p.risk_report()
    """

    def __init__(self, returns: pd.DataFrame,
                 weights: list[float] = None,
                 portfolio_value: float = 1_000_000):

        self.returns         = returns.dropna()
        self.tickers         = list(returns.columns)
        self.n               = len(self.tickers)
        self.portfolio_value = portfolio_value

        if weights is None:
            self.weights = np.array([1 / self.n] * self.n)
        else:
            w = np.array(weights, dtype=float)
            self.weights = w / w.sum()   # normalize

        self.port_returns = self._portfolio_returns()

    def _portfolio_returns(self) -> pd.Series:
        return self.returns.dot(self.weights)

    def update_weights(self, weights: list[float]):
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()
        self.port_returns = self._portfolio_returns()

    # ── Correlation & Covariance ──────────────────────────────────────────────

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()

    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns.cov() * TRADING_DAYS

    # ── Portfolio Risk Metrics ────────────────────────────────────────────────

    def annualized_return(self) -> float:
        return float(self.port_returns.mean() * TRADING_DAYS)

    def annualized_volatility(self) -> float:
        cov = self.returns.cov() * TRADING_DAYS
        return float(np.sqrt(self.weights @ cov.values @ self.weights))

    def sharpe(self) -> float:
        return sharpe_ratio(self.port_returns)

    def max_dd(self) -> float:
        return max_drawdown(self.port_returns)

    def portfolio_var(self, confidence: float = 0.95, horizon: int = 1) -> dict:
        """Portfolio-level VaR using historical simulation."""
        h_var = var_historical(self.port_returns, confidence, horizon)
        cvar  = cvar_historical(self.port_returns, confidence)
        return {
            "var_historical" : round(h_var, 6),
            "cvar"           : round(cvar, 6),
            "var_dollar"     : round(h_var * self.portfolio_value, 2),
            "cvar_dollar"    : round(cvar * self.portfolio_value, 2),
        }

    def component_var(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Component VaR — contribution of each asset to portfolio VaR.
        FRM: Risk decomposition for portfolio management.
        """
        port_var  = var_historical(self.port_returns, confidence)
        cov_daily = self.returns.cov()

        component_vars = []
        for i, ticker in enumerate(self.tickers):
            cov_with_port = cov_daily.iloc[i].dot(self.weights)
            marginal_var  = cov_with_port / self.port_returns.std()
            comp_var      = self.weights[i] * marginal_var * port_var / self.port_returns.std()
            component_vars.append({
                "ticker"        : ticker,
                "weight"        : round(self.weights[i], 4),
                "marginal_var"  : round(float(marginal_var), 6),
                "component_var" : round(float(abs(comp_var)), 6),
                "pct_of_var"    : 0,  # filled below
            })

        df  = pd.DataFrame(component_vars)
        tot = df["component_var"].sum()
        df["pct_of_var"] = (df["component_var"] / tot * 100).round(2) if tot > 0 else 0
        return df

    # ── Risk Report ───────────────────────────────────────────────────────────

    def risk_report(self, confidence: float = 0.95) -> dict:
        var_data = self.portfolio_var(confidence)
        return {
            "tickers"              : self.tickers,
            "weights"              : {t: round(float(w), 4)
                                      for t, w in zip(self.tickers, self.weights)},
            "portfolio_value"      : self.portfolio_value,
            "annualized_return_pct": round(self.annualized_return() * 100, 2),
            "annualized_vol_pct"   : round(self.annualized_volatility() * 100, 2),
            "sharpe_ratio"         : round(self.sharpe(), 4),
            "max_drawdown_pct"     : round(self.max_dd() * 100, 2),
            **var_data,
        }

    # ── Optimization ─────────────────────────────────────────────────────────

    def optimize(self, objective: str = "sharpe") -> np.ndarray:
        """
        Optimize portfolio weights.
        objective: 'sharpe' | 'min_vol' | 'min_var'
        """
        cov = self.returns.cov() * TRADING_DAYS
        mu  = self.returns.mean() * TRADING_DAYS

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds      = tuple((0.05, 0.60) for _ in range(self.n))
        w0          = np.array([1 / self.n] * self.n)

        if objective == "sharpe":
            def neg_sharpe(w):
                ret = float(w @ mu)
                vol = float(np.sqrt(w @ cov.values @ w))
                return -(ret - RISK_FREE) / vol if vol > 0 else 0
            result = minimize(neg_sharpe, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints)

        elif objective == "min_vol":
            def portfolio_vol(w):
                return float(np.sqrt(w @ cov.values @ w))
            result = minimize(portfolio_vol, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints)

        else:
            def portfolio_var_obj(w):
                port_r = self.returns.dot(w)
                return var_historical(port_r)
            result = minimize(portfolio_var_obj, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints)

        if result.success:
            return result.x / result.x.sum()
        return self.weights

    def efficient_frontier(self, n_portfolios: int = 500) -> pd.DataFrame:
        """
        Simulate random portfolios to approximate the efficient frontier.
        Returns DataFrame with return, vol, sharpe for each simulated portfolio.
        """
        cov = self.returns.cov() * TRADING_DAYS
        mu  = self.returns.mean() * TRADING_DAYS

        results = []
        np.random.seed(42)
        for _ in range(n_portfolios):
            w   = np.random.dirichlet(np.ones(self.n))
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov.values @ w))
            sr  = (ret - RISK_FREE) / vol if vol > 0 else 0
            results.append({"return": ret, "volatility": vol, "sharpe": sr,
                            "weights": w.tolist()})

        return pd.DataFrame(results)


if __name__ == "__main__":
    from data_fetcher import fetch_returns, get_date_range

    tickers = ["AAPL", "JPM", "GS", "SPY"]
    start, end = get_date_range("2y")
    returns = fetch_returns(tickers, start, end)

    p = Portfolio(returns, weights=[0.3, 0.25, 0.25, 0.2])
    report = p.risk_report()

    print("\n── Portfolio Risk Report ──")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n── Component VaR ──")
    print(p.component_var())

    print("\n── Optimized Weights (Max Sharpe) ──")
    opt_w = p.optimize("sharpe")
    for t, w in zip(tickers, opt_w):
        print(f"  {t}: {w:.2%}")
