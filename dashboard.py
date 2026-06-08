"""
dashboard.py
------------
Streamlit dashboard for the Financial Risk Dashboard.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Financial Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #21262d; }
  .metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 18px 20px; text-align: center;
  }
  .metric-value { font-size: 1.8rem; font-weight: 800; color: #00d4aa; font-family: monospace; }
  .metric-label { font-size: 0.7rem; color: #7d8590; letter-spacing: 1px;
                  text-transform: uppercase; margin-top: 4px; }
  .risk-box {
    background: #161b22; border-left: 3px solid #00d4aa;
    border-radius: 4px; padding: 14px 18px; margin: 8px 0;
    font-family: monospace; font-size: 0.85rem; color: #c9d1d9; line-height: 1.7;
  }
  h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#21262d"
ACCENT  = "#00d4aa"
PURPLE  = "#7c6af7"
ORANGE  = "#f7a74a"
RED     = "#ff6b6b"
MUTED   = "#7d8590"
TEXT    = "#e6edf3"

PRESET_PORTFOLIOS = {
    "Tech Heavy"     : {"tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"], "weights": [0.3, 0.3, 0.2, 0.2]},
    "Banking Focus"  : {"tickers": ["JPM", "GS", "BAC", "MS"],        "weights": [0.35, 0.25, 0.25, 0.15]},
    "Balanced"       : {"tickers": ["AAPL", "JPM", "SPY", "GS"],      "weights": [0.25, 0.25, 0.30, 0.20]},
    "Custom"         : {"tickers": ["SPY", "QQQ"],                     "weights": [0.5, 0.5]},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Risk Dashboard")
    st.markdown("*FRM-Aligned Quantitative Analytics*")
    st.divider()

    preset = st.selectbox("Portfolio Preset", list(PRESET_PORTFOLIOS.keys()))
    default_tickers = ", ".join(PRESET_PORTFOLIOS[preset]["tickers"])
    ticker_input = st.text_input("Tickers (comma separated)", default_tickers)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("**Portfolio Weights**")
    default_w = PRESET_PORTFOLIOS[preset]["weights"]
    if len(default_w) != len(tickers):
        default_w = [1/len(tickers)] * len(tickers)
    weights_input = st.text_input("Weights (comma separated)",
                                  ", ".join(str(w) for w in default_w))
    try:
        weights = [float(w.strip()) for w in weights_input.split(",")]
        weights = [w/sum(weights) for w in weights]
    except:
        weights = [1/len(tickers)] * len(tickers)

    period = st.select_slider("Period", ["3m", "6m", "1y", "2y", "3y"], value="1y")
    confidence = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
    port_value = st.number_input("Portfolio Value ($)", value=1_000_000, step=100_000)
    st.divider()
    st.markdown("*Built by Darsh Jogani*")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/darsh-jogani-37b97218b)")


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching market data...", ttl=3600)
def load_data(tickers, period):
    from data_fetcher import fetch_returns, fetch_benchmark, get_date_range
    start, end = get_date_range(period)
    returns   = fetch_returns(tickers + ["SPY"], start, end)
    benchmark = returns["SPY"] if "SPY" in returns.columns else None
    asset_returns = returns[[t for t in tickers if t in returns.columns]]
    return asset_returns, benchmark, start, end

try:
    returns, benchmark, start_date, end_date = load_data(tuple(tickers), period)
    valid_tickers = list(returns.columns)

    if not valid_tickers:
        st.error("No valid tickers found. Please check your input.")
        st.stop()

    weights_aligned = weights[:len(valid_tickers)]
    w_sum = sum(weights_aligned)
    weights_aligned = [w/w_sum for w in weights_aligned]

except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()


# ── Portfolio & Risk ──────────────────────────────────────────────────────────
from portfolio import Portfolio
from risk_engine import (full_risk_report, rolling_volatility,
                         drawdown_series, ewma_volatility, var_summary)

port    = Portfolio(returns, weights=weights_aligned, portfolio_value=port_value)
p_report = port.risk_report(confidence)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Financial Risk Dashboard")
st.markdown(f"*{', '.join(valid_tickers)} · {start_date} → {end_date} · {confidence:.0%} Confidence*")
st.divider()

# ── Portfolio KPIs ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, value, label, color=ACCENT):
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:{color}'>{value}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

ret_color = ACCENT if p_report["annualized_return_pct"] > 0 else RED
kpi(k1, f"{p_report['annualized_return_pct']:.1f}%",  "Ann. Return",   ret_color)
kpi(k2, f"{p_report['annualized_vol_pct']:.1f}%",     "Ann. Volatility", ORANGE)
kpi(k3, f"{p_report['sharpe_ratio']:.2f}",            "Sharpe Ratio",
    ACCENT if p_report['sharpe_ratio'] > 1 else ORANGE)
kpi(k4, f"{p_report['max_drawdown_pct']:.1f}%",       "Max Drawdown",  RED)
kpi(k5, f"${p_report['var_dollar']:,.0f}",            f"VaR ({confidence:.0%})", PURPLE)
kpi(k6, f"${p_report['cvar_dollar']:,.0f}",           "CVaR (ES)",     "#f7a74a")

st.markdown("<br>", unsafe_allow_html=True)

# ── Cumulative Returns ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Returns", "📉 Risk Metrics", "🔗 Correlation", "⚖️ Portfolio", "📊 VaR Analysis"
])

with tab1:
    cum_returns = (1 + returns).cumprod() - 1
    port_cum    = (1 + port.port_returns).cumprod() - 1

    fig = go.Figure()
    colors_list = [ACCENT, PURPLE, ORANGE, RED, "#febc2e", "#28c840"]
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Scatter(
            x=cum_returns.index, y=cum_returns[col] * 100,
            name=col, line=dict(color=colors_list[i % len(colors_list)], width=1.5),
            opacity=0.7,
        ))
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum * 100,
        name="Portfolio", line=dict(color=TEXT, width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title="Cumulative Returns (%)",
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis_title="Return (%)", margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Return distribution
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        for i, col in enumerate(returns.columns):
            fig2.add_trace(go.Histogram(
                x=returns[col] * 100, name=col, opacity=0.7,
                nbinsx=50, marker_color=colors_list[i % len(colors_list)],
            ))
        fig2.update_layout(
            title="Daily Return Distribution (%)", barmode="overlay",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
            font_color=MUTED, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        port_returns_ser = port.port_returns
        roll_vol = rolling_volatility(port_returns_ser, window=21)
        ewma_vol = ewma_volatility(port_returns_ser)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol * 100,
                                  name="21D Rolling Vol", line=dict(color=ACCENT, width=1.5)))
        fig3.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol * 100,
                                  name="EWMA Vol (λ=0.94)", line=dict(color=ORANGE, width=1.5)))
        fig3.update_layout(
            title="Portfolio Volatility — Rolling vs EWMA (%)",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
            font_color=MUTED, yaxis_title="Annualized Vol (%)", margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)


with tab2:
    # Drawdown
    dd = drawdown_series(port.port_returns) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd, fill="tozeroy",
        fillcolor="rgba(255,107,107,0.2)", line=dict(color=RED, width=1.5),
        name="Drawdown",
    ))
    fig_dd.update_layout(
        title="Portfolio Drawdown (%)",
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, yaxis_title="Drawdown (%)", margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Individual risk metrics table
    st.markdown("### Individual Asset Risk Metrics")
    risk_rows = []
    for ticker in valid_tickers:
        r = returns[ticker]
        b = benchmark if benchmark is not None else r
        from risk_engine import (sharpe_ratio, sortino_ratio, max_drawdown,
                                  annualized_volatility, var_historical,
                                  cvar_historical, beta as beta_fn, alpha as alpha_fn)
        risk_rows.append({
            "Ticker"    : ticker,
            "Ann. Return%": round(r.mean() * 252 * 100, 2),
            "Ann. Vol%"   : round(annualized_volatility(r) * 100, 2),
            "Sharpe"      : round(sharpe_ratio(r), 3),
            "Sortino"     : round(sortino_ratio(r), 3),
            "Max DD%"     : round(max_drawdown(r) * 100, 2),
            f"VaR {confidence:.0%}": round(var_historical(r, confidence) * 100, 3),
            "CVaR"        : round(cvar_historical(r, confidence) * 100, 3),
            "Beta"        : round(beta_fn(r, b), 3),
        })

    df_risk = pd.DataFrame(risk_rows).set_index("Ticker")
    st.dataframe(
        df_risk.style.background_gradient(subset=["Sharpe"], cmap="RdYlGn")
                     .background_gradient(subset=["Max DD%"], cmap="RdYlGn_r"),
        use_container_width=True,
    )


with tab3:
    corr = port.correlation_matrix()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Asset Correlation Matrix",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Rolling correlation (first two assets)
    if len(valid_tickers) >= 2:
        roll_corr = returns[valid_tickers[0]].rolling(63).corr(returns[valid_tickers[1]])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            fill="tozeroy", fillcolor="rgba(124,106,247,0.15)",
            line=dict(color=PURPLE, width=1.5),
            name=f"{valid_tickers[0]} / {valid_tickers[1]}",
        ))
        fig_rc.add_hline(y=0, line_dash="dash", line_color=MUTED, opacity=0.5)
        fig_rc.update_layout(
            title=f"63-Day Rolling Correlation: {valid_tickers[0]} vs {valid_tickers[1]}",
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
            font_color=MUTED, yaxis_title="Correlation", margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_rc, use_container_width=True)


with tab4:
    col1, col2 = st.columns(2)
    with col1:
        # Weights pie
        fig_w = px.pie(
            values=weights_aligned, names=valid_tickers,
            title="Portfolio Weights",
            color_discrete_sequence=[ACCENT, PURPLE, ORANGE, RED, "#febc2e"],
            hole=0.45,
        )
        fig_w.update_layout(
            template="plotly_dark", paper_bgcolor=BG,
            font_color=MUTED, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        # Component VaR
        comp_var = port.component_var(confidence)
        fig_cv = px.bar(
            comp_var, x="ticker", y="pct_of_var",
            title="Component VaR Contribution (%)",
            color="pct_of_var",
            color_continuous_scale=[[0, ACCENT], [0.5, ORANGE], [1, RED]],
            text="pct_of_var",
        )
        fig_cv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cv.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
            font_color=MUTED, showlegend=False, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Efficient frontier
    st.markdown("### Efficient Frontier (Monte Carlo Simulation)")
    ef = port.efficient_frontier(n_portfolios=800)
    max_sr_idx = ef["sharpe"].idxmax()
    min_vol_idx = ef["volatility"].idxmin()

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=ef["volatility"] * 100, y=ef["return"] * 100,
        mode="markers", marker=dict(
            color=ef["sharpe"], colorscale="Viridis",
            size=5, opacity=0.6,
            colorbar=dict(title="Sharpe", tickfont=dict(color=MUTED)),
        ), name="Simulated Portfolios",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[ef.loc[max_sr_idx, "volatility"] * 100],
        y=[ef.loc[max_sr_idx, "return"] * 100],
        mode="markers+text", marker=dict(color=ACCENT, size=14, symbol="star"),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=ACCENT), name="Max Sharpe",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[ef.loc[min_vol_idx, "volatility"] * 100],
        y=[ef.loc[min_vol_idx, "return"] * 100],
        mode="markers+text", marker=dict(color=ORANGE, size=14, symbol="diamond"),
        text=["Min Vol"], textposition="top right",
        textfont=dict(color=ORANGE), name="Min Volatility",
    ))
    fig_ef.update_layout(
        title="Efficient Frontier — 800 Simulated Portfolios",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_ef, use_container_width=True)


with tab5:
    st.markdown("### Value at Risk — Three Methods Compared")

    var_data = var_summary(port.port_returns, confidence, 1, port_value)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"${var_data['var_hist_dollar']:,.0f}",  "Historical VaR",    PURPLE)
    kpi(c2, f"${var_data['var_param_dollar']:,.0f}", "Parametric VaR",    ORANGE)
    kpi(c3, f"${var_data['var_mc_dollar']:,.0f}",   "Monte Carlo VaR",   ACCENT)
    kpi(c4, f"${var_data['cvar_dollar']:,.0f}",     "CVaR / ES",         RED)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class='risk-box'>
    <strong style='color:#e6edf3'>FRM Note:</strong> At {confidence:.0%} confidence over a 1-day horizon,
    the portfolio has a <strong style='color:#7c6af7'>Historical VaR of ${var_data['var_hist_dollar']:,.0f}</strong>,
    meaning there is a {(1-confidence):.0%} probability of losing more than this amount in a single day.
    The <strong style='color:#ff6b6b'>CVaR (Expected Shortfall) of ${var_data['cvar_dollar']:,.0f}</strong>
    represents the average loss in the worst {(1-confidence):.0%}% of scenarios — a coherent risk measure
    preferred under Basel III over standard VaR.
    </div>""", unsafe_allow_html=True)

    # VaR comparison bar
    methods = ["Historical", "Parametric", "Monte Carlo", "CVaR"]
    values  = [var_data["var_hist_dollar"], var_data["var_param_dollar"],
                var_data["var_mc_dollar"],  var_data["cvar_dollar"]]
    fig_var = go.Figure(go.Bar(
        x=methods, y=values, marker_color=[PURPLE, ORANGE, ACCENT, RED],
        text=[f"${v:,.0f}" for v in values], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig_var.update_layout(
        title=f"VaR Comparison — ${port_value:,.0f} Portfolio · {confidence:.0%} Confidence",
        yaxis_title="Dollar Loss ($)",
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, showlegend=False, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Rolling VaR
    roll_var_series = port.port_returns.rolling(63).apply(
        lambda x: var_historical(pd.Series(x), confidence) * port_value, raw=False
    )
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=roll_var_series.index, y=roll_var_series,
        fill="tozeroy", fillcolor="rgba(124,106,247,0.15)",
        line=dict(color=PURPLE, width=1.5), name="63D Rolling VaR",
    ))
    fig_rv.update_layout(
        title=f"Rolling 63-Day Historical VaR ($) — {confidence:.0%} Confidence",
        yaxis_title="VaR ($)",
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font_color=MUTED, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_rv, use_container_width=True)
