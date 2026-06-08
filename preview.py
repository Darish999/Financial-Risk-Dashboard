"""
preview.py
----------
Generates a realistic dashboard preview image for the GitHub repo.
Run with: python preview.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#21262d"
ACCENT  = "#00d4aa"
PURPLE  = "#7c6af7"
ORANGE  = "#f7a74a"
RED     = "#ff6b6b"
TEXT    = "#e6edf3"
MUTED   = "#7d8590"


def generate_preview(output_path: str = "preview.png"):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    # Title
    fig.text(0.03, 0.96, "Financial Risk Dashboard",
             color=TEXT, fontsize=15, fontweight="bold",
             va="top", fontfamily="monospace")
    fig.text(0.03, 0.92,
             "VaR · CVaR · Sharpe · Sortino · Drawdown · Efficient Frontier · EWMA Volatility",
             color=MUTED, fontsize=8, va="top", fontfamily="monospace")

    # ── KPI row ──────────────────────────────────────────────────────────────
    kpi_data = [
        ("18.4%", "Ann. Return",    ACCENT),
        ("14.2%", "Ann. Volatility",ORANGE),
        ("1.24",  "Sharpe Ratio",   ACCENT),
        ("-12.8%","Max Drawdown",   RED),
        ("$24,800","VaR 95%",       PURPLE),
        ("$31,200","CVaR / ES",     "#f7a74a"),
    ]
    kpi_axes = []
    for i, (val, lbl, col) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i % 4]) if i < 4 else None
        if i >= 4:
            break
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.58, val, transform=ax.transAxes,
                ha="center", va="center", color=col,
                fontsize=16, fontweight="bold", fontfamily="monospace")
        ax.text(0.5, 0.18, lbl, transform=ax.transAxes,
                ha="center", va="center", color=MUTED,
                fontsize=7, fontfamily="monospace")

    # ── Cumulative returns ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    np.random.seed(42)
    days   = 252
    tickers = ["AAPL", "JPM", "GS", "Portfolio"]
    colors  = [ACCENT, PURPLE, ORANGE, TEXT]
    widths  = [1.2, 1.2, 1.2, 2.2]
    dashes  = ["-", "-", "-", "--"]

    for i, (t, c, w, d) in enumerate(zip(tickers, colors, widths, dashes)):
        drift = np.random.uniform(0.06, 0.22)
        vol   = np.random.uniform(0.12, 0.25) / np.sqrt(days)
        rets  = np.random.normal(drift / days, vol, days)
        cum   = (np.cumprod(1 + rets) - 1) * 100
        ax1.plot(cum, color=c, linewidth=w, linestyle=d, label=t, alpha=0.85)

    ax1.set_facecolor(SURFACE)
    ax1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor(BORDER)
    ax1.set_title("Cumulative Returns (%)", color=TEXT, fontsize=9,
                  pad=6, fontfamily="monospace")
    ax1.legend(fontsize=7, labelcolor=TEXT, facecolor=SURFACE,
               edgecolor=BORDER, loc="upper left")
    ax1.set_ylabel("%", color=MUTED, fontsize=7)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2:])
    np.random.seed(7)
    rets2 = np.random.normal(0.0005, 0.012, days)
    cum2  = np.cumprod(1 + rets2)
    peak2 = np.maximum.accumulate(cum2)
    dd    = (cum2 - peak2) / peak2 * 100

    ax2.fill_between(range(days), dd, 0, color=RED, alpha=0.35)
    ax2.plot(range(days), dd, color=RED, linewidth=1.3)
    ax2.set_facecolor(SURFACE)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor(BORDER)
    ax2.set_title("Portfolio Drawdown (%)", color=TEXT, fontsize=9,
                  pad=6, fontfamily="monospace")
    ax2.set_ylabel("%", color=MUTED, fontsize=7)

    # ── Efficient frontier ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    np.random.seed(99)
    n_pts  = 600
    vols   = np.random.uniform(0.08, 0.30, n_pts)
    rets3  = vols * np.random.uniform(0.4, 1.6, n_pts) + np.random.normal(0, 0.02, n_pts)
    sharpe = (rets3 - 0.05) / vols
    sc = ax3.scatter(vols * 100, rets3 * 100, c=sharpe, cmap="plasma",
                     s=8, alpha=0.6)
    # Max sharpe star
    best = np.argmax(sharpe)
    ax3.scatter(vols[best] * 100, rets3[best] * 100,
                color=ACCENT, s=120, marker="*", zorder=5)
    ax3.set_facecolor(SURFACE)
    ax3.tick_params(colors=MUTED, labelsize=7)
    for spine in ax3.spines.values(): spine.set_edgecolor(BORDER)
    ax3.set_title("Efficient Frontier — 600 Simulated Portfolios",
                  color=TEXT, fontsize=9, pad=6, fontfamily="monospace")
    ax3.set_xlabel("Volatility (%)", color=MUTED, fontsize=7)
    ax3.set_ylabel("Return (%)", color=MUTED, fontsize=7)

    # ── VaR comparison bar ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2:])
    methods = ["Historical", "Parametric", "Monte Carlo", "CVaR (ES)"]
    var_vals = [24800, 23100, 25400, 31200]
    bar_colors = [PURPLE, ORANGE, ACCENT, RED]
    bars = ax4.bar(methods, var_vals, color=bar_colors, width=0.6, alpha=0.9)
    for bar, val in zip(bars, var_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"${val:,}", ha="center", va="bottom",
                 color=MUTED, fontsize=7, fontfamily="monospace")
    ax4.set_facecolor(SURFACE)
    ax4.tick_params(colors=MUTED, labelsize=7, axis="x")
    ax4.tick_params(colors=MUTED, labelsize=7, axis="y")
    for spine in ax4.spines.values(): spine.set_edgecolor(BORDER)
    ax4.set_title("VaR Method Comparison — $1M Portfolio · 95%",
                  color=TEXT, fontsize=9, pad=6, fontfamily="monospace")
    ax4.set_ylabel("Dollar Loss ($)", color=MUTED, fontsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Preview saved to {output_path}")


if __name__ == "__main__":
    generate_preview("preview.png")
