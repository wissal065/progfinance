import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# ==================================================
# Black-Scholes helper functions
# ==================================================
def bs_d1(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_d2(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    return bs_d1(S, K, tau, r, sigma) - sigma * np.sqrt(tau)


def bs_price_delta(option_type: str, S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    """
    Returns Black-Scholes price and delta for one option.
    Financial meaning:
    - price = marked-to-model value for current market state
    - delta = local sensitivity to underlying, used for dynamic hedge rebalancing
    """
    if tau <= 1e-10:
        if option_type == "call":
            intrinsic = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        else:
            intrinsic = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        return intrinsic, delta

    sigma = max(sigma, 1e-6)
    d1_val = bs_d1(S, K, tau, r, sigma)
    d2_val = bs_d2(S, K, tau, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1_val) - K * np.exp(-r * tau) * norm.cdf(d2_val)
        delta = norm.cdf(d1_val)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
        delta = norm.cdf(d1_val) - 1.0

    return float(price), float(delta)


# ==================================================
# Strategy and simulation builders
# ==================================================
def build_strategy(strategy: str, S0: float, qty: float, width_pct: float) -> list[dict]:
    """Creates portfolio legs with quantity sign convention (positive = long)."""
    width = S0 * width_pct

    if strategy == "Long call":
        return [{"type": "call", "K": S0, "qty": qty}]

    if strategy == "Long put":
        return [{"type": "put", "K": S0, "qty": qty}]

    if strategy == "Straddle":
        return [
            {"type": "call", "K": S0, "qty": qty},
            {"type": "put", "K": S0, "qty": qty},
        ]

    if strategy == "Strangle":
        return [
            {"type": "call", "K": S0 + width, "qty": qty},
            {"type": "put", "K": max(S0 - width, 1.0), "qty": qty},
        ]

    if strategy == "Bull call spread":
        return [
            {"type": "call", "K": S0, "qty": qty},
            {"type": "call", "K": S0 + width, "qty": -qty},
        ]

    # Custom portfolio: simple 2-leg builder
    return [
        {"type": "call", "K": S0, "qty": qty},
        {"type": "put", "K": S0, "qty": 0.0},
    ]


def simulate_underlying_path(S0: float, mu: float, sigma_real: float, T: float, n_steps: int, seed: int) -> np.ndarray:
    """Simulates realized market path (this drives actual hedge PnL and realized volatility)."""
    rng = np.random.default_rng(seed)
    dt = T / (n_steps - 1)
    prices = np.zeros(n_steps)
    prices[0] = S0

    for t in range(1, n_steps):
        z = rng.normal(0, 1)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)

    return prices


def simulate_implied_vol_path(
    sigma0: float,
    n_steps: int,
    mode: str,
    vol_noise: float,
    low_regime: float,
    high_regime: float,
    seed: int,
) -> np.ndarray:
    """Simulates the implied vol used for repricing at each time step."""
    rng = np.random.default_rng(seed)
    vols = np.zeros(n_steps)
    vols[0] = sigma0

    if mode == "Stochastic noise":
        for t in range(1, n_steps):
            vols[t] = max(0.05, vols[t - 1] + rng.normal(0, vol_noise))
    else:
        # Regime shifts: low-vol to high-vol then partial normalization
        split_1 = n_steps // 3
        split_2 = 2 * n_steps // 3
        for t in range(1, n_steps):
            if t < split_1:
                target = low_regime
            elif t < split_2:
                target = high_regime
            else:
                target = 0.5 * (low_regime + high_regime)
            vols[t] = max(0.05, 0.9 * vols[t - 1] + 0.1 * target + rng.normal(0, vol_noise))

    return vols


def realized_annual_vol(path: np.ndarray, T_years: float) -> float:
    log_returns = np.diff(np.log(path))
    if len(log_returns) == 0:
        return 0.0
    dt = T_years / len(log_returns)
    return float(log_returns.std(ddof=1) / np.sqrt(dt)) if len(log_returns) > 1 else 0.0


def run_dynamic_hedged_backtest(
    legs: list[dict],
    S_path: np.ndarray,
    iv_path: np.ndarray,
    r: float,
    T: float,
    tx_cost_rate: float,
) -> pd.DataFrame:
    """
    Reprices the full option portfolio each step and dynamically delta hedges using underlying shares.
    Trading friction model: cost = tx_cost_rate * |change in hedge| * S
    """
    n_steps = len(S_path)
    dt = T / (n_steps - 1)

    option_values = np.zeros(n_steps)
    portfolio_delta = np.zeros(n_steps)
    hedge_shares = np.zeros(n_steps)
    cumulative_tc = np.zeros(n_steps)
    pnl = np.zeros(n_steps)

    cash = 0.0
    h_prev = 0.0
    tc_cum = 0.0

    for i in range(n_steps):
        tau = max(T - i * dt, 0.0)
        S_t = float(S_path[i])
        sigma_t = float(iv_path[i])

        # 1) Mark option portfolio and total option delta
        opt_val_t = 0.0
        opt_delta_t = 0.0
        for leg in legs:
            leg_price, leg_delta = bs_price_delta(leg["type"], S_t, leg["K"], tau, r, sigma_t)
            opt_val_t += leg["qty"] * leg_price
            opt_delta_t += leg["qty"] * leg_delta

        option_values[i] = opt_val_t
        portfolio_delta[i] = opt_delta_t

        # 2) Cash account earns risk-free between rehedges
        if i > 0:
            cash *= (1.0 + r * dt)

        # 3) Rebalance hedge position to -delta (delta-neutral target)
        h_new = -opt_delta_t
        trade_shares = h_new - h_prev
        trade_notional = trade_shares * S_t
        tc = tx_cost_rate * abs(trade_notional)

        cash -= trade_notional
        cash -= tc
        tc_cum += tc

        hedge_shares[i] = h_new
        cumulative_tc[i] = tc_cum

        # 4) Hedged portfolio PnL over time
        total_mark_to_market = opt_val_t + h_new * S_t + cash
        pnl[i] = total_mark_to_market

        h_prev = h_new

    return pd.DataFrame(
        {
            "step": np.arange(n_steps),
            "underlying": S_path,
            "implied_vol": iv_path,
            "option_value": option_values,
            "portfolio_delta": portfolio_delta,
            "hedge_shares": hedge_shares,
            "cum_tx_cost": cumulative_tc,
            "pnl": pnl,
        }
    )


def make_smooth_vs_volatile_paths(S0: float, ST: float, n_steps: int, bump: float) -> tuple[np.ndarray, np.ndarray]:
    """Two paths with same start and end price, but very different path characteristics."""
    x = np.linspace(0, 1, n_steps)
    smooth = S0 + (ST - S0) * x
    volatile = S0 + (ST - S0) * x + bump * np.sin(8 * np.pi * x) + 0.5 * bump * np.sin(17 * np.pi * x)
    volatile[0] = S0
    volatile[-1] = ST
    return smooth, volatile


@st.cache_data(show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    return yf.download(ticker, period=period, auto_adjust=True, progress=False)


# ==================================================
# Page config and style
# ==================================================
st.set_page_config(page_title="Dynamic Option Portfolio Pricer", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 0.6rem; max-width: 1450px;}
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1c1f26, #12151b);
        border: 1px solid #2a2e39;
        border-radius: 12px;
        padding: 10px 12px;
    }
    button[data-baseweb="tab"] {height: 40px; border-radius: 10px; background-color: rgba(255,255,255,0.03);}
    button[data-baseweb="tab"][aria-selected="true"] {background-color: rgba(255,255,255,0.10);}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dynamic Option-Portfolio Pricer")
st.caption(
    "Why one-shot option pricing is not enough: dynamic repricing, changing implied vol, hedging, and path risk."
)


# ==================================================
# Sidebar controls
# ==================================================
st.sidebar.header("Portfolio + simulation controls")

strategy = st.sidebar.selectbox(
    "Strategy",
    ["Long call", "Long put", "Straddle", "Strangle", "Bull call spread", "Custom portfolio"],
)

S0 = st.sidebar.number_input("Initial underlying price S0", min_value=10.0, max_value=1000.0, value=100.0, step=1.0)
T = st.sidebar.slider("Horizon / maturity (years)", 0.1, 2.0, 1.0, 0.05)
r = st.sidebar.slider("Risk-free rate r", 0.0, 0.10, 0.03, 0.005)
qty = st.sidebar.slider("Contracts per leg", 1.0, 10.0, 1.0, 1.0)
width_pct = st.sidebar.slider("Strike width (% of S0) for spreads/strangles", 0.05, 0.35, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Path assumptions")
mu = st.sidebar.slider("Underlying drift μ", -0.05, 0.15, 0.03, 0.01)
sigma_real = st.sidebar.slider("Realized volatility assumption", 0.05, 0.80, 0.20, 0.01)
n_steps = st.sidebar.slider("Repricing / hedge steps", 20, 300, 120, 10)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("Implied volatility dynamics")
iv_mode = st.sidebar.selectbox("IV path model", ["Stochastic noise", "Regime shifts"])
sigma0 = st.sidebar.slider("Initial implied volatility σ0", 0.05, 0.90, 0.25, 0.01)
vol_noise = st.sidebar.slider("IV noise level", 0.0, 0.08, 0.01, 0.001)
low_regime = st.sidebar.slider("Low IV regime", 0.05, 0.60, 0.18, 0.01)
high_regime = st.sidebar.slider("High IV regime", 0.10, 1.00, 0.40, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Hedging friction")
tx_cost_rate = st.sidebar.slider("Proportional transaction cost", 0.0, 0.01, 0.001, 0.0001)


# ==================================================
# Portfolio construction (includes optional custom legs)
# ==================================================
legs = build_strategy(strategy, S0, qty, width_pct)
if strategy == "Custom portfolio":
    st.sidebar.markdown("Custom portfolio legs")
    c1, c2 = st.sidebar.columns(2)
    call_k = c1.number_input("Call strike", min_value=1.0, value=float(S0), step=1.0)
    call_q = c2.slider("Call qty", -5.0, 5.0, 1.0, 1.0)

    c3, c4 = st.sidebar.columns(2)
    put_k = c3.number_input("Put strike", min_value=1.0, value=float(S0), step=1.0)
    put_q = c4.slider("Put qty", -5.0, 5.0, 0.0, 1.0)

    legs = [
        {"type": "call", "K": call_k, "qty": call_q},
        {"type": "put", "K": put_k, "qty": put_q},
    ]


# ==================================================
# Main simulation data
# ==================================================
S_path = simulate_underlying_path(S0, mu, sigma_real, T, n_steps, int(seed))
iv_path = simulate_implied_vol_path(sigma0, n_steps, iv_mode, vol_noise, low_regime, high_regime, int(seed) + 1)
base_df = run_dynamic_hedged_backtest(legs, S_path, iv_path, r, T, tx_cost_rate)

realized_vol = realized_annual_vol(S_path, T)
avg_iv = float(np.mean(iv_path))
summary = {
    "initial_portfolio_value": float(base_df["option_value"].iloc[0]),
    "final_portfolio_value": float(base_df["option_value"].iloc[-1]),
    "realized_pnl": float(base_df["pnl"].iloc[-1]),
    "total_tx_costs": float(base_df["cum_tx_cost"].iloc[-1]),
    "realized_vol": realized_vol,
    "avg_implied_vol": avg_iv,
}


# ==================================================
# Tabs
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Theory / intuition", "Dynamic portfolio simulator", "Path risk comparison", "Real-market intuition"]
)

# --------------------------------------------------
# Tab 1 - Theory
# --------------------------------------------------
with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Why static Black-Scholes pricing is insufficient")
        st.markdown(
            """
            - Traders manage **portfolios of options**, not isolated contracts.
            - The portfolio must be **repriced at each time step** as:\
              underlying price, time-to-maturity, and implied volatility change.
            - Delta is not constant, so hedge positions must be rebalanced dynamically.
            - Rebalancing introduces **trading costs**, making outcomes path-dependent.
            """
        )
        st.latex(r"\text{Transaction cost}_t = c \times |\Delta h_t| \times S_t")
        st.latex(r"\text{PnL}_t = V_{\text{options},t} + h_t S_t + \text{cash}_t")

    with c2:
        st.subheader("Current strategy legs")
        st.dataframe(pd.DataFrame(legs), use_container_width=True, hide_index=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Realized vol (path)", f"{summary['realized_vol']:.2%}")
        m2.metric("Average implied vol", f"{summary['avg_implied_vol']:.2%}")
        m3.metric("Tx cost rate", f"{tx_cost_rate:.3%}")

    st.info(
        "In practice, implied volatility is often above realized volatility because option sellers require compensation for risk and market frictions."
    )

# --------------------------------------------------
# Tab 2 - Dynamic simulator
# --------------------------------------------------
with tab2:
    st.subheader("Dynamic repricing + delta hedge results")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Initial portfolio value", f"{summary['initial_portfolio_value']:.2f}")
    r1c2.metric("Final portfolio value", f"{summary['final_portfolio_value']:.2f}")
    r1c3.metric("Realized hedged PnL", f"{summary['realized_pnl']:.2f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Total transaction costs", f"{summary['total_tx_costs']:.2f}")
    r2c2.metric("Realized volatility", f"{summary['realized_vol']:.2%}")
    r2c3.metric("Average implied volatility", f"{summary['avg_implied_vol']:.2%}")

    g1, g2 = st.columns(2)
    with g1:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(y=base_df["underlying"], mode="lines", name="Underlying", line=dict(width=2)))
        fig_s.update_layout(title="Underlying price path", xaxis_title="Step", yaxis_title="Price", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_s, use_container_width=True)

    with g2:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(y=base_df["implied_vol"], mode="lines", name="Implied vol", line=dict(width=2)))
        fig_iv.update_layout(title="Implied volatility path", xaxis_title="Step", yaxis_title="Volatility", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_iv, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(y=base_df["option_value"], mode="lines", name="Option portfolio", line=dict(width=2)))
        fig_v.update_layout(title="Portfolio option value", xaxis_title="Step", yaxis_title="Value", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_v, use_container_width=True)

    with g4:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(y=base_df["portfolio_delta"], mode="lines", name="Portfolio delta", line=dict(width=2)))
        fig_d.update_layout(title="Portfolio delta", xaxis_title="Step", yaxis_title="Delta", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_d, use_container_width=True)

    g5, g6 = st.columns(2)
    with g5:
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(y=base_df["cum_tx_cost"], mode="lines", name="Cumulative costs", line=dict(width=2)))
        fig_tc.update_layout(title="Cumulative transaction costs", xaxis_title="Step", yaxis_title="Cost", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_tc, use_container_width=True)

    with g6:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(y=base_df["pnl"], mode="lines", name="PnL", line=dict(width=2)))
        fig_p.update_layout(title="Hedged PnL over time", xaxis_title="Step", yaxis_title="PnL", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Summary table")
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Initial portfolio value",
                "Final portfolio value",
                "Realized PnL",
                "Total transaction costs",
                "Realized volatility",
                "Average implied volatility",
            ],
            "Value": [
                f"{summary['initial_portfolio_value']:.4f}",
                f"{summary['final_portfolio_value']:.4f}",
                f"{summary['realized_pnl']:.4f}",
                f"{summary['total_tx_costs']:.4f}",
                f"{summary['realized_vol']:.2%}",
                f"{summary['avg_implied_vol']:.2%}",
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# --------------------------------------------------
# Tab 3 - Path risk comparison
# --------------------------------------------------
with tab3:
    st.subheader("Same terminal price, different path -> different PnL")
    ST = st.slider("Common terminal price for Path A and B", 0.7 * S0, 1.3 * S0, S0, 1.0)
    bump = st.slider("Volatility intensity for Path B", 1.0, 40.0, 12.0, 1.0)

    path_a, path_b = make_smooth_vs_volatile_paths(S0, ST, n_steps, bump)
    iv_cmp = simulate_implied_vol_path(sigma0, n_steps, iv_mode, vol_noise, low_regime, high_regime, int(seed) + 7)

    res_a = run_dynamic_hedged_backtest(legs, path_a, iv_cmp, r, T, tx_cost_rate)
    res_b = run_dynamic_hedged_backtest(legs, path_b, iv_cmp, r, T, tx_cost_rate)

    cpa, cpb, cpd = st.columns(3)
    pnl_a = float(res_a["pnl"].iloc[-1])
    pnl_b = float(res_b["pnl"].iloc[-1])
    cpa.metric("Final PnL - Path A (smooth)", f"{pnl_a:.2f}")
    cpb.metric("Final PnL - Path B (volatile)", f"{pnl_b:.2f}")
    cpd.metric("PnL difference", f"{(pnl_b - pnl_a):.2f}")

    ga, gb = st.columns(2)
    with ga:
        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(y=res_a["underlying"], name="Path A (smooth)", line=dict(width=2)))
        fig_path.add_trace(go.Scatter(y=res_b["underlying"], name="Path B (volatile)", line=dict(width=2)))
        fig_path.update_layout(title="Underlying paths (same start and end)", xaxis_title="Step", yaxis_title="Price", template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_path, use_container_width=True)

    with gb:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(y=res_a["pnl"], name="PnL Path A", line=dict(width=2)))
        fig_cmp.add_trace(go.Scatter(y=res_b["pnl"], name="PnL Path B", line=dict(width=2)))
        fig_cmp.update_layout(title="PnL path comparison", xaxis_title="Step", yaxis_title="PnL", template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown(
        """
        **Interpretation:**
        - Both paths end at the same underlying price, so a naive terminal-only view may say outcomes should match.
        - In dynamic hedging, the hedge is traded at intermediate prices, and each rebalance incurs cost.
        - Therefore **path shape matters**: choppier paths often force more (or larger) hedge adjustments and higher costs.
        """
    )

# --------------------------------------------------
# Tab 4 - Real-market intuition
# --------------------------------------------------
with tab4:
    st.subheader("Historical realized vol vs assumed implied vol")

    ticker_map = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Tesla": "TSLA",
        "SPY ETF": "SPY",
    }

    rc1, rc2, rc3 = st.columns(3)
    stock_name = rc1.selectbox("Ticker", list(ticker_map.keys()))
    period = rc2.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    implied_assumption = rc3.slider("User implied vol assumption", 0.05, 1.00, 0.30, 0.01)

    ticker = ticker_map[stock_name]
    raw = load_prices(ticker, period)

    if raw.empty:
        st.warning("Could not fetch market data for this ticker right now.")
    else:
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)
        ret = np.log(close / close.shift(1)).dropna()

        hist_vol = float(ret.std() * np.sqrt(252))
        S_last = float(close.iloc[-1])
        T_mini = 30 / 365

        bs_price_hist, _ = bs_price_delta("call", S_last, S_last, T_mini, r, hist_vol)
        bs_price_implied, _ = bs_price_delta("call", S_last, S_last, T_mini, r, implied_assumption)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Last price", f"{S_last:.2f}")
        k2.metric("Realized hist vol", f"{hist_vol:.2%}")
        k3.metric("Assumed implied vol", f"{implied_assumption:.2%}")
        k4.metric("30d ATM call premium gap", f"{(bs_price_implied - bs_price_hist):.2f}")

        h1, h2 = st.columns(2)
        with h1:
            f_close = go.Figure()
            f_close.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close", line=dict(width=2)))
            f_close.update_layout(title=f"{ticker} price history", xaxis_title="Date", yaxis_title="Price", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(f_close, use_container_width=True)

        with h2:
            f_ret = go.Figure()
            f_ret.add_trace(go.Scatter(x=ret.index, y=ret.values, mode="lines", name="Log returns", line=dict(width=1.6)))
            f_ret.update_layout(title=f"{ticker} log returns", xaxis_title="Date", yaxis_title="Log return", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(f_ret, use_container_width=True)

        st.markdown(
            """
            ### Why implied volatility is often above realized volatility
            1. **Market risk premium**: option sellers demand compensation for bearing crash/tail risk.
            2. **Liquidity constraints**: warehousing option risk and market impact are costly.
            3. **Supply/demand imbalance**: persistent demand for downside protection lifts option prices.
            4. **Stress periods and hedging pressure**: in turbulent markets, hedging flows can push implied vols up.
            """
        )

st.divider()
st.caption("Educational project dashboard: dynamic portfolio repricing under changing implied volatility and trading frictions.")
