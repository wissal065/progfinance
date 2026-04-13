import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
import yfinance as yf
import numpy as np


# ==================================================
# Fonctions Black-Scholes
# ==================================================
def bs_d1(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_d2(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    return bs_d1(S, K, tau, r, sigma) - sigma * np.sqrt(tau)


def bs_price_delta(option_type: str, S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    """
    Retourne le prix Black-Scholes et le delta d'une option européenne.
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
# Construction de stratégies simples
# ==================================================
def build_strategy(strategy: str, S0: float, qty: float) -> list[dict]:
    """
    Construit un portefeuille simple d'options.
    """
    if strategy == "Call long":
        return [{"type": "call", "K": S0, "qty": qty}]

    if strategy == "Put long":
        return [{"type": "put", "K": S0, "qty": qty}]

    if strategy == "Straddle":
        return [
            {"type": "call", "K": S0, "qty": qty},
            {"type": "put", "K": S0, "qty": qty},
        ]

    return [{"type": "call", "K": S0, "qty": qty}]


# ==================================================
# Simulation du sous-jacent
# ==================================================
def simulate_underlying_path(
    S0: float,
    mu: float,
    sigma_real: float,
    T: float,
    n_steps: int,
    seed: int
) -> np.ndarray:
    """
    Simule une trajectoire réalisée du sous-jacent.
    """
    rng = np.random.default_rng(seed)
    dt = T / (n_steps - 1)

    prices = np.zeros(n_steps)
    prices[0] = S0

    for t in range(1, n_steps):
        z = rng.normal(0, 1)
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z
        )

    return prices


def realized_annual_vol(path: np.ndarray, T_years: float) -> float:
    log_returns = np.diff(np.log(path))
    if len(log_returns) <= 1:
        return 0.0
    dt = T_years / len(log_returns)
    return float(log_returns.std(ddof=1) / np.sqrt(dt))


# ==================================================
# Backtest dynamique avec couverture delta
# ==================================================
def run_dynamic_hedged_backtest(
    legs: list[dict],
    S_path: np.ndarray,
    sigma_iv: float,
    r: float,
    T: float,
    tx_cost_rate: float,
) -> pd.DataFrame:
    """
    Reprice le portefeuille à chaque date et couvre le delta avec le sous-jacent.
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

        opt_val_t = 0.0
        opt_delta_t = 0.0

        for leg in legs:
            leg_price, leg_delta = bs_price_delta(
                leg["type"], S_t, leg["K"], tau, r, sigma_iv
            )
            opt_val_t += leg["qty"] * leg_price
            opt_delta_t += leg["qty"] * leg_delta

        option_values[i] = opt_val_t
        portfolio_delta[i] = opt_delta_t

        if i > 0:
            cash *= (1.0 + r * dt)

        h_new = -opt_delta_t
        trade_shares = h_new - h_prev
        trade_notional = trade_shares * S_t
        tc = tx_cost_rate * abs(trade_notional)

        cash -= trade_notional
        cash -= tc
        tc_cum += tc

        hedge_shares[i] = h_new
        cumulative_tc[i] = tc_cum

        pnl[i] = opt_val_t + h_new * S_t + cash
        h_prev = h_new

    return pd.DataFrame(
        {
            "step": np.arange(n_steps),
            "underlying": S_path,
            "option_value": option_values,
            "portfolio_delta": portfolio_delta,
            "hedge_shares": hedge_shares,
            "cum_tx_cost": cumulative_tc,
            "pnl": pnl,
        }
    )


# ==================================================
# Comparaison de deux trajectoires : même début, même fin
# ==================================================
def make_smooth_vs_volatile_paths(
    S0: float,
    ST: float,
    n_steps: int,
    bump: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, n_steps)

    smooth = S0 + (ST - S0) * x
    volatile = (
        S0
        + (ST - S0) * x
        + bump * np.sin(8 * np.pi * x)
        + 0.5 * bump * np.sin(17 * np.pi * x)
    )

    volatile[0] = S0
    volatile[-1] = ST
    return smooth, volatile
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    return yf.download(ticker, period=period, auto_adjust=True, progress=False)

# ==================================================
# Configuration Streamlit
# ==================================================
st.set_page_config(
    page_title="Pricer dynamique de portefeuille d'options",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.6rem;
        max-width: 1450px;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1c1f26, #12151b);
        border: 1px solid #2a2e39;
        border-radius: 12px;
        padding: 10px 12px;
    }
    button[data-baseweb="tab"] {
        height: 40px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.03);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255,255,255,0.10);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pricer dynamique de portefeuille d'options")
st.caption(
    "Black-Scholes comme point de départ, puis revalorisation dynamique, couverture delta et coûts de transaction."
)


# ==================================================
# Sidebar
# ==================================================
st.sidebar.header("Paramètres")

strategy = st.sidebar.selectbox(
    "Stratégie",
    ["Call long", "Put long", "Straddle"],
)

S0 = st.sidebar.number_input(
    "Prix initial du sous-jacent S0",
    min_value=10.0,
    max_value=1000.0,
    value=100.0,
    step=1.0,
)

T = st.sidebar.slider("Horizon / maturité (années)", 0.1, 2.0, 1.0, 0.05)
r = st.sidebar.slider("Taux sans risque r", 0.0, 0.10, 0.03, 0.005)
qty = st.sidebar.slider("Quantité", 1.0, 10.0, 1.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Trajectoire du sous-jacent")
mu = st.sidebar.slider("Dérive μ", -0.05, 0.15, 0.03, 0.01)
sigma_real = st.sidebar.slider("Volatilité réalisée", 0.05, 0.80, 0.20, 0.01)
n_steps = st.sidebar.slider("Nombre de pas", 20, 300, 120, 10)
seed = st.sidebar.number_input("Graine aléatoire", min_value=0, max_value=99999, value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("Repricing et friction")
sigma_iv = st.sidebar.slider("Volatilité implicite", 0.05, 0.90, 0.25, 0.01)
tx_cost_rate = st.sidebar.slider("Coût de transaction proportionnel", 0.0, 0.01, 0.001, 0.0001)


# ==================================================
# Calcul principal
# ==================================================
legs = build_strategy(strategy, S0, qty)

S_path = simulate_underlying_path(S0, mu, sigma_real, T, n_steps, int(seed))
base_df = run_dynamic_hedged_backtest(legs, S_path, sigma_iv, r, T, tx_cost_rate)

summary = {
    "initial_portfolio_value": float(base_df["option_value"].iloc[0]),
    "final_portfolio_value": float(base_df["option_value"].iloc[-1]),
    "realized_pnl": float(base_df["pnl"].iloc[-1]),
    "total_tx_costs": float(base_df["cum_tx_cost"].iloc[-1]),
    "realized_vol": realized_annual_vol(S_path, T),
    "sigma_iv": sigma_iv,
}


# ==================================================
# Onglets
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Idée du projet",
        "Simulation dynamique",
        "Risque de trajectoire",
        "Marché réel",
    ]
)


# ==================================================
# Onglet 1
# ==================================================
with tab1:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Pourquoi un prix Black-Scholes calculé une fois ne suffit pas")
        st.markdown(
            """
            - Black-Scholes donne une **valeur instantanée** d'une option.
            - En pratique, un trader gère un **portefeuille** qui évolue dans le temps.
            - Le portefeuille doit être **revalorisé en continu** car :
              - le sous-jacent bouge,
              - le temps restant diminue,
              - l'exposition du portefeuille change.
            - Le **delta** change donc la couverture doit être ajustée.
            - Chaque ajustement de couverture crée des **coûts de transaction**.
            """
        )

        st.latex(r"\text{coût}_t = c \times |\Delta h_t| \times S_t")
        st.latex(r"\text{Valeur couverte}_t = V_{\text{options},t} + h_tS_t + cash_t")

    with c2:
        st.subheader("Portefeuille étudié")
        st.dataframe(pd.DataFrame(legs), use_container_width=True, hide_index=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Vol réalisée", f"{summary['realized_vol']:.2%}")
        m2.metric("Vol implicite", f"{summary['sigma_iv']:.2%}")
        m3.metric("Tx cost", f"{tx_cost_rate:.3%}")

    st.info(
        "Idée centrale : le vrai sujet n'est pas juste le prix initial, mais la gestion dynamique du portefeuille dans le temps."
    )


# ==================================================
# Onglet 2
# ==================================================
with tab2:
    st.subheader("Revalorisation dynamique du portefeuille")

    a, b, c = st.columns(3)
    a.metric("Valeur initiale", f"{summary['initial_portfolio_value']:.2f}")
    b.metric("Valeur finale", f"{summary['final_portfolio_value']:.2f}")
    c.metric("PnL couverte finale", f"{summary['realized_pnl']:.2f}")

    d, e, f = st.columns(3)
    d.metric("Coûts cumulés", f"{summary['total_tx_costs']:.2f}")
    e.metric("Vol réalisée", f"{summary['realized_vol']:.2%}")
    f.metric("Vol implicite", f"{summary['sigma_iv']:.2%}")

    g1, g2 = st.columns(2)

    with g1:
        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                y=base_df["underlying"],
                mode="lines",
                name="Sous-jacent",
                line=dict(width=2),
            )
        )
        fig_s.update_layout(
            title="Trajectoire du sous-jacent",
            xaxis_title="Pas",
            yaxis_title="Prix",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with g2:
        fig_v = go.Figure()
        fig_v.add_trace(
            go.Scatter(
                y=base_df["option_value"],
                mode="lines",
                name="Valeur options",
                line=dict(width=2),
            )
        )
        fig_v.update_layout(
            title="Valeur du portefeuille d'options",
            xaxis_title="Pas",
            yaxis_title="Valeur",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_v, use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        fig_d = go.Figure()
        fig_d.add_trace(
            go.Scatter(
                y=base_df["portfolio_delta"],
                mode="lines",
                name="Delta",
                line=dict(width=2),
            )
        )
        fig_d.update_layout(
            title="Delta du portefeuille",
            xaxis_title="Pas",
            yaxis_title="Delta",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with g4:
        fig_h = go.Figure()
        fig_h.add_trace(
            go.Scatter(
                y=base_df["hedge_shares"],
                mode="lines",
                name="Couverture",
                line=dict(width=2),
            )
        )
        fig_h.update_layout(
            title="Nombre d'actions de couverture",
            xaxis_title="Pas",
            yaxis_title="Actions",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    g5, g6 = st.columns(2)

    with g5:
        fig_tc = go.Figure()
        fig_tc.add_trace(
            go.Scatter(
                y=base_df["cum_tx_cost"],
                mode="lines",
                name="Coûts cumulés",
                line=dict(width=2),
            )
        )
        fig_tc.update_layout(
            title="Coûts de transaction cumulés",
            xaxis_title="Pas",
            yaxis_title="Coût",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_tc, use_container_width=True)

    with g6:
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(
                y=base_df["pnl"],
                mode="lines",
                name="PnL",
                line=dict(width=2),
            )
        )
        fig_p.update_layout(
            title="Valeur du portefeuille couvert",
            xaxis_title="Pas",
            yaxis_title="Valeur",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("### Tableau de synthèse")

    summary_df = pd.DataFrame(
        {
            "Métrique": [
                "Valeur initiale du portefeuille",
                "Valeur finale du portefeuille",
                "PnL couverte finale",
                "Coûts de transaction cumulés",
                "Volatilité réalisée",
                "Volatilité implicite",
            ],
            "Valeur": [
                f"{summary['initial_portfolio_value']:.2f}",
                f"{summary['final_portfolio_value']:.2f}",
                f"{summary['realized_pnl']:.2f}",
                f"{summary['total_tx_costs']:.2f}",
                f"{summary['realized_vol']:.2%}",
                f"{summary['sigma_iv']:.2%}",
            ],
        }
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("### Tableau détaillé de la simulation")

    detailed_df = base_df.copy()
    detailed_df = detailed_df.rename(
        columns={
            "step": "Pas",
            "underlying": "Sous-jacent",
            "option_value": "Valeur options",
            "portfolio_delta": "Delta portefeuille",
            "hedge_shares": "Couverture",
            "cum_tx_cost": "Coûts cumulés",
            "pnl": "Valeur portefeuille couvert",
        }
    )

    st.dataframe(detailed_df, use_container_width=True, hide_index=True, height=300)
# ==================================================
# Onglet 3
# ==================================================
with tab3:
    st.subheader("Même prix final, chemin différent")

    ST = st.slider("Prix terminal commun", 0.7 * S0, 1.3 * S0, S0, 1.0)
    bump = st.slider("Amplitude de la trajectoire agitée", 1.0, 40.0, 12.0, 1.0)

    path_a, path_b = make_smooth_vs_volatile_paths(S0, ST, n_steps, bump)

    res_a = run_dynamic_hedged_backtest(legs, path_a, sigma_iv, r, T, tx_cost_rate)
    res_b = run_dynamic_hedged_backtest(legs, path_b, sigma_iv, r, T, tx_cost_rate)

    pnl_a = float(res_a["pnl"].iloc[-1])
    pnl_b = float(res_b["pnl"].iloc[-1])

    p1, p2, p3 = st.columns(3)
    p1.metric("PnL finale A", f"{pnl_a:.2f}")
    p2.metric("PnL finale B", f"{pnl_b:.2f}")
    p3.metric("Écart", f"{(pnl_b - pnl_a):.2f}")

    c1, c2 = st.columns(2)

    with c1:
        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(y=res_a["underlying"], name="Trajectoire lisse", line=dict(width=2)))
        fig_path.add_trace(go.Scatter(y=res_b["underlying"], name="Trajectoire agitée", line=dict(width=2)))
        fig_path.update_layout(
            title="Deux trajectoires avec même début et même fin",
            xaxis_title="Pas",
            yaxis_title="Prix",
            template="plotly_dark",
            height=330,
        )
        st.plotly_chart(fig_path, use_container_width=True)

    with c2:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(y=res_a["pnl"], name="PnL trajectoire A", line=dict(width=2)))
        fig_cmp.add_trace(go.Scatter(y=res_b["pnl"], name="PnL trajectoire B", line=dict(width=2)))
        fig_cmp.update_layout(
            title="Comparaison des PnL",
            xaxis_title="Pas",
            yaxis_title="Valeur",
            template="plotly_dark",
            height=330,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown(
        """
        **Interprétation**
        - Les deux trajectoires finissent au même prix.
        - Pourtant, la PnL finale n'est pas forcément la même.
        - Pourquoi ? Parce que la couverture se fait tout au long du chemin.
        - Donc la forme de la trajectoire influence les échanges réalisés et les coûts payés.
        """
    )
# ==================================================
# Onglet 4
# ==================================================
with tab4:
    st.subheader("Volatilité réalisée historique vs volatilité implicite")

    ticker_map = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Tesla": "TSLA",
        "ETF SPY": "SPY",
    }

    c1, c2, c3 = st.columns(3)
    stock_name = c1.selectbox("Ticker", list(ticker_map.keys()))
    period = c2.selectbox("Fenêtre historique", ["6mo", "1y", "2y"], index=1)
    implied_assumption = c3.slider("Volatilité implicite supposée", 0.05, 1.00, 0.30, 0.01)

    ticker = ticker_map[stock_name]
    raw = load_prices(ticker, period)

    if raw.empty:
        st.warning("Impossible de récupérer les données de marché pour ce ticker.")
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
        K_atm = S_last

        bs_price_hist, _ = bs_price_delta("call", S_last, K_atm, T_mini, r, hist_vol)
        bs_price_implied, _ = bs_price_delta("call", S_last, K_atm, T_mini, r, implied_assumption)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dernier prix", f"{S_last:.2f}")
        m2.metric("Volatilité réalisée", f"{hist_vol:.2%}")
        m3.metric("Volatilité implicite", f"{implied_assumption:.2%}")
        m4.metric("Écart de prime call ATM 30j", f"{(bs_price_implied - bs_price_hist):.2f}")

        p1, p2 = st.columns(2)

        with p1:
            fig_close = go.Figure()
            fig_close.add_trace(
                go.Scatter(
                    x=close.index,
                    y=close.values,
                    mode="lines",
                    name="Prix",
                    line=dict(width=2),
                )
            )
            fig_close.update_layout(
                title=f"Historique de prix - {ticker}",
                xaxis_title="Date",
                yaxis_title="Prix",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(fig_close, use_container_width=True)

        with p2:
            fig_ret = go.Figure()
            fig_ret.add_trace(
                go.Scatter(
                    x=ret.index,
                    y=ret.values,
                    mode="lines",
                    name="Rendements log",
                    line=dict(width=1.5),
                )
            )
            fig_ret.update_layout(
                title=f"Rendements logarithmiques - {ticker}",
                xaxis_title="Date",
                yaxis_title="Rendement log",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("### Comparaison Black-Scholes")
        t1, t2 = st.columns(2)
        t1.metric("Prix call ATM 30j avec vol réalisée", f"{bs_price_hist:.2f}")
        t2.metric("Prix call ATM 30j avec vol implicite", f"{bs_price_implied:.2f}")

        st.markdown(
            """
            **Interprétation**
            - La volatilité réalisée est calculée à partir des rendements historiques observés.
            - La volatilité implicite correspond à l'hypothèse utilisée pour pricer l'option.
            - Quand la volatilité implicite est supérieure à la volatilité réalisée, l'option est plus chère.
            - Cela met en évidence une prime de risque, des contraintes de liquidité et l'effet de l'offre/demande sur les options.
            """
        )

st.divider()
st.caption(
    "Projet simplifié : gestion dynamique d'un portefeuille d'options avec couverture delta et frictions de marché."
)
