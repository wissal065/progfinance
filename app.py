import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# ==================================================
# Fonctions utilitaires Black-Scholes
# ==================================================
def bs_d1(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_d2(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    return bs_d1(S, K, tau, r, sigma) - sigma * np.sqrt(tau)


def bs_price_delta(option_type: str, S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    """
    Retourne le prix Black-Scholes et le delta d'une option.
    Sens financier :
    - prix = valeur mark-to-model dans l'état de marché courant
    - delta = sensibilité locale au sous-jacent, utilisée pour le rééquilibrage de couverture
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
# Construction des stratégies et des simulations
# ==================================================
def build_strategy(strategy: str, S0: float, qty: float, width_pct: float) -> list[dict]:
    """Construit les jambes du portefeuille (quantité positive = position longue)."""
    width = S0 * width_pct

    if strategy == "Call long":
        return [{"type": "call", "K": S0, "qty": qty}]

    if strategy == "Put long":
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

    if strategy == "Call spread haussier":
        return [
            {"type": "call", "K": S0, "qty": qty},
            {"type": "call", "K": S0 + width, "qty": -qty},
        ]

    # Portefeuille personnalisé : version simple à 2 jambes
    return [
        {"type": "call", "K": S0, "qty": qty},
        {"type": "put", "K": S0, "qty": 0.0},
    ]


def simulate_underlying_path(S0: float, mu: float, sigma_real: float, T: float, n_steps: int, seed: int) -> np.ndarray:
    """Simule la trajectoire réalisée du marché (impacte la PnL de couverture et la volatilité réalisée)."""
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
    """Simule la volatilité implicite utilisée pour le repricing à chaque pas de temps."""
    rng = np.random.default_rng(seed)
    vols = np.zeros(n_steps)
    vols[0] = sigma0

    if mode == "Bruit stochastique":
        for t in range(1, n_steps):
            vols[t] = max(0.05, vols[t - 1] + rng.normal(0, vol_noise))
    else:
        # Changement de régime : bas niveau de vol -> haut niveau -> normalisation partielle
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
    Reprice le portefeuille complet à chaque étape et couvre dynamiquement le delta avec le sous-jacent.
    Modèle de friction de marché : coût = tx_cost_rate * |variation de couverture| * S
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

        # 1) Valorisation du portefeuille d'options et delta agrégé
        opt_val_t = 0.0
        opt_delta_t = 0.0
        for leg in legs:
            leg_price, leg_delta = bs_price_delta(leg["type"], S_t, leg["K"], tau, r, sigma_t)
            opt_val_t += leg["qty"] * leg_price
            opt_delta_t += leg["qty"] * leg_delta

        option_values[i] = opt_val_t
        portfolio_delta[i] = opt_delta_t

        # 2) Le compte cash rémunère au taux sans risque entre deux rééquilibrages
        if i > 0:
            cash *= (1.0 + r * dt)

        # 3) Rééquilibrer la couverture sur -delta (objectif delta-neutre)
        h_new = -opt_delta_t
        trade_shares = h_new - h_prev
        trade_notional = trade_shares * S_t
        tc = tx_cost_rate * abs(trade_notional)

        cash -= trade_notional
        cash -= tc
        tc_cum += tc

        hedge_shares[i] = h_new
        cumulative_tc[i] = tc_cum

        # 4) PnL du portefeuille couvert dans le temps
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
    """Deux trajectoires avec même prix initial/final, mais dynamiques très différentes."""
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
# Configuration de la page et style
# ==================================================
st.set_page_config(page_title="Pricer dynamique de portefeuille d'options", layout="wide", initial_sidebar_state="expanded")

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

st.title("Pricer dynamique de portefeuille d'options")
st.caption(
    "Pourquoi le pricing ponctuel ne suffit pas : repricing dynamique, volatilité implicite changeante, couverture delta et risque de trajectoire."
)


# ==================================================
# Contrôles de la barre latérale
# ==================================================
st.sidebar.header("Contrôles portefeuille + simulation")

strategy = st.sidebar.selectbox(
    "Stratégie",
    ["Call long", "Put long", "Straddle", "Strangle", "Call spread haussier", "Portefeuille personnalisé"],
)

S0 = st.sidebar.number_input("Prix initial du sous-jacent S0", min_value=10.0, max_value=1000.0, value=100.0, step=1.0)
T = st.sidebar.slider("Horizon / maturité (années)", 0.1, 2.0, 1.0, 0.05)
r = st.sidebar.slider("Taux sans risque r", 0.0, 0.10, 0.03, 0.005)
qty = st.sidebar.slider("Contrats par jambe", 1.0, 10.0, 1.0, 1.0)
width_pct = st.sidebar.slider("Écart de strike (% de S0) pour spreads/strangles", 0.05, 0.35, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Hypothèses de trajectoire")
mu = st.sidebar.slider("Dérive du sous-jacent μ", -0.05, 0.15, 0.03, 0.01)
sigma_real = st.sidebar.slider("Hypothèse de volatilité réalisée", 0.05, 0.80, 0.20, 0.01)
n_steps = st.sidebar.slider("Nombre de pas (repricing / couverture)", 20, 300, 120, 10)
seed = st.sidebar.number_input("Graine aléatoire", min_value=0, max_value=99999, value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("Dynamique de volatilité implicite")
iv_mode = st.sidebar.selectbox("Modèle de trajectoire de VI", ["Bruit stochastique", "Changement de régimes"])
sigma0 = st.sidebar.slider("Volatilité implicite initiale σ0", 0.05, 0.90, 0.25, 0.01)
vol_noise = st.sidebar.slider("Niveau de bruit de VI", 0.0, 0.08, 0.01, 0.001)
low_regime = st.sidebar.slider("Régime VI bas", 0.05, 0.60, 0.18, 0.01)
high_regime = st.sidebar.slider("Régime VI haut", 0.10, 1.00, 0.40, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Friction de couverture")
tx_cost_rate = st.sidebar.slider("Coût de transaction proportionnel", 0.0, 0.01, 0.001, 0.0001)


# ==================================================
# Construction du portefeuille (avec jambes personnalisées)
# ==================================================
legs = build_strategy(strategy, S0, qty, width_pct)
if strategy == "Portefeuille personnalisé":
    st.sidebar.markdown("Jambes du portefeuille personnalisé")
    c1, c2 = st.sidebar.columns(2)
    call_k = c1.number_input("Strike call", min_value=1.0, value=float(S0), step=1.0)
    call_q = c2.slider("Quantité call", -5.0, 5.0, 1.0, 1.0)

    c3, c4 = st.sidebar.columns(2)
    put_k = c3.number_input("Strike put", min_value=1.0, value=float(S0), step=1.0)
    put_q = c4.slider("Quantité put", -5.0, 5.0, 0.0, 1.0)

    legs = [
        {"type": "call", "K": call_k, "qty": call_q},
        {"type": "put", "K": put_k, "qty": put_q},
    ]


# ==================================================
# Données principales de simulation
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
# Onglets
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Théorie / intuition", "Simulateur dynamique de portefeuille", "Comparaison du risque de trajectoire", "Intuition marché réel"]
)

# --------------------------------------------------
# Onglet 1 - Théorie
# --------------------------------------------------
with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Pourquoi un pricing Black-Scholes statique est insuffisant")
        st.markdown(
            """
            - Les traders gèrent des **portefeuilles d'options**, pas des contrats isolés.
            - Le portefeuille doit être **repricé à chaque pas de temps** car :\
              prix du sous-jacent, temps restant et volatilité implicite évoluent.
            - Le delta n'est pas constant, donc la couverture doit être rééquilibrée dynamiquement.
            - Le rééquilibrage crée des **coûts de trading**, ce qui rend la PnL dépendante de la trajectoire.
            """
        )
        st.latex(r"\text{Transaction cost}_t = c \times |\Delta h_t| \times S_t")
        st.latex(r"\text{PnL}_t = V_{\text{options},t} + h_t S_t + \text{cash}_t")

    with c2:
        st.subheader("Jambes de stratégie actuelles")
        st.dataframe(pd.DataFrame(legs), use_container_width=True, hide_index=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Volatilité réalisée (trajectoire)", f"{summary['realized_vol']:.2%}")
        m2.metric("Volatilité implicite moyenne", f"{summary['avg_implied_vol']:.2%}")
        m3.metric("Taux de coût de transaction", f"{tx_cost_rate:.3%}")

    st.info(
        "En pratique, la volatilité implicite est souvent supérieure à la volatilité réalisée, car les vendeurs d'options exigent une prime pour le risque et les frictions de marché."
    )

# --------------------------------------------------
# Onglet 2 - Simulateur dynamique
# --------------------------------------------------
with tab2:
    st.subheader("Résultats du repricing dynamique et de la couverture delta")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Valeur initiale du portefeuille", f"{summary['initial_portfolio_value']:.2f}")
    r1c2.metric("Valeur finale du portefeuille", f"{summary['final_portfolio_value']:.2f}")
    r1c3.metric("PnL réalisée (couverte)", f"{summary['realized_pnl']:.2f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Coûts de transaction totaux", f"{summary['total_tx_costs']:.2f}")
    r2c2.metric("Volatilité réalisée", f"{summary['realized_vol']:.2%}")
    r2c3.metric("Volatilité implicite moyenne", f"{summary['avg_implied_vol']:.2%}")

    g1, g2 = st.columns(2)
    with g1:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(y=base_df["underlying"], mode="lines", name="Underlying", line=dict(width=2)))
        fig_s.update_layout(title="Trajectoire du sous-jacent", xaxis_title="Pas", yaxis_title="Prix", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_s, use_container_width=True)

    with g2:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(y=base_df["implied_vol"], mode="lines", name="Implied vol", line=dict(width=2)))
        fig_iv.update_layout(title="Trajectoire de volatilité implicite", xaxis_title="Pas", yaxis_title="Volatilité", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_iv, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(y=base_df["option_value"], mode="lines", name="Option portfolio", line=dict(width=2)))
        fig_v.update_layout(title="Valeur du portefeuille d'options", xaxis_title="Pas", yaxis_title="Valeur", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_v, use_container_width=True)

    with g4:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(y=base_df["portfolio_delta"], mode="lines", name="Portfolio delta", line=dict(width=2)))
        fig_d.update_layout(title="Delta du portefeuille", xaxis_title="Pas", yaxis_title="Delta", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_d, use_container_width=True)

    g5, g6 = st.columns(2)
    with g5:
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(y=base_df["cum_tx_cost"], mode="lines", name="Cumulative costs", line=dict(width=2)))
        fig_tc.update_layout(title="Coûts de transaction cumulés", xaxis_title="Pas", yaxis_title="Coût", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_tc, use_container_width=True)

    with g6:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(y=base_df["pnl"], mode="lines", name="PnL", line=dict(width=2)))
        fig_p.update_layout(title="PnL couverte au cours du temps", xaxis_title="Pas", yaxis_title="PnL", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Tableau de synthèse")
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Valeur initiale du portefeuille",
                "Valeur finale du portefeuille",
                "PnL réalisée",
                "Coûts de transaction totaux",
                "Volatilité réalisée",
                "Volatilité implicite moyenne",
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
# Onglet 3 - Comparaison du risque de trajectoire
# --------------------------------------------------
with tab3:
    st.subheader("Même prix terminal, trajectoire différente -> PnL différente")
    ST = st.slider("Prix terminal commun pour les trajectoires A et B", 0.7 * S0, 1.3 * S0, S0, 1.0)
    bump = st.slider("Intensité de volatilité pour la trajectoire B", 1.0, 40.0, 12.0, 1.0)

    path_a, path_b = make_smooth_vs_volatile_paths(S0, ST, n_steps, bump)
    iv_cmp = simulate_implied_vol_path(sigma0, n_steps, iv_mode, vol_noise, low_regime, high_regime, int(seed) + 7)

    res_a = run_dynamic_hedged_backtest(legs, path_a, iv_cmp, r, T, tx_cost_rate)
    res_b = run_dynamic_hedged_backtest(legs, path_b, iv_cmp, r, T, tx_cost_rate)

    cpa, cpb, cpd = st.columns(3)
    pnl_a = float(res_a["pnl"].iloc[-1])
    pnl_b = float(res_b["pnl"].iloc[-1])
    cpa.metric("PnL finale - Trajectoire A (lisse)", f"{pnl_a:.2f}")
    cpb.metric("PnL finale - Trajectoire B (volatile)", f"{pnl_b:.2f}")
    cpd.metric("Écart de PnL", f"{(pnl_b - pnl_a):.2f}")

    ga, gb = st.columns(2)
    with ga:
        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(y=res_a["underlying"], name="Trajectoire A (lisse)", line=dict(width=2)))
        fig_path.add_trace(go.Scatter(y=res_b["underlying"], name="Trajectoire B (volatile)", line=dict(width=2)))
        fig_path.update_layout(title="Trajectoires du sous-jacent (même début et fin)", xaxis_title="Pas", yaxis_title="Prix", template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_path, use_container_width=True)

    with gb:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(y=res_a["pnl"], name="PnL Trajectoire A", line=dict(width=2)))
        fig_cmp.add_trace(go.Scatter(y=res_b["pnl"], name="PnL Trajectoire B", line=dict(width=2)))
        fig_cmp.update_layout(title="Comparaison des trajectoires de PnL", xaxis_title="Pas", yaxis_title="PnL", template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown(
        """
        **Interprétation :**
        - Les deux trajectoires finissent au même prix, donc une vision terminale naïve suggère des résultats identiques.
        - En couverture dynamique, on traite à des prix intermédiaires, avec un coût à chaque rééquilibrage.
        - Donc la **forme de la trajectoire compte** : une trajectoire plus heurtée force souvent plus d'ajustements (ou plus gros) et plus de coûts.
        """
    )

# --------------------------------------------------
# Onglet 4 - Intuition marché réel
# --------------------------------------------------
with tab4:
    st.subheader("Volatilité réalisée historique vs volatilité implicite supposée")

    ticker_map = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Tesla": "TSLA",
        "ETF SPY": "SPY",
    }

    rc1, rc2, rc3 = st.columns(3)
    stock_name = rc1.selectbox("Ticker", list(ticker_map.keys()))
    period = rc2.selectbox("Fenêtre historique", ["6mo", "1y", "2y"], index=1)
    implied_assumption = rc3.slider("Hypothèse utilisateur de volatilité implicite", 0.05, 1.00, 0.30, 0.01)

    ticker = ticker_map[stock_name]
    raw = load_prices(ticker, period)

    if raw.empty:
        st.warning("Impossible de récupérer les données de marché pour ce ticker actuellement.")
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
        k1.metric("Dernier prix", f"{S_last:.2f}")
        k2.metric("Volatilité historique réalisée", f"{hist_vol:.2%}")
        k3.metric("Volatilité implicite supposée", f"{implied_assumption:.2%}")
        k4.metric("Écart de prime call ATM 30j", f"{(bs_price_implied - bs_price_hist):.2f}")

        h1, h2 = st.columns(2)
        with h1:
            f_close = go.Figure()
            f_close.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close", line=dict(width=2)))
            f_close.update_layout(title=f"Historique de prix {ticker}", xaxis_title="Date", yaxis_title="Prix", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(f_close, use_container_width=True)

        with h2:
            f_ret = go.Figure()
            f_ret.add_trace(go.Scatter(x=ret.index, y=ret.values, mode="lines", name="Log returns", line=dict(width=1.6)))
            f_ret.update_layout(title=f"Rendements logarithmiques {ticker}", xaxis_title="Date", yaxis_title="Rendement log", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(f_ret, use_container_width=True)

        st.markdown(
            """
            ### Pourquoi la volatilité implicite est souvent au-dessus de la volatilité réalisée
            1. **Prime de risque de marché** : les vendeurs d'options demandent une compensation pour le risque de krach / risque de queue.
            2. **Contraintes de liquidité** : porter le risque optionnel et l'impact de marché coûtent cher.
            3. **Déséquilibre offre/demande** : la demande persistante de protection fait monter les prix d'options.
            4. **Périodes de stress et pression de couverture** : en marché agité, les flux de hedging peuvent pousser la VI vers le haut.
            """
        )

st.divider()
st.caption("Dashboard pédagogique : repricing dynamique de portefeuille d'options sous volatilité implicite variable et frictions de trading.")
