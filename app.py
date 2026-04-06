import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime


# --------------------------------------------------
# Fonctions Black-Scholes
# --------------------------------------------------
def d1(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))


def put_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    return K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)) - S * norm.cdf(-d1(S, K, T, r, sigma))


# --------------------------------------------------
# Données réelles
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, period="1y"):
    return yf.download(ticker, period=period, auto_adjust=True, progress=False)


def compute_historical_volatility(data):
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    sigma_daily = returns.std()
    sigma_annual = sigma_daily * np.sqrt(252)
    return float(sigma_annual), returns


@st.cache_data(show_spinner=False)
def get_option_expirations(ticker):
    tk = yf.Ticker(ticker)
    return tk.options


@st.cache_data(show_spinner=False)
def get_option_chain(ticker, expiration):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiration)
    return chain.calls, chain.puts


def nearest_strike_row(df, strike):
    if df is None or df.empty or "strike" not in df.columns:
        return None
    idx = (df["strike"] - strike).abs().idxmin()
    return df.loc[idx]


def time_to_maturity(expiration_str):
    exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
    today = datetime.today().date()
    delta_days = (exp_date - today).days
    return max(delta_days / 365.0, 1 / 365.0)


# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------
st.set_page_config(
    page_title="Black-Scholes Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Style CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.8rem;
        padding-left: 1.8rem;
        padding-right: 1.8rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        letter-spacing: 0.2px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1c1f26, #12151b);
        border: 1px solid #2a2e39;
        padding: 10px 12px;
        border-radius: 12px;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        padding-left: 18px;
        padding-right: 18px;
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
    unsafe_allow_html=True
)

# --------------------------------------------------
# En-tête
# --------------------------------------------------
st.title("Black-Scholes Dashboard")
st.caption(
    "Pricing d’options européennes, analyse de sensibilité, données de marché réelles et comparaison théorie / marché."
)

# --------------------------------------------------
# Sidebar : paramètres du modèle
# --------------------------------------------------
st.sidebar.title("Paramètres")
st.sidebar.caption("Réglez les variables du modèle.")

S = st.sidebar.slider("Prix du sous-jacent (S)", 50.0, 300.0, 100.0)
K = st.sidebar.slider("Strike (K)", 50.0, 300.0, 100.0)
T = st.sidebar.slider("Maturité (T en années)", 0.1, 2.0, 1.0)
r = st.sidebar.slider("Taux sans risque (r)", 0.0, 0.10, 0.05)
sigma = st.sidebar.slider("Volatilité (σ)", 0.01, 1.0, 0.2)

call = call_price(S, K, T, r, sigma)
put = put_price(S, K, T, r, sigma)

# --------------------------------------------------
# Résumé haut de page
# --------------------------------------------------
top1, top2, top3, top4 = st.columns(4)
with top1:
    st.metric("Call", f"{call:.2f}")
with top2:
    st.metric("Put", f"{put:.2f}")
with top3:
    st.metric("Volatilité", f"{sigma:.2%}")
with top4:
    st.metric("Maturité", f"{T:.2f} an(s)")

# --------------------------------------------------
# Onglets
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Vue générale", "Sensibilité", "Cas réel"])

# ==================================================
# ONGLET 1
# ==================================================
with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Présentation")
        st.write(
            "Cette application calcule le prix théorique d’un call et d’un put européens "
            "avec le modèle de Black-Scholes."
        )
        st.write("Le modèle utilise les paramètres suivants :")
        st.write("- S : prix du sous-jacent")
        st.write("- K : strike")
        st.write("- T : maturité")
        st.write("- r : taux sans risque")
        st.write("- σ : volatilité")

        st.subheader("Interprétation")
        interpretation = []

        if sigma > 0.5:
            interpretation.append("Volatilité élevée : les options ont tendance à être plus chères.")
        elif sigma > 0.25:
            interpretation.append("Volatilité modérée : les options ont une valeur intermédiaire.")
        else:
            interpretation.append("Volatilité faible : les options ont tendance à être moins chères.")

        if T > 1:
            interpretation.append("Maturité longue : l’option a plus de temps pour devenir profitable.")
        else:
            interpretation.append("Maturité courte : la valeur temps de l’option est plus limitée.")

        if S > K:
            interpretation.append("Le call est dans la monnaie.")
        elif S == K:
            interpretation.append("Le call est à la monnaie.")
        else:
            interpretation.append("Le call est hors de la monnaie.")

        for line in interpretation:
            st.write(f"- {line}")

    with right:
        st.subheader("Paramètres actuels")
        param1, param2 = st.columns(2)

        with param1:
            st.metric("Sous-jacent S", f"{S:.2f}")
            st.metric("Maturité T", f"{T:.2f}")
            st.metric("Volatilité σ", f"{sigma:.2%}")

        with param2:
            st.metric("Strike K", f"{K:.2f}")
            st.metric("Taux r", f"{r:.2%}")

        st.subheader("Rappel du modèle")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.latex(r"C = S_0 N(d_1) - Ke^{-rT}N(d_2)")
        st.latex(r"P = Ke^{-rT}N(-d_2) - S_0N(-d_1)")

# ==================================================
# ONGLET 2
# ==================================================
with tab2:
    st.subheader("Analyse de sensibilité")

    g1, g2 = st.columns(2)

    with g1:
        S_values = np.linspace(50, 150, 100)
        call_values = [call_price(Si, K, T, r, sigma) for Si in S_values]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=S_values,
            y=call_values,
            mode="lines",
            name="Call",
            line=dict(width=3)
        ))
        fig1.update_layout(
            title="Call en fonction de S",
            xaxis_title="Sous-jacent (S)",
            yaxis_title="Prix du Call",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        sigma_values = np.linspace(0.01, 1.0, 100)
        call_sigma = [call_price(S, K, T, r, sig) for sig in sigma_values]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sigma_values,
            y=call_sigma,
            mode="lines",
            name="Call",
            line=dict(width=3)
        ))
        fig2.update_layout(
            title="Call en fonction de la volatilité",
            xaxis_title="Volatilité (σ)",
            yaxis_title="Prix du Call",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    T_values = np.linspace(0.01, 2.0, 100)
    put_T = [put_price(S, K, Ti, r, sigma) for Ti in T_values]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=T_values,
        y=put_T,
        mode="lines",
        name="Put",
        line=dict(width=3)
    ))
    fig3.update_layout(
        title="Put en fonction de la maturité",
        xaxis_title="Maturité (T)",
        yaxis_title="Prix du Put",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# ONGLET 3
# ==================================================
with tab3:
    st.subheader("Cas réel avec données de marché")

    ticker_map = {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "TotalEnergies": "TTE"
    }

    c1, c2, c3 = st.columns(3)

    with c1:
        stock_name = st.selectbox("Action", list(ticker_map.keys()))
    with c2:
        period = st.selectbox("Historique volatilité", ["6mo", "1y", "2y"], index=1)
    with c3:
        r_real = st.slider("Taux sans risque réel", 0.0, 0.10, 0.05)

    ticker = ticker_map[stock_name]
    data = get_stock_data(ticker, period=period)

    if data.empty:
        st.error("Impossible de récupérer les données du marché pour cette action.")
    else:
        # Nettoyage pour être sûr d'avoir des séries 1D propres
        data = data.copy()

        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_series = pd.to_numeric(close_series, errors="coerce").dropna()

        close_series.index = pd.to_datetime(close_series.index).tz_localize(None)

        returns = np.log(close_series / close_series.shift(1)).dropna()
        sigma_daily = returns.std()
        sigma_real = float(sigma_daily * np.sqrt(252))
        S_real = float(close_series.iloc[-1])

        row1, row2 = st.columns(2)

        with row1:
            K_real = st.number_input(
                "Strike réel (K)",
                min_value=1.0,
                value=round(S_real, 2),
                step=1.0
            )

        with row2:
            T_real = st.slider("Maturité théorique (T)", 0.1, 2.0, 1.0)

        call_real = call_price(S_real, K_real, T_real, r_real, sigma_real)
        put_real = put_price(S_real, K_real, T_real, r_real, sigma_real)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Ticker", ticker)
        with m2:
            st.metric("Dernier prix", f"{S_real:.2f}")
        with m3:
            st.metric("Volatilité hist.", f"{sigma_real:.2%}")
        with m4:
            st.metric("Strike", f"{K_real:.2f}")

        m5, m6 = st.columns(2)
        with m5:
            st.metric("Call théorique", f"{call_real:.2f}")
        with m6:
            st.metric("Put théorique", f"{put_real:.2f}")

        charts1, charts2 = st.columns(2)

        with charts1:
            fig4 = go.Figure()

            fig4.add_trace(go.Scatter(
                x=close_series.index,
                y=close_series.values,
                mode="lines",
                name="Prix",
                line=dict(width=2)
            ))

            fig4.update_layout(
                title=f"Cours de clôture - {stock_name}",
                xaxis_title="Date",
                yaxis_title="Prix",
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig4, use_container_width=True)

        with charts2:
            fig5 = go.Figure()

            fig5.add_trace(go.Scatter(
                x=returns.index,
                y=returns.values,
                mode="lines",
                name="Rendement",
                line=dict(width=2)
            ))

            fig5.update_layout(
                title=f"Rendements log - {stock_name}",
                xaxis_title="Date",
                yaxis_title="Rendement",
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Comparaison Black-Scholes vs réalité")

        expirations = get_option_expirations(ticker)

        if expirations is None or len(expirations) == 0:
            st.warning("Aucune échéance d’options disponible pour cette action.")
        else:
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                selected_expiry = st.selectbox("Échéance d’option disponible", expirations)

            with exp_col2:
                T_market = time_to_maturity(selected_expiry)
                st.metric("Maturité marché (années)", f"{T_market:.3f}")

            calls_df, puts_df = get_option_chain(ticker, selected_expiry)

            if calls_df.empty or puts_df.empty:
                st.warning("Impossible de récupérer la chaîne d’options pour cette échéance.")
            else:
                call_row = nearest_strike_row(calls_df, K_real)
                put_row = nearest_strike_row(puts_df, K_real)

                if call_row is None or put_row is None:
                    st.warning("Aucune option proche du strike sélectionné.")
                else:
                    strike_market = float(call_row["strike"])

                    bs_call_market = call_price(S_real, strike_market, T_market, r_real, sigma_real)
                    bs_put_market = put_price(S_real, strike_market, T_market, r_real, sigma_real)

                    market_call_price = float(call_row["lastPrice"]) if pd.notna(call_row["lastPrice"]) else np.nan
                    market_put_price = float(put_row["lastPrice"]) if pd.notna(put_row["lastPrice"]) else np.nan

                    cmp1, cmp2, cmp3 = st.columns(3)
                    with cmp1:
                        st.metric("Strike comparé", f"{strike_market:.2f}")
                    with cmp2:
                        st.metric("Call marché", f"{market_call_price:.2f}" if not np.isnan(market_call_price) else "N/A")
                    with cmp3:
                        st.metric("Put marché", f"{market_put_price:.2f}" if not np.isnan(market_put_price) else "N/A")

                    cmp4, cmp5 = st.columns(2)
                    with cmp4:
                        st.metric("Call Black-Scholes", f"{bs_call_market:.2f}")
                    with cmp5:
                        st.metric("Put Black-Scholes", f"{bs_put_market:.2f}")

                    diff_call = market_call_price - bs_call_market if not np.isnan(market_call_price) else np.nan
                    diff_put = market_put_price - bs_put_market if not np.isnan(market_put_price) else np.nan

                    cmp6, cmp7 = st.columns(2)
                    with cmp6:
                        st.metric(
                            "Écart Call (Marché - BS)",
                            f"{diff_call:.2f}" if not np.isnan(diff_call) else "N/A"
                        )
                    with cmp7:
                        st.metric(
                            "Écart Put (Marché - BS)",
                            f"{diff_put:.2f}" if not np.isnan(diff_put) else "N/A"
                        )

                    st.write(
                        "Cette comparaison montre la différence entre le prix théorique obtenu avec "
                        "Black-Scholes et le prix observé sur le marché pour une option de strike proche."
                    )
                    st.write(
                        "Ces écarts peuvent venir de la volatilité implicite, de l’offre et la demande, "
                        "des coûts de transaction, ou des hypothèses simplificatrices du modèle."
                    )

        st.subheader("Commentaire")
        if sigma_real > 0.4:
            st.write("Cette action est très volatile : les options sont relativement chères.")
        elif sigma_real > 0.25:
            st.write("Cette action a une volatilité modérée : les options ont une valeur intermédiaire.")
        else:
            st.write("Cette action a une volatilité plus faible : les options sont moins chères.")

        st.write(
            "Le prix affiché reste un prix théorique obtenu à partir du dernier cours "
            "et d’une volatilité historique estimée."
        )
# --------------------------------------------------
# Bas de page
# --------------------------------------------------
st.divider()
f1, f2 = st.columns(2)

with f1:
    st.subheader("Limites du modèle")
    st.write("- Volatilité supposée constante")
    st.write("- Taux sans risque supposé constant")
    st.write("- Modèle surtout adapté aux options européennes")
    st.write("- Pas de prise en compte des sauts de marché ni des coûts de transaction")
    st.write("- Le marché incorpore souvent une volatilité implicite différente de la volatilité historique")

with f2:
    st.subheader("Conclusion")
    st.write(
        "Cette application montre le calcul du prix d’une option européenne, "
        "l’effet des paramètres, des données de marché réelles, "
        "et une comparaison entre le modèle de Black-Scholes et les prix observés."
    )