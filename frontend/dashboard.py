"""
NexusIQ — Business Intelligence Dashboard
Luxury-editorial dark theme. Refined, data-first, zero decorative noise.
UI/UX: Bloomberg Terminal × Financial Times × premium SaaS.
"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import streamlit as st

API = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusIQ · Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Fix for "widget key conflict" warning:
# Never pass value=st.session_state["key"] when key= is also set.
# Use a SEPARATE prefill key that the widget reads once then clears.
# ─────────────────────────────────────────────────────────────────────
if "question_prefill" not in st.session_state:
    st.session_state["question_prefill"] = ""
if "analyst_result" not in st.session_state:
    st.session_state["analyst_result"] = None

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=JetBrains+Mono:wght@300;400&display=swap');

/* ── Reset ─────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 1.4rem 2.8rem 4rem; max-width: 1440px; }
.main { background: #060810; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #03050a;
    border-right: 1px solid #0c1018;
    width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem; }
[data-testid="stSidebarContent"] { gap: 0; }

/* ── Nav radio — custom pill style ─────────────────────────────── */
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div {
    display: flex; flex-direction: column; gap: 2px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center;
    padding: 7px 12px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #3a4a60 !important;
    cursor: pointer;
    transition: all .15s;
    border: none !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #0c1018 !important;
    color: #7090b0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div {
    background: #0c1828 !important;
    color: #90b8e0 !important;
}
/* Hide radio circle */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input { display: none !important; }

/* ── Typography ─────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #c8d8e8 !important;
    font-weight: 300 !important;
    letter-spacing: .02em;
}

/* ── KPI grid ───────────────────────────────────────────────────── */
.kg { display: grid; grid-template-columns: repeat(6,1fr); gap: 8px; margin: .8rem 0 1.8rem; }
.kc {
    background: #09101a;
    border: 1px solid #0f1820;
    border-top: 2px solid;
    border-radius: 0 0 6px 6px;
    padding: 1rem 1.1rem .9rem;
    transition: border-color .2s, background .2s;
}
.kc:hover { background: #0b1220; }
.kc.up  { border-top-color: #2a7a50; }
.kc.dn  { border-top-color: #7a2a2a; }
.kc.nt  { border-top-color: #2a4a7a; }
.kl {
    font-size: 9.5px; text-transform: uppercase;
    letter-spacing: .12em; color: rgba(140, 170, 210, 0.5);
    font-weight: 500; margin-bottom: 6px;
}
.kv {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem; color: #c0d0e0;
    line-height: 1; font-weight: 300;
}
.kd { font-size: 10px; font-family: 'JetBrains Mono', monospace; margin-top: 5px; }
.kd.up { color: #2a9a60; } .kd.dn { color: #9a3030; } .kd.nt { color: #3060a0; }

/* ── Section divider ─────────────────────────────────────────────── */
.sdv {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1rem; padding-bottom: .5rem;
    border-bottom: 1px solid #0c1018;
}
.sdv span {
    font-size: 9px; text-transform: uppercase;
    letter-spacing: .18em; color: rgba(140, 170, 210, 0.5);
    font-weight: 500; white-space: nowrap;
}
.sdv::after {
    content: ''; flex: 1;
    height: 1px; background: #0c1018;
}

/* ── Insight block ───────────────────────────────────────────────── */
.ib {
    background: #07101a;
    border: 1px solid #0f1c2a;
    border-left: 2px solid #1a4a80;
    border-radius: 0 6px 6px 0;
    padding: .9rem 1.3rem;
    margin: .8rem 0;
}
.ib p { font-size: 13.5px; color: #5070a0; line-height: 1.8; margin: 0; }
.ib strong { color: #6090c0; font-weight: 500; }
.ib em { color: #4a7090; font-style: italic; }

/* ── Recommendation card ─────────────────────────────────────────── */
.rcc {
    border: 1px solid #0c1520;
    border-left: 2px solid;
    border-radius: 0 6px 6px 0;
    padding: .85rem 1.1rem;
    margin-bottom: .5rem;
    background: #07090f;
    transition: background .15s;
}
.rcc:hover { background: #090d18; }
.rcc.C { border-left-color: #8a2020; }
.rcc.H { border-left-color: #8a4a10; }
.rcc.M { border-left-color: #6a5a10; }
.rcc.L { border-left-color: #1a6a40; }
.rt { font-size: 13px; font-weight: 500; color: #a0b8d0; margin-bottom: 4px; }
.rm { font-size: 10.5px; color: #2a4060; font-family: 'JetBrains Mono', monospace; }
.ri { font-size: 11.5px; color: #3a6090; margin-top: 4px; }

/* ── Priority badge ──────────────────────────────────────────────── */
.pb {
    display: inline-block; padding: 1px 7px;
    border-radius: 2px; font-size: 9px; font-weight: 500;
    letter-spacing: .08em; margin-right: 7px;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
}
.pb.C { background: #200a0a; color: #a04040; border: 1px solid #3a1010; }
.pb.H { background: #1e1008; color: #a06030; border: 1px solid #3a2010; }
.pb.M { background: #1a1808; color: #907030; border: 1px solid #302808; }
.pb.L { background: #081a10; color: #308050; border: 1px solid #103020; }

/* ── Source chip ─────────────────────────────────────────────────── */
.sc {
    display: inline-block; background: #070d18;
    border: 1px solid #0f1c2c; border-radius: 3px;
    padding: 2px 8px; font-size: 10px;
    color: #304a70; font-family: 'JetBrains Mono', monospace;
    margin: 2px 3px 2px 0;
}

/* ── Reasoning step ──────────────────────────────────────────────── */
.rstep {
    display: flex; gap: 12px; align-items: flex-start;
    padding: .5rem 0; border-bottom: 1px solid #090d14;
    font-size: 12.5px; color: #3a5070;
}
.rnum {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #1a3050;
    min-width: 20px; margin-top: 1px;
}

/* ── Status indicator ────────────────────────────────────────────── */
.si { display: flex; align-items: center; gap: 7px; padding: 4px 0; }
.si-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.si-dot.g { background: #2a7a50; box-shadow: 0 0 4px #2a7a5060; }
.si-dot.y { background: #7a6020; }
.si-dot.r { background: #7a2020; }
.si-label { font-size: 11px; color: rgba(140, 170, 210, 0.5); }

/* ── Text input overrides ────────────────────────────────────────── */
.stTextArea textarea {
    background: #07090f !important;
    border: 1px solid #0f1820 !important;
    border-radius: 6px !important;
    color: #7090b0 !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #1a3a60 !important;
    box-shadow: 0 0 0 1px #1a3a6040 !important;
}
.stTextArea textarea::placeholder { color: #1e2e40 !important; }

/* ── Selectbox overrides ─────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #07090f !important;
    border: 1px solid #0f1820 !important;
    color: #7090b0 !important;
    border-radius: 6px !important;
}

/* ── Button overrides ────────────────────────────────────────────── */
button[kind="primary"] {
    background: #07101e !important;
    border: 1px solid #1a3a60 !important;
    color: #4a80c0 !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: .03em !important;
    border-radius: 4px !important;
    transition: all .15s !important;
}
button[kind="primary"]:hover {
    background: #0a1828 !important;
    border-color: #2a5090 !important;
    color: #6090d0 !important;
}
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #0f1820 !important;
    color: #2a4060 !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
button[kind="secondary"]:hover {
    border-color: #1a3050 !important;
    color: #4a6080 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #0c1018 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(140, 170, 210, 0.5) !important;
    font-size: 12px !important;
    padding: 6px 16px !important;
    border-bottom: 1px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #5080b0 !important;
    border-bottom: 1px solid #2a5090 !important;
}

/* ── Metric override ─────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #07090f;
    border: 1px solid #0c1018;
    border-radius: 6px;
    padding: .7rem .9rem;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    color: rgba(140, 170, 210, 0.5) !important;
    text-transform: uppercase;
    letter-spacing: .1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.5rem !important;
    color: #9ab0c8 !important;
    font-weight: 300 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #0c1018 !important;
    border-radius: 6px !important;
}

/* ── Slider ──────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #4a6080 !important;
}

/* ── Spinner ─────────────────────────────────────────────────────── */
.stSpinner > div { border-color: #1a3a60 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
PLOT = dict(
    template="plotly_dark",
    plot_bgcolor="#060810",
    paper_bgcolor="#060810",
    font=dict(family="DM Sans", color="rgba(140, 170, 210, 0.5)", size=11),
    margin=dict(l=0, r=0, t=24, b=0),
)

def api_call(endpoint, method="GET", payload=None, timeout=120):
    try:
        url = f"{API}{endpoint}"
        r = (requests.post(url, json=payload, timeout=timeout)
             if method == "POST" else requests.get(url, timeout=30))
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"_err": "API offline. Start: uvicorn src.api.main:app --reload --port 8000"}
    except Exception as e:
        return {"_err": str(e)}

def sdiv(label: str):
    """Section divider with label."""
    st.markdown(f'<div class="sdv"><span>{label}</span></div>', unsafe_allow_html=True)

def insight(text: str):
    st.markdown(f'<div class="ib"><p>{text}</p></div>', unsafe_allow_html=True)

@st.cache_data(ttl=600)
def synth_sales():
    np.random.seed(42)
    dates = pd.date_range("2021-06-01", periods=30, freq="MS")
    rows = []
    for p, b in [("Widget-A",18),("Widget-B",13),("Widget-C",8),("Widget-D",5)]:
        for i, d in enumerate(dates):
            v = b*(1+.018*i)*(1+.22*np.sin(2*np.pi*(d.month-3)/12))*np.random.normal(1,.055)
            rows.append({"date": d, "product": p, "rev": max(0, v)})
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def synth_churn():
    np.random.seed(42); n = 500
    return pd.DataFrame({
        "id":       [f"C{i:05d}" for i in range(n)],
        "prob":     np.clip(np.random.beta(2,4,n), .01, .99),
        "mrr":      np.round(np.random.uniform(18,120,n), 2),
        "tenure":   np.random.randint(0, 72, n),
        "segment":  np.random.choice(["Enterprise","SMB","Consumer"], n, p=[.15,.35,.5]),
        "contract": np.random.choice(["Month-to-month","One year","Two year"], n, p=[.55,.21,.24]),
    })

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — clean, no tech mentions
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Wordmark
    st.markdown("""
    <div style="margin-bottom:2rem">
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.35rem;
                    color:#3a4f68;letter-spacing:.04em;font-weight:300">
            Nexus<span style="color:#2a3f58">IQ</span>
        </div>
        <div style="font-size:8.5px;color:#16202c;letter-spacing:.22em;
                    text-transform:uppercase;margin-top:2px">
            Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "nav", ["Overview","AI Analyst","Revenue Forecast","Customer Risk","Scenario Planner"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px;background:#0b1018;margin:1.2rem 0'></div>",
                unsafe_allow_html=True)

    # System status — business labels only, no vendor names
    health = api_call("/health")
    checks = health.get("checks", {}) if "_err" not in health else {}

    STATUS_LABELS = {
        "groq":           "Intelligence Engine",
        "ollama":         "Local Processing",
        "vector_store":   "Knowledge Base",
        "forecast_model": "Forecast Model",
        "churn_model":    "Risk Model",
    }
    STATUS_CLASS = {
        "ready": "g", "online": "g", "configured": "g",
        "not_built": "y", "not_trained": "y",
        "no_api_key": "r", "offline": "r",
    }
    STATUS_LABEL_TEXT = {
        "ready": "Active", "online": "Active", "configured": "Active",
        "not_built": "Pending setup", "not_trained": "Pending training",
        "no_api_key": "Not configured", "offline": "Unavailable",
    }

    if checks:
        st.markdown("<div style='font-size:8.5px;color:#1a2a38;letter-spacing:.16em;"
                    "text-transform:uppercase;margin-bottom:.6rem'>System</div>",
                    unsafe_allow_html=True)
        for k, v in checks.items():
            lbl = STATUS_LABELS.get(k, k.replace("_"," ").title())
            cls = STATUS_CLASS.get(v, "r")
            tip = STATUS_LABEL_TEXT.get(v, v)
            st.markdown(
                f'<div class="si">'
                f'<div class="si-dot {cls}"></div>'
                f'<div class="si-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    elif "_err" not in health:
        pass
    else:
        st.markdown('<div class="si"><div class="si-dot r"></div>'
                    '<div class="si-label">API offline</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#0b1018;margin:1.2rem 0'></div>",
                unsafe_allow_html=True)

    # Timestamp + refresh
    st.markdown(
        f"<div style='font-size:9.5px;color:#1a2a38;font-family:\"JetBrains Mono\",monospace'>"
        f"{datetime.now().strftime('%d %b %Y  %H:%M')}</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    if st.button("Refresh", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div style="margin-bottom:1.6rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                  color:#9ab8d0;font-weight:300;letter-spacing:.01em">
        Executive Overview
      </div>
      <div style="font-size:11.5px;color:#1e2e3e;margin-top:5px;letter-spacing:.02em">
        Q4 2024 · Predictive intelligence · Risk monitoring · Strategic signals
      </div>
    </div>""", unsafe_allow_html=True)

    kpis = [
        ("Q4 Revenue",    "£51.4M",  "+18.7% YoY",         "up", "up"),
        ("At-Risk Customers", "847", "Intervention needed",  "dn", "dn"),
        ("ARR Exposed",   "£3.24M",  "High-risk cohort",     "dn", "dn"),
        ("Gross Margin",  "61.2%",   "+3pp year-on-year",    "up", "up"),
        ("Forecast MAPE", "4.7%",    "Model accuracy 95.3%", "up", "nt"),
        ("Risk AUC-ROC",  "0.924",   "Top-quartile accuracy","up", "up"),
    ]
    st.markdown(
        '<div class="kg">' +
        "".join([
            f'<div class="kc {c2}">'
            f'<div class="kl">{l}</div>'
            f'<div class="kv">{v}</div>'
            f'<div class="kd {c1}">{d}</div>'
            f'</div>'
            for l,v,d,c1,c2 in kpis
        ]) +
        '</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        sdiv("Revenue Trend & Q4 Projection")
        df  = synth_sales()
        tot = df.groupby("date")["rev"].sum().reset_index()
        fd  = pd.date_range("2024-01-01", periods=6, freq="MS")
        fc  = tot["rev"].iloc[-1] * (1+.018)**np.arange(1,7) * (1+.18*np.sin(2*np.pi*fd.month/12))
        lo, hi = fc * .908, fc * 1.092

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tot["date"], y=tot["rev"], name="Actual",
            line=dict(color="#1a4a80", width=1.8),
            hovertemplate="£%{y:.1f}M<extra>Actual</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=list(fd)+list(fd[::-1]), y=list(hi)+list(lo[::-1]),
            fill="toself", fillcolor="rgba(26,74,128,0.06)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=fd, y=fc, name="Forecast",
            line=dict(color="#4a80c0", width=1.5, dash="dot"),
            hovertemplate="£%{y:.1f}M<extra>Forecast</extra>"
        ))
        fig.update_layout(
            **PLOT, height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)", font_size=10, font_color="rgba(140, 170, 210, 0.5)"),
            yaxis=dict(title="£M", gridcolor="#0a1018", tickprefix="£",
                       title_font_size=10, tickfont_size=10),
            xaxis=dict(gridcolor="#0a1018", tickfont_size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sdiv("Revenue Mix")
        prod = df.groupby("product")["rev"].sum()
        fig2 = go.Figure(go.Pie(
            labels=prod.index, values=prod.values, hole=.68,
            marker=dict(
                colors=["#1a3a6a","#1a5a40","#4a2a10","#3a1a5a"],
                line=dict(color="#060810", width=2)
            ),
            textfont=dict(family="DM Sans", size=11),
            hovertemplate="<b>%{label}</b><br>£%{value:.1f}M · %{percent}<extra></extra>"
        ))
        fig2.add_annotation(
            text="Portfolio", x=.5, y=.5,
            font=dict(size=9, color="#1a2a38", family="DM Sans"),
            showarrow=False
        )
        fig2.update_layout(
            **PLOT, height=300,
            showlegend=True,
            legend=dict(orientation="v", x=.82, y=.5,
                        bgcolor="rgba(0,0,0,0)", font_size=10, font_color="rgba(140, 170, 210, 0.5)")
        )
        st.plotly_chart(fig2, use_container_width=True)

    insight(
        "<strong>Strategic Assessment —</strong> "
        "Enterprise deals in the North region account for 42% of Q4 forecast variance. "
        "847 SMB customers carry a combined ARR exposure of <strong>£3.24M</strong>; "
        "proactive outreach within 7 days is estimated to recover <strong>£1.1M</strong> "
        "based on historical intervention rates. "
        "Q4 seasonal uplift of +18% is reflected in the forecast. "
        "Use <em>AI Analyst</em> for detailed investigation or <em>Scenario Planner</em> "
        "to quantify strategic options."
    )

    sdiv("Product Performance")
    perf = pd.DataFrame({
        "Product":      ["Widget-A","Widget-B","Widget-C","Widget-D"],
        "Q4 Forecast":  ["£18.2M","£13.1M","£7.8M","£4.9M"],
        "YoY Growth":   ["+22.1%","+9.3%","+5.1%","+3.4%"],
        "Gross Margin": ["64%","58%","61%","55%"],
        "Risk Level":   ["Low","Medium","High","Medium"],
        "Key Driver":   ["Seasonality","Supply chain","Competition","Market maturity"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    sdiv("Seasonal Revenue Pattern")
    df_h = synth_sales()
    df_h["month"] = df_h["date"].dt.month
    df_h["year"]  = df_h["date"].dt.year
    pw = df_h.groupby(["year","month"])["rev"].sum().reset_index().pivot(
        index="year", columns="month", values="rev"
    )
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig3 = go.Figure(go.Heatmap(
        z=pw.values, x=mn[:pw.shape[1]], y=pw.index.astype(str),
        colorscale=[[0,"#06080e"],[.5,"#0d2040"],[1,"#1a4a80"]],
        showscale=False, hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>£%{z:.1f}M<extra></extra>",
        texttemplate="%{z:.0f}", textfont=dict(size=9, color="#2a4a70")
    ))
    fig3.update_layout(**PLOT, height=190)
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 2 — AI ANALYST
# ═════════════════════════════════════════════════════════════════════
elif page == "AI Analyst":

    st.markdown("""
    <div style="margin-bottom:1.6rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                  color:#9ab8d0;font-weight:300">AI Analyst</div>
      <div style="font-size:11.5px;color:#1e2e3e;margin-top:5px">
        Evidence-based analysis · ML predictions · Ranked recommendations
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Suggested questions ──────────────────────────────────────────
    sdiv("Suggested Questions")
    EXAMPLES = [
        "What are our top 3 priorities to maximise Q4 revenue?",
        "Which customers are most at risk of churning and what should we do?",
        "How should we respond to a competitor launching a freemium tier?",
        "What is the ARR impact of reducing churn by 2pp and adding 10 enterprise deals per month?",
    ]
    cols = st.columns(2, gap="medium")
    for i, ex in enumerate(EXAMPLES):
        with cols[i % 2]:
            # KEY FIX: button writes to "question_prefill", NOT "question_input"
            # The text_area widget owns "question_input" via its key= param
            if st.button(ex, use_container_width=True, key=f"qbtn_{i}",
                         type="secondary"):
                st.session_state["question_prefill"] = ex
                st.rerun()

    st.markdown("<div style='height:1px;background:#0b1018;margin:.8rem 0 1rem'></div>",
                unsafe_allow_html=True)

    # ── Input area ───────────────────────────────────────────────────
    # KEY FIX: widget key="question_input" is SEPARATE from prefill key.
    # We seed the value once from prefill, then clear the prefill.
    if st.session_state["question_prefill"]:
        default_q = st.session_state["question_prefill"]
        st.session_state["question_prefill"] = ""  # consume it
    else:
        default_q = st.session_state.get("question_input", "")

    question = st.text_area(
        label="question_label",
        value=default_q,
        height=90,
        placeholder="Ask any business question…",
        key="question_input",
        label_visibility="collapsed",
    )

    run_btn = st.button(
        "Analyse →",
        type="primary",
        disabled=not (question or "").strip(),
        use_container_width=True
    )

    if run_btn and question.strip():
        with st.spinner("Retrieving · Predicting · Synthesising"):
            result = api_call("/query", "POST", {"question": question, "structured": True})
        st.session_state["analyst_result"] = result

    # ── Results (persisted in session state so they survive reruns) ──
    result = st.session_state.get("analyst_result")
    if result:
        if "_err" in result:
            st.error(result["_err"])
            st.session_state["analyst_result"] = None
        else:
            sdiv("Response")
            c1, c2, c3, c4 = st.columns(4, gap="small")
            c1.metric("Confidence",   result.get("confidence","—"))
            c2.metric("Tools Used",   len(result.get("tools_used",[])))
            c3.metric("Iterations",   result.get("iterations","—"))
            c4.metric("Latency",      f"{result.get('processing_time_ms',0):.0f}ms")

            ans = result.get("direct_answer") or result.get("answer","")
            insight(ans)

            srcs = result.get("data_sources_used", result.get("tools_used",[]))
            if srcs:
                chips = "".join(f'<span class="sc">{s}</span>' for s in srcs)
                st.markdown(
                    f"<div style='margin:.4rem 0 1rem'>{chips}</div>",
                    unsafe_allow_html=True
                )

            tab1, tab2, tab3 = st.tabs(["Recommendations", "Reasoning", "Raw"])

            with tab1:
                recs = result.get("recommendations",[])
                if recs:
                    for r in recs:
                        p  = str(r.get("priority","Medium"))
                        pc = p[0].upper() if p else "M"
                        st.markdown(f"""
<div class="rcc {pc}">
  <div class="rt">
    <span class="pb {pc}">{p}</span>{r.get('action','')}
  </div>
  <div class="rm">
    ⏱ {r.get('time_horizon','—')}
    &nbsp;·&nbsp; effort: {r.get('effort','—')}
    &nbsp;·&nbsp; {r.get('evidence','—')}
  </div>
  <div class="ri">↑ {r.get('expected_impact','—')}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='color:#3a5070;font-size:13px'>"
                        f"{result.get('raw_analysis','')}</div>",
                        unsafe_allow_html=True
                    )

            with tab2:
                risks = result.get("key_risks",[])
                if risks:
                    for risk in risks:
                        st.markdown(
                            f"<div style='font-size:12px;color:#6a3020;"
                            f"padding:.3rem 0;border-bottom:1px solid #0c1018'>"
                            f"⚠ {risk}</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

                reasoning = result.get("reasoning", result.get("reasoning_chain",[]))
                for i, s in enumerate(reasoning, 1):
                    st.markdown(
                        f'<div class="rstep">'
                        f'<span class="rnum">{i:02d}</span>'
                        f'<span style="color:#3a5070">{s}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with tab3:
                st.json(result)

            if st.button("Clear", type="secondary"):
                st.session_state["analyst_result"] = None
                st.rerun()


# ═════════════════════════════════════════════════════════════════════
# PAGE 3 — REVENUE FORECAST
# ═════════════════════════════════════════════════════════════════════
elif page == "Revenue Forecast":

    st.markdown("""
    <div style="margin-bottom:1.6rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                  color:#9ab8d0;font-weight:300">Revenue Forecast</div>
      <div style="font-size:11.5px;color:#1e2e3e;margin-top:5px">
        Time-series model · Cross-validated accuracy · Feature attribution
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        product = st.selectbox(
            "Product", ["all","Widget-A","Widget-B","Widget-C","Widget-D"], key="fc_product"
        )
    with c2:
        region = st.selectbox(
            "Region", ["all","United Kingdom","Germany","France","Spain","Netherlands"], key="fc_region"
        )
    with c3:
        months = st.slider("Horizon (months)", 1, 12, 3, key="fc_months")

    if st.button("Generate Forecast →", type="primary", key="fc_run"):
        with st.spinner("Computing…"):
            res = api_call("/forecast","POST",{
                "product": product, "region": region, "months_ahead": months
            })
        if "_err" in res:
            st.error(res["_err"])
        else:
            sdiv("Forecast Output")
            m1, m2, m3, m4 = st.columns(4, gap="small")
            m1.metric("Point Forecast",    res.get("forecast_revenue","—"), res.get("yoy_growth",""))
            m2.metric("Lower Bound (90%)", res.get("lower_bound_90pct","—"))
            m3.metric("Upper Bound (90%)", res.get("upper_bound_90pct","—"))
            m4.metric("Model MAPE",        res.get("model_mape","—"))
            insight(res.get("interpretation",""))

            feats = res.get("top_predictive_features",{})
            if feats:
                sdiv("Feature Attribution (SHAP)")
                nms = list(feats.keys())[:8][::-1]
                vls = list(feats.values())[:8][::-1]
                colors = ["#1a4a80" if i >= len(vls)-3 else "#0c1828" for i in range(len(vls))]
                fig = go.Figure(go.Bar(
                    y=nms, x=vls, orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:.3f}" for v in vls], textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=10, color="#2a4060"),
                ))
                fig.update_layout(
                    **PLOT, height=280,
                    xaxis=dict(title="Mean |SHAP|", gridcolor="#0a1018", tickfont_size=10),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    sdiv("Historical Trends")
    df_fc = synth_sales()
    if product != "all":
        df_fc = df_fc[df_fc["product"] == product]
    fig2 = px.line(
        df_fc, x="date", y="rev",
        color="product" if product == "all" else None,
        color_discrete_sequence=["#1a4a80","#1a6040","#6a3a10","#4a2080"],
        labels={"rev":"Revenue (£M)","date":""},
        template="plotly_dark"
    )
    fig2.update_layout(
        **PLOT, height=280,
        yaxis=dict(gridcolor="#0a1018", tickprefix="£", tickfont_size=10),
        xaxis=dict(gridcolor="#0a1018", tickfont_size=10),
        legend=dict(font_size=10, font_color="rgba(140, 170, 210, 0.5)", bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER RISK
# ═════════════════════════════════════════════════════════════════════
elif page == "Customer Risk":

    st.markdown("""
    <div style="margin-bottom:1.6rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                  color:#9ab8d0;font-weight:300">Customer Risk</div>
      <div style="font-size:11.5px;color:#1e2e3e;margin-top:5px">
        Churn probability scoring · Behavioural drivers · Intervention prioritisation
      </div>
    </div>""", unsafe_allow_html=True)

    seg = st.selectbox("Segment", ["all","Enterprise","SMB","Consumer"], key="cr_seg")

    if st.button("Score Risk →", type="primary", key="cr_run"):
        with st.spinner("Scoring…"):
            res = api_call(f"/churn?segment={seg}&top_n=100", "POST")
        if "_err" in res:
            st.error(res["_err"])
        else:
            sdiv("Risk Summary")
            m1,m2,m3,m4 = st.columns(4, gap="small")
            m1.metric("High-Risk Accounts", res.get("high_risk_customers","—"))
            m2.metric("Avg Churn Prob",     res.get("avg_churn_probability","—"))
            m3.metric("ARR at Risk",        res.get("arr_at_risk","—"))
            m4.metric("Model AUC-ROC",      res.get("model_auc_roc","—"))
            insight(res.get("interpretation",""))

            drivers = res.get("top_churn_drivers_shap",{})
            if drivers:
                sdiv("Churn Drivers")
                nms = list(drivers.keys())[:7]
                vls = list(drivers.values())[:7]
                clrs = ["#5a1a1a","#4a2a10","#3a3010","#2a3a10","#1a3a20","#1a2a3a","#1a1a3a"]
                fig = go.Figure(go.Bar(
                    x=nms, y=vls,
                    marker=dict(color=clrs[:len(nms)], line=dict(width=0)),
                    text=[f"{v:.3f}" for v in vls], textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=10, color="#2a4060"),
                ))
                fig.update_layout(
                    **PLOT, height=260,
                    xaxis=dict(tickangle=-25, gridcolor="#0a1018", tickfont_size=10),
                    yaxis=dict(title="Mean |SHAP|", gridcolor="#0a1018", tickfont_size=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            sdiv("Recommended Actions")
            interventions = res.get("recommended_interventions",[])
            for i, action in enumerate(interventions):
                priority = ["Critical","High","Medium","Low"][min(i,3)]
                pc = priority[0]
                st.markdown(
                    f'<div class="rcc {pc}">'
                    f'<div class="rt"><span class="pb {pc}">{priority}</span>{action}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Always-visible section
    sdiv("Distribution Analysis")
    churn_df = synth_churn()
    cl, cr = st.columns(2, gap="large")

    with cl:
        fig = px.histogram(
            churn_df, x="prob", nbins=30, color="segment",
            color_discrete_map={
                "Enterprise":"#1a3a6a","SMB":"#4a2a10","Consumer":"#4a1a1a"
            },
            opacity=.75, labels={"prob":"Churn Probability","count":""},
            template="plotly_dark"
        )
        fig.add_vline(x=.4, line_dash="dot", line_color="#3a4a20",
                      annotation_text="Threshold", annotation_font_size=9,
                      annotation_font_color="#3a5020")
        fig.update_layout(
            **PLOT, height=260,
            xaxis=dict(gridcolor="#0a1018", tickfont_size=10),
            yaxis=dict(title="Count", gridcolor="#0a1018", tickfont_size=10),
            legend=dict(font_size=10, font_color="rgba(140, 170, 210, 0.5)", bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        cr_data = churn_df.groupby("contract")["prob"].mean().reset_index()
        fig2 = go.Figure(go.Bar(
            x=cr_data["contract"], y=cr_data["prob"],
            marker=dict(color=["#4a1a1a","#3a3010","#1a3a20"], line=dict(width=0)),
            text=[f"{v:.1%}" for v in cr_data["prob"]], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#3a5070"),
        ))
        fig2.update_layout(
            **PLOT, height=260,
            xaxis=dict(gridcolor="#0a1018", tickfont_size=10),
            yaxis=dict(title="Avg Churn Probability", tickformat=".0%",
                       gridcolor="#0a1018", tickfont_size=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    sdiv("Highest-Risk Accounts")
    top = churn_df.nlargest(20,"prob")[["id","segment","contract","prob","mrr","tenure"]].copy()
    top.columns = ["Account","Segment","Contract","Churn Prob","MRR (£)","Tenure (mo)"]
    top["Churn Prob"] = top["Churn Prob"].map("{:.1%}".format)
    top["MRR (£)"]    = top["MRR (£)"].map("£{:.0f}".format)
    st.dataframe(top, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO PLANNER
# ═════════════════════════════════════════════════════════════════════
elif page == "Scenario Planner":

    st.markdown("""
    <div style="margin-bottom:1.6rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.2rem;
                  color:#9ab8d0;font-weight:300">Scenario Planner</div>
      <div style="font-size:11.5px;color:#1e2e3e;margin-top:5px">
        What-if modelling · Quantified business impact · Lever analysis
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        sdiv("Parameters")
        label     = st.text_input("Scenario name", "Q4 Growth Initiative", key="sc_label")
        churn_pp  = st.slider("Churn rate change (pp)", -5.0, 5.0, -2.0, .5, key="sc_churn",
                               help="Negative = improvement")
        ent_deals = st.slider("Enterprise deals / month", 0, 30, 10, key="sc_deals")
        margin_pp = st.slider("Gross margin change (pp)", -3.0, 5.0, 2.0, .5, key="sc_margin")
        launch    = st.checkbox("New product launch", True, key="sc_launch")
        l_arr     = st.number_input(
            "Product ARR (£M)", 0.0, 20.0, 6.0, .5,
            disabled=not launch, key="sc_arr"
        )

    with col2:
        sdiv("Projected Impact")
        BASE = 42.3
        ci   = churn_pp * -0.84
        di   = ent_deals * 0.112
        mi   = margin_pp * 4.2
        li   = float(l_arr) if launch else 0.0
        ta   = ci + di + li
        te   = mi + ci * 0.6
        bl   = (
            "Churn reduction" if abs(ci) > max(di, li)
            else "Enterprise deals" if di >= li
            else "Product launch"
        )

        m1, m2 = st.columns(2, gap="small")
        m1.metric("ARR Impact",    f"£{ta:+.2f}M", "vs current base")
        m2.metric("Revised ARR",   f"£{BASE+ta:.1f}M")
        m1.metric("EBITDA Impact", f"£{te:+.2f}M")
        m2.metric("Primary Lever", bl)

        fig = go.Figure(go.Waterfall(
            measure=["absolute","relative","relative","relative","relative","total"],
            x=["Base ARR", f"Churn ({churn_pp:+}pp)",
               f"+{ent_deals} deals", f"Margin ({margin_pp:+}pp)",
               "Launch", "Revised ARR"],
            y=[BASE, ci, di, mi/4, li, 0],
            connector=dict(line=dict(color="#0c1018", width=1)),
            increasing_marker_color="#1a4a30",
            decreasing_marker_color="#4a1a1a",
            totals_marker_color="#1a3a6a",
            texttemplate="%{y:+.1f}",
            textfont=dict(family="JetBrains Mono", size=10, color="#3a5070"),
        ))
        fig.update_layout(
            **PLOT, height=300,
            yaxis=dict(title="ARR (£M)", gridcolor="#0a1018",
                       tickprefix="£", tickfont_size=10),
            xaxis=dict(gridcolor="#0a1018", tickfont_size=9, tickangle=-20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    if st.button("Get AI Assessment →", type="primary",
                 use_container_width=True, key="sc_ai"):
        with st.spinner("Analysing…"):
            res = api_call("/scenario","POST",{
                "description":             label,
                "churn_rate_change_pp":    churn_pp,
                "enterprise_deals_added":  ent_deals,
                "gross_margin_change_pp":  margin_pp,
                "product_launch":          launch,
                "product_launch_arr_m":    float(l_arr) if launch else 0.0,
            })
        if "_err" in res:
            st.error(res["_err"])
        else:
            insight(res.get("interpretation","Scenario modelled."))
            ca, cb = st.columns(2, gap="large")
            with ca:
                sdiv("Impact Summary")
                for k, v in res.get("projected_impact",{}).items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:.25rem 0;border-bottom:1px solid #0a1018;"
                        f"font-size:12px'>"
                        f"<span style='color:#1e3050;font-family:\"JetBrains Mono\",monospace'>"
                        f"{k.replace('_',' ')}</span>"
                        f"<span style='color:#5070a0'>{v}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            with cb:
                sdiv("By Component")
                for k, v in res.get("component_breakdown",{}).items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:.25rem 0;border-bottom:1px solid #0a1018;"
                        f"font-size:12px'>"
                        f"<span style='color:#1e3050;font-family:\"JetBrains Mono\",monospace'>"
                        f"{k.replace('_',' ')}</span>"
                        f"<span style='color:#5070a0'>{v}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            st.markdown(
                f"<div style='font-size:9.5px;color:#1a2a38;margin-top:.8rem;"
                f"font-family:\"JetBrains Mono\",monospace'>"
                f"Confidence: {res.get('confidence','Medium')}</div>",
                unsafe_allow_html=True
            )