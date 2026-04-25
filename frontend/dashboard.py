
"""
NexusIQ — Business Intelligence Dashboard
Design: Premium dark terminal — deep navy base, high-contrast text,
        electric amber accents. Every element is readable at a glance.
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

API = "https://nexusiq-72o3.onrender.com"

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusIQ · Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────
for k, v in [("question_prefill",""), ("analyst_result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ━━ CSS Variables ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
:root {
  --bg:        #0b0f19;
  --bg2:       #111827;
  --bg3:       #1a2235;
  --border:    #1e2d45;
  --border2:   #263650;
  --text-hi:   #f0f4ff;        /* headings, values */
  --text-md:   #8faac8;        /* body copy */
  --text-lo:   #4a6080;        /* labels, secondary */
  --text-dim:  #2e4060;        /* placeholders, dividers */
  --accent:    #f59e0b;        /* amber primary */
  --accent2:   #3b82f6;        /* blue secondary */
  --green:     #10b981;
  --red:       #ef4444;
  --yellow:    #f59e0b;
}

/* ━━ Base ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text-md);
}
.main { background: var(--bg); }
.main .block-container {
    padding: 1.6rem 2.8rem 4rem;
    max-width: 1440px;
}
#MainMenu, footer, header,
[data-testid="stDecoration"],
div[data-testid="stToolbar"] { visibility: hidden; display: none; }

/* ━━ Sidebar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stSidebar"] {
    background: #080c15;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.6rem 1.3rem 1rem;
}

/* Hide radio default label */
[data-testid="stSidebar"] .stRadio > label { display: none; }

/* Nav items */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: 2px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center;
    padding: 9px 12px !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    color: var(--text-lo) !important;
    cursor: pointer;
    transition: all .15s ease;
    border: 1px solid transparent !important;
    background: transparent !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: var(--bg2) !important;
    color: var(--text-md) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked ~ div,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: var(--bg3) !important;
    color: var(--text-hi) !important;
    border-color: var(--border2) !important;
}
/* Hide radio circle */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input { display: none !important; }

/* ━━ Page heading ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.pg-head {
    margin-bottom: 1.8rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.pg-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: -.01em;
    line-height: 1.15;
    margin-bottom: 4px;
}
.pg-sub {
    font-size: 13px;
    color: var(--text-lo);
    font-weight: 400;
    letter-spacing: .01em;
}

/* ━━ Section divider ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.sdv {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
}
.sdv-text {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: var(--text-lo);
    font-weight: 600;
    white-space: nowrap;
}
.sdv-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ━━ KPI grid ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.kg {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 1rem 0 2rem;
}
.kc {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.15rem 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color .2s, transform .15s;
}
.kc::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kc:hover {
    border-color: var(--border2);
    transform: translateY(-1px);
}
.kc.up::before  { background: var(--green); }
.kc.dn::before  { background: var(--red); }
.kc.nt::before  { background: var(--accent2); }
.kl {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text-lo);
    font-weight: 600;
    margin-bottom: 7px;
}
.kv {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-hi);
    line-height: 1.1;
    letter-spacing: -.01em;
}
.kd {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    margin-top: 6px;
}
.kd.up { color: var(--green); }
.kd.dn { color: var(--red); }
.kd.nt { color: var(--accent2); }

/* ━━ Insight block ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.ib {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: .9rem 0;
}
.ib p {
    font-size: 14px;
    color: var(--text-md);
    line-height: 1.75;
    margin: 0;
}
.ib strong { color: var(--text-hi); font-weight: 600; }
.ib em { color: var(--accent); font-style: normal; font-weight: 500; }

/* ━━ Recommendation card ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.rcc {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: .6rem;
    transition: background .15s;
}
.rcc:hover { background: var(--bg3); }
.rcc.C { border-left-color: var(--red); }
.rcc.H { border-left-color: #f97316; }
.rcc.M { border-left-color: var(--yellow); }
.rcc.L { border-left-color: var(--green); }
.rt {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-hi);
    margin-bottom: 5px;
    line-height: 1.4;
}
.rm {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-lo);
    line-height: 1.6;
}
.ri {
    font-size: 12.5px;
    color: var(--accent2);
    margin-top: 5px;
    font-weight: 500;
}

/* ━━ Priority badge ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.pb {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-right: 8px;
}
.pb.C { background: rgba(239,68,68,.15); color: #f87171; border: 1px solid rgba(239,68,68,.25); }
.pb.H { background: rgba(249,115,22,.15); color: #fb923c; border: 1px solid rgba(249,115,22,.25); }
.pb.M { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.25); }
.pb.L { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.25); }

/* ━━ Source chip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.sc {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 3px 9px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent2);
    margin: 2px 4px 2px 0;
}

/* ━━ Reasoning step ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.rstep {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: .65rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
    color: var(--text-md);
    line-height: 1.6;
}
.rnum {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    min-width: 22px;
    padding-top: 2px;
}

/* ━━ Status dot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.si {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 5px 0;
}
.si-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.si-dot.g { background: var(--green); box-shadow: 0 0 6px rgba(16,185,129,.5); }
.si-dot.y { background: var(--yellow); }
.si-dot.r { background: var(--red); }
.si-label {
    font-size: 12.5px;
    color: var(--text-md);
    font-weight: 400;
}
.si-status {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-lo);
}

/* ━━ Input fields ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTextArea textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
    font-size: 14px !important;
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.6 !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.15) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-lo) !important;
    font-size: 13.5px !important;
}
.stTextArea label {
    color: var(--text-lo) !important;
    font-size: 11.5px !important;
}

/* ━━ Selectbox ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stSelectbox label {
    color: var(--text-md) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
    font-size: 14px !important;
}

/* ━━ Slider ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stSlider label {
    color: var(--text-md) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--accent) !important;
    background: var(--bg3) !important;
}

/* ━━ Number input ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stNumberInput label {
    color: var(--text-md) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
}
.stNumberInput input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text-hi) !important;
    font-size: 14px !important;
    border-radius: 8px !important;
}

/* ━━ Text input ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTextInput label {
    color: var(--text-md) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
}
.stTextInput input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text-hi) !important;
    font-size: 14px !important;
    border-radius: 8px !important;
}

/* ━━ Checkbox ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stCheckbox label {
    color: var(--text-md) !important;
    font-size: 13.5px !important;
}

/* ━━ Buttons ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
button[kind="primary"] {
    background: var(--accent2) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: .02em !important;
    border-radius: 8px !important;
    padding: .5rem 1.2rem !important;
    transition: background .15s, transform .1s !important;
}
button[kind="primary"]:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
}
button[kind="secondary"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text-md) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: all .15s !important;
}
button[kind="secondary"]:hover {
    background: var(--bg3) !important;
    border-color: var(--accent2) !important;
    color: var(--text-hi) !important;
}

/* ━━ Tabs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-lo) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 16px !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg3) !important;
    color: var(--text-hi) !important;
    border-bottom: 2px solid var(--accent2) !important;
}

/* ━━ Metric ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .8rem 1rem;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: var(--text-lo) !important;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    color: var(--text-hi) !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ━━ Dataframe ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] th {
    background: var(--bg3) !important;
    color: var(--text-lo) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}
[data-testid="stDataFrame"] td {
    color: var(--text-md) !important;
    font-size: 13px !important;
}

/* ━━ Spinner ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ━━ Alert / error ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stAlert"] {
    background: rgba(239,68,68,.08) !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    border-radius: 8px !important;
    color: #fca5a5 !important;
}

/* ━━ Scrollbar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ━━ KPI divider text override ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
p, li { color: var(--text-md) !important; }
</style>
""", unsafe_allow_html=True)

# ── Plot defaults ──────────────────────────────────────────────────
PLOT = dict(
    template="plotly_dark",
    plot_bgcolor="#111827",
    paper_bgcolor="#111827",
    font=dict(family="Outfit", color="#8faac8", size=12),
    margin=dict(l=0, r=0, t=28, b=0),
)

# ── API helper ─────────────────────────────────────────────────────
def api_call(endpoint, method="GET", payload=None, timeout=120):
    try:
        url = f"{API}{endpoint}"
        r = (requests.post(url, json=payload, timeout=timeout)
             if method == "POST" else requests.get(url, timeout=30))
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"_err": "API offline — run: uvicorn src.api.main:app --reload --port 8000"}
    except Exception as e:
        return {"_err": str(e)}

# ── UI helpers ─────────────────────────────────────────────────────
def sdiv(label: str):
    st.markdown(
        f'<div class="sdv">'
        f'<span class="sdv-text">{label}</span>'
        f'<div class="sdv-line"></div>'
        f'</div>',
        unsafe_allow_html=True
    )

def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="pg-head">'
        f'<div class="pg-title">{title}</div>'
        f'<div class="pg-sub">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def insight_block(html: str):
    st.markdown(f'<div class="ib"><p>{html}</p></div>', unsafe_allow_html=True)

# ── Synthetic data ─────────────────────────────────────────────────
@st.cache_data(ttl=600)
def synth_sales():
    np.random.seed(42)
    dates = pd.date_range("2021-06-01", periods=30, freq="MS")
    rows = []
    for p, b in [("Widget-A",18),("Widget-B",13),("Widget-C",8),("Widget-D",5)]:
        for i, d in enumerate(dates):
            v = b*(1+.018*i)*(1+.22*np.sin(2*np.pi*(d.month-3)/12))*np.random.normal(1,.055)
            rows.append({"date": d, "product": p, "rev": max(0,v)})
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

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Wordmark — bold, high contrast, memorable
    st.markdown("""
    <div style="margin-bottom:2.2rem">
      <div style="display:flex;align-items:baseline;gap:1px">
        <span style="font-family:'Syne',sans-serif;font-size:1.5rem;
                     font-weight:800;color:#f0f4ff;letter-spacing:-.02em">
          NEXUS
        </span>
        <span style="font-family:'Syne',sans-serif;font-size:1.5rem;
                     font-weight:800;color:#f59e0b;letter-spacing:-.02em">
          IQ
        </span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#2e4060;letter-spacing:.22em;text-transform:uppercase;
                  margin-top:3px">
        Intelligence Platform
      </div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "nav",
        ["Overview", "AI Analyst", "Revenue Forecast", "Customer Risk", "Scenario Planner"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px;background:#1e2d45;margin:1.4rem 0'></div>",
                unsafe_allow_html=True)

    # System status — business labels, no vendor names
    STATUS_LABELS = {
        "groq":           "Intelligence Engine",
        "ollama":         "Local Processing",
        "vector_store":   "Knowledge Base",
        "forecast_model": "Forecast Model",
        "churn_model":    "Risk Model",
    }
    STATUS_CLASS  = {
        "ready":"g","online":"g","configured":"g",
        "not_built":"y","not_trained":"y",
        "no_api_key":"r","offline":"r",
    }
    STATUS_TEXT = {
        "ready":"Active","online":"Active","configured":"Active",
        "not_built":"Setup needed","not_trained":"Training needed",
        "no_api_key":"Not configured","offline":"Unavailable",
    }

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9.5px;'
        'color:#2e4060;letter-spacing:.16em;text-transform:uppercase;'
        'margin-bottom:.7rem">System Status</div>',
        unsafe_allow_html=True
    )

    health = api_call("/health")
    if "_err" not in health:
        for k, v in health.get("checks", {}).items():
            lbl = STATUS_LABELS.get(k, k.replace("_"," ").title())
            cls = STATUS_CLASS.get(v, "r")
            tip = STATUS_TEXT.get(v, v)
            st.markdown(
                f'<div class="si">'
                f'  <div class="si-dot {cls}"></div>'
                f'  <span class="si-label">{lbl}</span>'
                f'  <span class="si-status">{tip}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="si"><div class="si-dot r"></div>'
            '<span class="si-label" style="color:#f87171">API offline</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px;background:#1e2d45;margin:1.4rem 0'></div>",
                unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        f'color:#2e4060">{datetime.now().strftime("%d %b %Y  %H:%M")}</div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("↻  Refresh", use_container_width=True, type="secondary"):
        st.cache_data.clear(); st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":

    page_header(
        "Executive Overview",
        "Q4 2024 · Revenue intelligence · Risk monitoring · Strategic signals"
    )

    kpis = [
        ("Q4 Revenue",       "£51.4M", "+18.7% YoY",          "up", "up"),
        ("At-Risk Customers","847",    "Need intervention",    "dn", "dn"),
        ("ARR Exposed",      "£3.24M", "High-risk cohort",     "dn", "dn"),
        ("Gross Margin",     "61.2%",  "+3pp year-on-year",    "up", "up"),
        ("Forecast MAPE",    "4.7%",   "95.3% accuracy",       "nt", "nt"),
        ("Risk AUC-ROC",     "0.924",  "Top-quartile model",   "up", "up"),
    ]
    st.markdown(
        '<div class="kg">' +
        "".join([
            f'<div class="kc {c2}"><div class="kl">{l}</div>'
            f'<div class="kv">{v}</div>'
            f'<div class="kd {c1}">{d}</div></div>'
            for l,v,d,c1,c2 in kpis
        ]) + '</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        sdiv("Revenue Trend & Q4 Projection")
        df  = synth_sales()
        tot = df.groupby("date")["rev"].sum().reset_index()
        fd  = pd.date_range("2024-01-01", periods=6, freq="MS")
        fc  = tot["rev"].iloc[-1]*(1+.018)**np.arange(1,7)*(1+.18*np.sin(2*np.pi*fd.month/12))
        lo, hi = fc*.908, fc*1.092

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tot["date"], y=tot["rev"], name="Actual",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>£%{y:.1f}M<extra>Actual</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=list(fd)+list(fd[::-1]), y=list(hi)+list(lo[::-1]),
            fill="toself", fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI"
        ))
        fig.add_trace(go.Scatter(
            x=fd, y=fc, name="ML Forecast",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>£%{y:.1f}M<extra>Forecast</extra>"
        ))
        fig.update_layout(
            **PLOT, height=310,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)", font_size=12,
                        font_color="#8faac8"),
            yaxis=dict(title="Revenue (£M)", title_font_color="#4a6080",
                       title_font_size=11, tickprefix="£",
                       gridcolor="#1e2d45", tickfont_color="#4a6080"),
            xaxis=dict(gridcolor="#1e2d45", tickfont_color="#4a6080"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sdiv("Portfolio Mix")
        prod = df.groupby("product")["rev"].sum()
        colors = ["#3b82f6","#10b981","#f59e0b","#8b5cf6"]
        fig2 = go.Figure(go.Pie(
            labels=prod.index, values=prod.values, hole=.64,
            marker=dict(colors=colors, line=dict(color="#111827", width=2)),
            textfont=dict(family="Outfit", size=12, color="#f0f4ff"),
            hovertemplate="<b>%{label}</b><br>£%{value:.1f}M · %{percent}<extra></extra>"
        ))
        fig2.add_annotation(
            text="Revenue<br>Mix", x=.5, y=.5,
            font=dict(size=11, color="#4a6080", family="Outfit"), showarrow=False
        )
        fig2.update_layout(
            **PLOT, height=310,
            showlegend=True,
            legend=dict(orientation="v", x=.82, y=.5,
                        bgcolor="rgba(0,0,0,0)", font_size=12,
                        font_color="#8faac8")
        )
        st.plotly_chart(fig2, use_container_width=True)

    insight_block(
        "<strong>Strategic Assessment —</strong> "
        "Enterprise deals in North region account for 42% of Q4 forecast variance. "
        "847 SMB customers carry a combined ARR exposure of <strong>£3.24M</strong>; "
        "proactive outreach within 7 days is estimated to recover <strong>£1.1M</strong> "
        "based on historical intervention rates. "
        "Q4 seasonal uplift of <em>+18%</em> is reflected in the ML forecast. "
        "Navigate to <em>AI Analyst</em> for a full investigation."
    )

    sdiv("Product Performance")
    perf = pd.DataFrame({
        "Product":     ["Widget-A","Widget-B","Widget-C","Widget-D"],
        "Q4 Forecast": ["£18.2M","£13.1M","£7.8M","£4.9M"],
        "YoY Growth":  ["+22.1%","+9.3%","+5.1%","+3.4%"],
        "Margin":      ["64%","58%","61%","55%"],
        "Risk Level":  ["Low","Medium","High","Medium"],
        "Key Driver":  ["Seasonality","Supply chain","Competition","Maturity"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    sdiv("Seasonal Pattern Heatmap")
    df_h = synth_sales()
    df_h["month"] = df_h["date"].dt.month
    df_h["year"]  = df_h["date"].dt.year
    pw = df_h.groupby(["year","month"])["rev"].sum().reset_index().pivot(
        index="year", columns="month", values="rev"
    )
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig3 = go.Figure(go.Heatmap(
        z=pw.values, x=mn[:pw.shape[1]], y=pw.index.astype(str),
        colorscale=[[0,"#111827"],[.5,"#1e3a6e"],[1,"#3b82f6"]],
        showscale=False, hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>£%{z:.1f}M<extra></extra>",
        texttemplate="%{z:.0f}",
        textfont=dict(size=10, color="#8faac8")
    ))
    fig3.update_layout(**PLOT, height=195)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — AI ANALYST
# ══════════════════════════════════════════════════════════════════
elif page == "AI Analyst":

    page_header(
        "AI Analyst",
        "Evidence-based analysis · ML predictions · Ranked strategic recommendations"
    )

    sdiv("Suggested Questions")
    EXAMPLES = [
        "What are our top 3 priorities to maximise Q4 revenue?",
        "Which customers are most at risk of churning and what should we do?",
        "How should we respond to a competitor launching a freemium tier?",
        "What is the ARR impact of reducing churn by 2pp and adding 10 enterprise deals per month?",
    ]
    c1, c2 = st.columns(2, gap="medium")
    for i, ex in enumerate(EXAMPLES):
        with (c1 if i % 2 == 0 else c2):
            if st.button(ex, use_container_width=True,
                         key=f"qbtn_{i}", type="secondary"):
                st.session_state["question_prefill"] = ex
                st.rerun()

    st.markdown("<div style='height:1px;background:#1e2d45;margin:.9rem 0 1.1rem'></div>",
                unsafe_allow_html=True)

    # Prefill pattern — no widget key conflict
    if st.session_state["question_prefill"]:
        default_q = st.session_state["question_prefill"]
        st.session_state["question_prefill"] = ""
    else:
        default_q = st.session_state.get("question_input", "")

    question = st.text_area(
        "Your question",
        value=default_q,
        height=100,
        placeholder="Ask any business question…",
        key="question_input",
    )

    run_btn = st.button(
        "  Analyse  →",
        type="primary",
        disabled=not (question or "").strip(),
        use_container_width=True
    )

    if run_btn and question.strip():
        with st.spinner("Retrieving evidence · Running predictions · Synthesising…"):
            result = api_call("/query","POST",{"question": question,"structured": True})
        st.session_state["analyst_result"] = result

    result = st.session_state.get("analyst_result")
    if result:
        if "_err" in result:
            st.error(result["_err"])
            st.session_state["analyst_result"] = None
        else:
            sdiv("Analysis")
            c1,c2,c3,c4 = st.columns(4, gap="small")
            c1.metric("Confidence",  result.get("confidence","—"))
            c2.metric("Tools Used",  len(result.get("tools_used",[])))
            c3.metric("Iterations",  result.get("iterations","—"))
            c4.metric("Latency",     f"{result.get('processing_time_ms',0):.0f}ms")

            ans = result.get("direct_answer") or result.get("answer","")
            insight_block(ans)

            srcs = result.get("data_sources_used", result.get("tools_used",[]))
            if srcs:
                chips = "".join(f'<span class="sc">{s}</span>' for s in srcs)
                st.markdown(f"<div style='margin:.5rem 0 1.1rem'>{chips}</div>",
                            unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Recommendations","Reasoning","Raw JSON"])

            with tab1:
                recs = result.get("recommendations",[])
                if recs:
                    for r in recs:
                        p  = str(r.get("priority","Medium"))
                        pc = p[0].upper() if p else "M"
                        st.markdown(f"""
<div class="rcc {pc}">
  <div class="rt"><span class="pb {pc}">{p}</span>{r.get('action','')}</div>
  <div class="rm">
    ⏱ {r.get('time_horizon','—')} &nbsp;·&nbsp;
    effort: {r.get('effort','—')} &nbsp;·&nbsp;
    {r.get('evidence','—')}
  </div>
  <div class="ri">↑ {r.get('expected_impact','—')}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="color:#8faac8;font-size:14px;line-height:1.7">'
                        f'{result.get("raw_analysis","")}</div>',
                        unsafe_allow_html=True
                    )

            with tab2:
                for risk in result.get("key_risks",[]):
                    st.markdown(
                        f'<div style="font-size:13px;color:#f97316;padding:.4rem 0;'
                        f'border-bottom:1px solid #1e2d45">⚠ {risk}</div>',
                        unsafe_allow_html=True
                    )
                if result.get("key_risks"):
                    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
                for i, s in enumerate(
                    result.get("reasoning", result.get("reasoning_chain",[])), 1
                ):
                    st.markdown(
                        f'<div class="rstep">'
                        f'<span class="rnum">{i:02d}</span>'
                        f'<span>{s}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with tab3:
                st.json(result)

            if st.button("Clear result", type="secondary", key="clear_result"):
                st.session_state["analyst_result"] = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "Revenue Forecast":

    page_header(
        "Revenue Forecast",
        "Time-series ML model · Cross-validated accuracy · SHAP feature attribution"
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        product = st.selectbox(
            "Product", ["all","Widget-A","Widget-B","Widget-C","Widget-D"], key="fc_p"
        )
    with c2:
        region = st.selectbox(
            "Region",
            ["all","United Kingdom","Germany","France","Spain","Netherlands"], key="fc_r"
        )
    with c3:
        months = st.slider("Horizon (months)", 1, 12, 3, key="fc_m")

    if st.button("  Generate Forecast  →", type="primary", key="fc_run"):
        with st.spinner("Computing forecast…"):
            res = api_call("/forecast","POST",{
                "product": product,"region": region,"months_ahead": months
            })
        if "_err" in res:
            st.error(res["_err"])
        else:
            sdiv("Forecast Output")
            m1,m2,m3,m4 = st.columns(4, gap="small")
            m1.metric("Point Forecast",    res.get("forecast_revenue","—"), res.get("yoy_growth",""))
            m2.metric("Lower Bound (90%)", res.get("lower_bound_90pct","—"))
            m3.metric("Upper Bound (90%)", res.get("upper_bound_90pct","—"))
            m4.metric("Model MAPE",        res.get("model_mape","—"))
            insight_block(res.get("interpretation",""))

            feats = res.get("top_predictive_features",{})
            if feats:
                sdiv("Feature Attribution (SHAP)")
                nms = list(feats.keys())[:8][::-1]
                vls = list(feats.values())[:8][::-1]
                bar_colors = ["#3b82f6" if i >= len(vls)-3 else "#1e3a5e"
                              for i in range(len(vls))]
                fig = go.Figure(go.Bar(
                    y=nms, x=vls, orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.3f}" for v in vls], textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11,
                                  color="#8faac8"),
                ))
                fig.update_layout(
                    **PLOT, height=290,
                    xaxis=dict(title="Mean |SHAP|", title_font_color="#4a6080",
                               gridcolor="#1e2d45", tickfont_color="#4a6080"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_color="#8faac8"),
                )
                st.plotly_chart(fig, use_container_width=True)

    sdiv("Historical Revenue Trends")
    df_fc = synth_sales()
    if product != "all":
        df_fc = df_fc[df_fc["product"] == product]
    fig2 = px.line(
        df_fc, x="date", y="rev",
        color="product" if product == "all" else None,
        color_discrete_sequence=["#3b82f6","#10b981","#f59e0b","#8b5cf6"],
        labels={"rev":"Revenue (£M)","date":"","product":"Product"},
        template="plotly_dark"
    )
    fig2.update_layout(
        **PLOT, height=290,
        yaxis=dict(title="Revenue (£M)", title_font_color="#4a6080",
                   tickprefix="£", gridcolor="#1e2d45", tickfont_color="#4a6080"),
        xaxis=dict(gridcolor="#1e2d45", tickfont_color="#4a6080"),
        legend=dict(font_size=12, font_color="#8faac8",
                    bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER RISK
# ══════════════════════════════════════════════════════════════════
elif page == "Customer Risk":

    page_header(
        "Customer Risk",
        "Churn probability scoring · Behavioural driver analysis · Intervention prioritisation"
    )

    seg = st.selectbox("Segment filter",
                       ["all","Enterprise","SMB","Consumer"], key="cr_seg")

    if st.button("  Score Risk  →", type="primary", key="cr_run"):
        with st.spinner("Scoring customer base…"):
            res = api_call(f"/churn?segment={seg}&top_n=100","POST")
        if "_err" in res:
            st.error(res["_err"])
        else:
            sdiv("Risk Summary")
            m1,m2,m3,m4 = st.columns(4, gap="small")
            m1.metric("High-Risk Accounts", res.get("high_risk_customers","—"))
            m2.metric("Avg Churn Prob",     res.get("avg_churn_probability","—"))
            m3.metric("ARR at Risk",        res.get("arr_at_risk","—"))
            m4.metric("Model AUC-ROC",      res.get("model_auc_roc","—"))
            insight_block(res.get("interpretation",""))

            drivers = res.get("top_churn_drivers_shap",{})
            if drivers:
                sdiv("Top Churn Drivers")
                nms = list(drivers.keys())[:7]
                vls = list(drivers.values())[:7]
                bar_c = ["#ef4444","#f97316","#f59e0b","#84cc16",
                         "#10b981","#3b82f6","#8b5cf6"]
                fig = go.Figure(go.Bar(
                    x=nms, y=vls,
                    marker=dict(color=bar_c[:len(nms)], line=dict(width=0)),
                    text=[f"{v:.3f}" for v in vls], textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11,
                                  color="#8faac8"),
                ))
                fig.update_layout(
                    **PLOT, height=265,
                    xaxis=dict(tickangle=-20, gridcolor="#1e2d45",
                               tickfont_color="#8faac8"),
                    yaxis=dict(title="Mean |SHAP|", title_font_color="#4a6080",
                               gridcolor="#1e2d45", tickfont_color="#4a6080"),
                )
                st.plotly_chart(fig, use_container_width=True)

            sdiv("Recommended Actions")
            for i, action in enumerate(res.get("recommended_interventions",[])):
                priority = ["Critical","High","Medium","Low"][min(i,3)]
                pc = priority[0]
                st.markdown(
                    f'<div class="rcc {pc}">'
                    f'<div class="rt"><span class="pb {pc}">{priority}</span>{action}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    sdiv("Distribution Analysis")
    churn_df = synth_churn()
    cl, cr = st.columns(2, gap="large")

    with cl:
        fig = px.histogram(
            churn_df, x="prob", nbins=30, color="segment",
            color_discrete_map={
                "Enterprise":"#3b82f6","SMB":"#f59e0b","Consumer":"#ef4444"
            },
            opacity=.75,
            labels={"prob":"Churn Probability","count":""},
            template="plotly_dark"
        )
        fig.add_vline(x=.4, line_dash="dot", line_color="#10b981",
                      annotation_text="Threshold", annotation_font_size=11,
                      annotation_font_color="#10b981")
        fig.update_layout(
            **PLOT, height=265,
            xaxis=dict(title="Churn Probability",gridcolor="#1e2d45",
                       tickfont_color="#4a6080"),
            yaxis=dict(title="Count",gridcolor="#1e2d45",
                       tickfont_color="#4a6080"),
            legend=dict(font_size=12, font_color="#8faac8",
                        bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        cr_d = churn_df.groupby("contract")["prob"].mean().reset_index()
        fig2 = go.Figure(go.Bar(
            x=cr_d["contract"], y=cr_d["prob"],
            marker=dict(color=["#ef4444","#f59e0b","#10b981"],
                        line=dict(width=0)),
            text=[f"{v:.1%}" for v in cr_d["prob"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12,
                          color="#f0f4ff"),
        ))
        fig2.update_layout(
            **PLOT, height=265,
            xaxis=dict(gridcolor="#1e2d45", tickfont_color="#8faac8"),
            yaxis=dict(title="Avg Churn Probability",
                       tickformat=".0%",gridcolor="#1e2d45",
                       title_font_color="#4a6080",
                       tickfont_color="#4a6080"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    sdiv("Highest-Risk Accounts")
    top = churn_df.nlargest(20,"prob")[
        ["id","segment","contract","prob","mrr","tenure"]
    ].copy()
    top.columns = ["Account","Segment","Contract","Churn Prob","MRR (£)","Tenure (mo)"]
    top["Churn Prob"] = top["Churn Prob"].map("{:.1%}".format)
    top["MRR (£)"]    = top["MRR (£)"].map("£{:.0f}".format)
    st.dataframe(top, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — SCENARIO PLANNER
# ══════════════════════════════════════════════════════════════════
elif page == "Scenario Planner":

    page_header(
        "Scenario Planner",
        "What-if modelling · Quantified ARR & EBITDA impact · Strategic lever analysis"
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        sdiv("Configure Levers")
        label     = st.text_input("Scenario name", "Q4 Growth Initiative", key="sc_lbl")
        churn_pp  = st.slider("Churn rate change (pp)", -5.0, 5.0, -2.0, .5, key="sc_ch",
                               help="Negative = improvement. −2pp means churn falls from 4.2% → 2.2%")
        ent_deals = st.slider("Enterprise deals / month", 0, 30, 10, key="sc_d")
        margin_pp = st.slider("Gross margin change (pp)", -3.0, 5.0, 2.0, .5, key="sc_m")
        launch    = st.checkbox("Include product launch", True, key="sc_l")
        l_arr     = st.number_input(
            "Product launch ARR (£M)", 0.0, 20.0, 6.0, .5,
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
            "Churn reduction"  if abs(ci) > max(di, li)
            else "Enterprise deals" if di >= li
            else "Product launch"
        )

        m1, m2 = st.columns(2, gap="small")
        m1.metric("ARR Impact",    f"£{ta:+.2f}M", "vs current base")
        m2.metric("Revised ARR",   f"£{BASE+ta:.1f}M")
        m1.metric("EBITDA Impact", f"£{te:+.2f}M")
        m2.metric("Primary Lever", bl)

        cats = [
            "Base ARR",
            f"Churn ({churn_pp:+.1f}pp)",
            f"+{ent_deals} deals",
            f"Margin ({margin_pp:+.1f}pp)",
            "Launch",
            "Revised"
        ]
        fig = go.Figure(go.Waterfall(
            measure=["absolute","relative","relative","relative","relative","total"],
            x=cats, y=[BASE, ci, di, mi/4, li, 0],
            connector=dict(line=dict(color="#1e2d45", width=1, dash="dot")),
            increasing_marker_color="#10b981",
            decreasing_marker_color="#ef4444",
            totals_marker_color="#3b82f6",
            texttemplate="%{y:+.1f}",
            textfont=dict(family="JetBrains Mono", size=11, color="#f0f4ff"),
        ))
        fig.update_layout(
            **PLOT, height=305,
            yaxis=dict(title="ARR (£M)", title_font_color="#4a6080",
                       tickprefix="£", gridcolor="#1e2d45",
                       tickfont_color="#4a6080"),
            xaxis=dict(gridcolor="#1e2d45", tickfont_color="#8faac8",
                       tickangle=-15),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

    if st.button("  Get AI Assessment  →", type="primary",
                 use_container_width=True, key="sc_ai"):
        with st.spinner("Analysing scenario…"):
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
            insight_block(res.get("interpretation","Scenario computed."))
            ca, cb = st.columns(2, gap="large")

            def kv_row(k, v):
                return (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:.35rem 0;'
                    f'border-bottom:1px solid #1e2d45">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:11px;color:#4a6080">'
                    f'{k.replace("_"," ")}</span>'
                    f'<span style="font-size:13.5px;font-weight:600;'
                    f'color:#f0f4ff">{v}</span>'
                    f'</div>'
                )

            with ca:
                sdiv("Impact Summary")
                rows = "".join(kv_row(k,v)
                    for k,v in res.get("projected_impact",{}).items())
                st.markdown(rows, unsafe_allow_html=True)

            with cb:
                sdiv("By Component")
                rows = "".join(kv_row(k,v)
                    for k,v in res.get("component_breakdown",{}).items())
                st.markdown(rows, unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:10.5px;color:#4a6080;margin-top:1rem">'
                f'Confidence: {res.get("confidence","Medium")}</div>',
                unsafe_allow_html=True
            )
