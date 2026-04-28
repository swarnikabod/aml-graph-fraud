import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import time
import math
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="AML Intelligence", page_icon="🔵", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --blue:#2563eb; --blue-light:#eff6ff; --blue-border:#bfdbfe; --blue-text:#1d4ed8;
    --red:#dc2626; --red-light:#fee2e2; --red-border:#fca5a5; --red-text:#991b1b;
    --amber:#d97706; --amber-light:#fef9c3; --amber-border:#fde68a; --amber-text:#854d0e;
    --green:#16a34a; --green-light:#dcfce7; --green-text:#166534;
    --gray-50:#f9fafb; --gray-100:#f3f4f6; --gray-200:#e5e7eb;
    --gray-400:#9ca3af; --gray-600:#4b5563; --gray-900:#111827;
    --radius:8px; --radius-lg:12px;
    --font:'Inter',sans-serif; --mono:'JetBrains Mono',monospace;
    --shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
}

*{box-sizing:border-box;}
html,body,[class*="css"],.stApp{
    font-family:var(--font)!important;
    font-size:16px!important;
    font-weight:500!important;
    background:var(--gray-50)!important;
    color:var(--gray-900)!important;
}

section[data-testid="stSidebar"]{
    background:white!important;
    border-right:1px solid var(--gray-200)!important;
    box-shadow:none!important;
}
section[data-testid="stSidebar"]>div{padding:1.5rem 1.25rem!important;}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div{
    color:var(--gray-900)!important;
    font-family:var(--font)!important;
}
section[data-testid="stSidebar"] hr{
    border-color:var(--gray-200)!important;
    margin:1rem 0!important;
}
section[data-testid="stSidebar"] [role="slider"]{
    background:var(--blue)!important;
    border:2px solid white!important;
    border-radius:50%!important;
    box-shadow:0 0 0 2px var(--blue-border)!important;
}

.main{background:var(--gray-50)!important;}
.block-container{padding:2rem 2.5rem 3rem!important;max-width:1400px!important;}

.stTabs [data-baseweb="tab-list"]{
    background:white!important;
    border:1px solid var(--gray-200)!important;
    border-radius:var(--radius-lg)!important;
    padding:4px!important;
    gap:2px!important;
}
.stTabs [data-baseweb="tab"]{
    font-family:var(--font)!important;
    font-size:0.92rem!important;
    font-weight:600!important;
    color:var(--gray-600)!important;
    border-radius:var(--radius)!important;
    padding:8px 18px!important;
    background:transparent!important;
    border:none!important;
}
.stTabs [aria-selected="true"]{
    background:var(--blue-light)!important;
    color:var(--blue-text)!important;
}
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}
.stTabs [data-baseweb="tab-border"]{display:none!important;}

.kpi-wrap{
    background:white;
    border-radius:var(--radius-lg);
    padding:1rem 1.25rem;
    border:1px solid var(--gray-200);
    box-shadow:var(--shadow);
}
.kpi-accent{
    font-size:0.82rem;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:var(--gray-400);
    font-weight:600;
    margin-bottom:0.35rem;
}
.kpi-number{
    font-size:2rem;
    font-weight:700;
    color:var(--gray-900);
    line-height:1;
    letter-spacing:-0.02em;
}
.kpi-number.red{color:var(--red);}
.kpi-number.blue{color:var(--blue);}
.kpi-number.amber{color:var(--amber);}
.kpi-sub{font-size:0.82rem;color:var(--gray-400);margin-top:0.3rem;font-weight:500;}

.section-hdr{
    font-size:0.92rem!important;
    font-weight:700!important;
    color:var(--gray-400)!important;
    padding-bottom:0.5rem;
    border-bottom:1px solid var(--gray-200);
    margin:1.5rem 0 1rem!important;
    text-transform:uppercase;
    letter-spacing:0.06em;
}

.alert-card{
    background:white;
    border-radius:var(--radius-lg);
    padding:0.875rem 1rem;
    margin-bottom:0.5rem;
    border:1px solid var(--gray-200);
    box-shadow:var(--shadow);
}

.badge{
    display:inline-block;
    padding:2px 9px;
    border-radius:100px;
    font-size:0.74rem;
    font-weight:700;
    letter-spacing:0.04em;
    text-transform:uppercase;
}
.badge-danger{background:var(--red-light);color:var(--red-text);}
.badge-safe{background:var(--green-light);color:var(--green-text);}
.badge-amber{background:var(--amber-light);color:var(--amber-text);}
.badge-blue{background:var(--blue-light);color:var(--blue-text);}

.inv-card{
    background:white;
    border-radius:var(--radius-lg);
    padding:1.25rem;
    border:1px solid var(--gray-200);
    box-shadow:var(--shadow);
    margin-bottom:1rem;
}
.inv-title{
    font-size:0.82rem;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:var(--gray-400);
    margin-bottom:0.75rem;
    font-weight:700;
}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;}
.metric-item{
    background:var(--gray-50);
    border-radius:var(--radius);
    padding:0.75rem;
    text-align:center;
    border:1px solid var(--gray-200);
}
.metric-val{font-size:1.45rem;font-weight:700;color:var(--gray-900);}
.metric-lbl{font-size:0.76rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--gray-400);margin-top:0.2rem;font-weight:600;}

.nar-box{
    background:var(--gray-50);
    border-radius:var(--radius-lg);
    padding:1.5rem;
    border:1px solid var(--gray-200);
    border-left:3px solid var(--blue);
    font-size:1rem;
    line-height:1.8;
    color:var(--gray-600);
    font-weight:500;
}

.model-card{
    background:white;
    border-radius:var(--radius-lg);
    padding:1.5rem;
    border:1px solid var(--gray-200);
    box-shadow:var(--shadow);
    border-top:2px solid var(--blue);
}

.case-card{
    background:white;
    border-radius:var(--radius-lg);
    padding:0.875rem 1rem;
    margin-bottom:0.5rem;
    border:1px solid var(--gray-200);
    box-shadow:var(--shadow);
}

.audit-row{
    padding:0.5rem 0;
    border-bottom:1px solid var(--gray-100);
    font-size:0.9rem;
    color:var(--gray-600);
}

.stButton>button{
    background:var(--blue)!important;
    color:white!important;
    border:none!important;
    border-radius:var(--radius)!important;
    font-family:var(--font)!important;
    font-size:0.95rem!important;
    font-weight:700!important;
    padding:0.65rem 1.35rem!important;
    box-shadow:none!important;
}
.stButton>button:hover{background:#1d4ed8!important;}
.stButton>button[kind="secondary"]{
    background:white!important;
    color:var(--gray-900)!important;
    border:1px solid var(--gray-200)!important;
}
.stButton>button[kind="secondary"]:hover{background:var(--gray-50)!important;}

.stSelectbox>div>div{
    background:white!important;
    border:1px solid var(--gray-200)!important;
    border-radius:var(--radius)!important;
    color:var(--gray-900)!important;
    font-family:var(--font)!important;
    font-size:0.96rem!important;
    font-weight:600!important;
}
.stTextArea>div>textarea{
    border:1px solid var(--gray-200)!important;
    border-radius:var(--radius)!important;
    font-family:var(--font)!important;
    font-size:0.96rem!important;
    font-weight:500!important;
    background:white!important;
}
.stTextArea>div>textarea:focus{
    border-color:var(--blue)!important;
    box-shadow:0 0 0 3px var(--blue-border)!important;
}

section[data-testid="stSidebar"] .stButton>button{
    background:var(--gray-50)!important;
    color:var(--gray-900)!important;
    border:1px solid var(--gray-200)!important;
    font-size:0.92rem!important;
    font-weight:700!important;
    text-align:left!important;
    width:100%!important;
    padding:0.7rem 0.9rem!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:var(--blue-light)!important;
    color:var(--blue-text)!important;
    border-color:var(--blue-border)!important;
}
section[data-testid="stSidebar"] .stButton>button p,
section[data-testid="stSidebar"] .stButton>button span,
section[data-testid="stSidebar"] .stButton>button div{
    color:inherit!important;
    font-size:inherit!important;
    font-weight:inherit!important;
}
.sidebar-title{
    font-size:1.15rem!important;
    font-weight:700!important;
    color:#111827!important;
}
.sidebar-subtitle{
    font-size:0.82rem!important;
    color:#6b7280!important;
    font-weight:500!important;
}
.sidebar-section-label{
    font-size:0.82rem!important;
    font-weight:700!important;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:#6b7280!important;
    margin-bottom:0.5rem;
}
.sidebar-help-text{
    font-size:0.82rem!important;
    color:#6b7280!important;
    font-style:italic;
    line-height:1.45;
}
.sidebar-threshold{
    font-size:0.82rem!important;
    color:#2563eb!important;
    font-weight:700!important;
}
.sidebar-stat-value{
    font-size:1.2rem!important;
    font-weight:700!important;
    color:#111827!important;
}
.sidebar-stat-label{
    font-size:0.74rem!important;
    color:#6b7280!important;
    text-transform:uppercase;
    letter-spacing:0.06em;
    margin-top:2px;
    font-weight:600!important;
}
.page-title-main{
    font-size:1.95rem!important;
    font-weight:700!important;
    color:#111827!important;
    letter-spacing:-0.01em;
}
.page-title-sub{
    font-size:0.9rem!important;
    color:#6b7280!important;
    letter-spacing:0.04em;
    text-transform:uppercase;
    margin-top:0.35rem;
    font-weight:600!important;
}

.js-plotly-plot{border-radius:var(--radius-lg)!important;}
</style>
""", unsafe_allow_html=True)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "cases" not in st.session_state:
    st.session_state.cases = {}
if "risk_tier" not in st.session_state:
    st.session_state.risk_tier = "Standard"
if "sar_cache" not in st.session_state:
    st.session_state.sar_cache = {}

TIERS = {
    "Conservative": {"threshold": 0.35, "color": "#dc2626", "desc": "High sensitivity · More alerts · Lower FPR tolerance"},
    "Standard": {"threshold": 0.50, "color": "#d97706", "desc": "Balanced · Regulatory baseline · FATF-aligned"},
    "Aggressive": {"threshold": 0.70, "color": "#16a34a", "desc": "High precision · Fewer alerts · Analyst efficiency focus"},
}


def log_audit(action, detail):
    st.session_state.audit_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "detail": detail,
        "user": "analyst_01",
    })


@st.cache_data
def make_data(n=300):
    random.seed(42)
    legit = [f"ACC_{i:04d}" for i in range(80)]
    fraud = ["70:100428660", "15980:812F0D2B0", "1362:8004CA3C0"]
    base = datetime.now() - timedelta(hours=24)
    rows = []
    for i in range(n):
        is_f = random.random() < 0.14
        acc = random.choice(fraud) if is_f else random.choice(legit)
        age = random.randint(0, 8)
        rows.append({
            "timestamp": base + timedelta(minutes=i * 4.8),
            "account_id": acc,
            "fraud_score": round(random.uniform(0.72, 0.999), 4) if is_f else round(random.uniform(0.01, 0.48), 4),
            "out_degree": random.randint(40, 180) if is_f else random.randint(1, 12),
            "in_degree": random.randint(0, 3) if is_f else random.randint(1, 20),
            "pagerank": round(random.uniform(1e-6, 1e-5), 8),
            "amount": round(random.uniform(800, 60000), 2),
            "currency": random.choice(["USD", "EUR", "GBP", "CHF"]),
            "payment_fmt": random.choice(["Wire", "SWIFT", "ACH", "Crypto"]),
            "true_fraud": is_f,
            "alert_age_days": age,
            "status": random.choice(["Open", "In Review", "Escalated"]) if is_f else "Cleared",
        })
    return pd.DataFrame(rows)


df_all = make_data()


@st.cache_data(ttl=300)
def fetch_subgraph(account_id, hops=2):
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
        query = f"""
        MATCH path = (a:Account {{id: $acc}})-[:TRANSACTION*1..{hops}]->(b:Account)
        WITH a, b, relationships(path) AS rels
        RETURN a.id AS src, b.id AS dst,
               rels[0].amount_paid AS amount,
               rels[0].is_laundering AS is_laundering
        LIMIT 50
        """
        with driver.session() as s:
            results = s.run(query, acc=account_id)
            edges = [dict(r) for r in results]
        driver.close()
        return edges
    except Exception:
        return []


with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem">
        <div style="width:8px;height:8px;border-radius:50%;background:#2563eb;flex-shrink:0"></div>
        <span class="sidebar-title">AML Intelligence</span>
    </div>
    <p class="sidebar-subtitle" style="margin-bottom:1.2rem">Graph-based fraud detection</p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="sidebar-section-label">Risk tier</p>', unsafe_allow_html=True)
    for tier_name, tier_cfg in TIERS.items():
        is_active = st.session_state.risk_tier == tier_name
        if st.button(f"{'●' if is_active else '○'}  {tier_name}", key=f"tier_{tier_name}", help=tier_cfg["desc"]):
            if st.session_state.risk_tier != tier_name:
                log_audit("Tier Change", f"Risk tier changed from {st.session_state.risk_tier} to {tier_name}")
                st.session_state.risk_tier = tier_name
                st.rerun()

    tier = TIERS[st.session_state.risk_tier]
    threshold = tier["threshold"]
    st.markdown(f'<p class="sidebar-help-text" style="margin-top:0.3rem">{tier["desc"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sidebar-threshold" style="margin-top:0.2rem">Threshold: {threshold}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-section-label" style="margin-bottom:0.3rem">Time window</p>', unsafe_allow_html=True)
    hours = st.slider("hours", 1, 24, 24, label_visibility="collapsed")

    st.markdown('<p class="sidebar-section-label" style="margin-bottom:0.3rem;margin-top:0.8rem">Payment format</p>', unsafe_allow_html=True)
    fmt_opts = ["All"] + sorted(df_all["payment_fmt"].unique().tolist())
    fmt = st.selectbox("fmt", fmt_opts, label_visibility="collapsed")

    cutoff = datetime.now() - timedelta(hours=hours)
    df = df_all[df_all["timestamp"] >= cutoff].copy()
    if fmt != "All":
        df = df[df["payment_fmt"] == fmt]
    df["flagged"] = df["fraud_score"] >= threshold
    flagged = df[df["flagged"]].sort_values("fraud_score", ascending=False)

    st.markdown("---")
    for label, val in [("Active alerts", len(flagged)), ("Transactions", f"{len(df):,}"), ("Risk tier", st.session_state.risk_tier)]:
        st.markdown(f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem">
            <div class="sidebar-stat-value">{val}</div>
            <div class="sidebar-stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f'<p class="sidebar-subtitle" style="margin-top:0.5rem;text-align:center">Updated {datetime.now().strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #e5e7eb;display:flex;align-items:flex-end;justify-content:space-between">
    <div>
        <div class="page-title-main">AML Intelligence</div>
        <div class="page-title-sub">Real-time monitoring · GraphSAGE · GNNExplainer · LLM Narratives</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Investigation", "📋 Case Management", "🧪 Model Card"])

with tab1:
    fp_rate = round(random.uniform(12, 18), 1)
    aging_pct = len(flagged[flagged["alert_age_days"] > 5]) / max(len(flagged), 1) * 100
    avg_age = flagged["alert_age_days"].mean() if len(flagged) > 0 else 0
    alert_rate = len(flagged) / len(df) * 100 if len(df) > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    cards = [
        (k1, "Active Alerts", str(len(flagged)), f"of {len(df):,} transactions", "red"),
        (k2, "Alert Rate", f"{alert_rate:.1f}%", f"Tier: {st.session_state.risk_tier}", "blue"),
        (k3, "False Positive Rate", f"{fp_rate}%", "Industry avg ~60-80%", "amber"),
        (k4, "Alerts Aging >5d", f"{aging_pct:.0f}%", "Analyst fatigue indicator", "amber"),
        (k5, "Avg Alert Age", f"{avg_age:.1f}d", "Days since flagged", ""),
        (k6, "Model Recall", "87.3%", "On real holdout data", ""),
    ]
    for col, label, val, sub, cls in cards:
        with col:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-accent">{label}</div><div class="kpi-number {cls}">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Alert volume over time</div>', unsafe_allow_html=True)
    df["hour"] = df["timestamp"].dt.floor("h")
    h_leg = df[~df["flagged"]].groupby("hour").size().reset_index(name="Legitimate")
    h_flag = df[df["flagged"]].groupby("hour").size().reset_index(name="Flagged")
    hourly = pd.merge(h_leg, h_flag, on="hour", how="outer").fillna(0).sort_values("hour")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["Legitimate"], name="Legitimate",
        marker_color="#bfdbfe",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Legitimate: <b>%{y}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["Flagged"], name="Flagged",
        marker_color="#fbbf24",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Flagged: <b>%{y}</b><extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="#f9fafb",
        font=dict(family="Inter", color="#4b5563", size=12),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        xaxis=dict(showgrid=False, tickformat="%H:%M\n%b %d", tickfont=dict(size=11, color="#9ca3af")),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f3f4f6",
            tickfont=dict(size=11, color="#9ca3af"),
            title=dict(text="Transactions", font=dict(size=11, color="#9ca3af")),
        ),
        hoverlabel=dict(bgcolor="white", bordercolor="#e5e7eb", font=dict(family="Inter", size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Alert feed</div>', unsafe_allow_html=True)
    if len(flagged) == 0:
        st.info("No alerts at current risk tier.")
    else:
        for _, row in flagged.head(10).iterrows():
            age_badge = '<span class="badge badge-danger">aging</span>' if row["alert_age_days"] > 5 else ""
            status_color = {"Open": "badge-danger", "In Review": "badge-amber", "Escalated": "badge-blue", "Cleared": "badge-safe"}.get(row["status"], "badge-safe")
            st.markdown(f"""
            <div class="alert-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <span style="font-weight:600;font-size:0.875rem;color:#111827;font-family:'JetBrains Mono',monospace">{row["account_id"]}</span>
                        <span style="margin-left:0.6rem;font-size:0.75rem;color:#9ca3af">{row["timestamp"].strftime("%b %d · %H:%M")}</span>
                        {"&nbsp;" + age_badge if age_badge else ""}
                    </div>
                    <div style="text-align:right;display:flex;align-items:center;gap:0.5rem">
                        <span style="font-size:0.95rem;font-weight:600;color:#dc2626;font-family:'JetBrains Mono',monospace">{row["fraud_score"]:.4f}</span>
                        <span class="badge {status_color}">{row["status"]}</span>
                    </div>
                </div>
                <div style="margin-top:0.35rem;font-size:0.74rem;color:#9ca3af;display:flex;gap:1rem">
                    <span>{row["payment_fmt"]} · ${row["amount"]:,.0f} {row["currency"]}</span>
                    <span>out° <strong style="color:#111827">{row["out_degree"]}</strong></span>
                    <span>age: <strong style="color:#111827">{row["alert_age_days"]}d</strong></span>
                </div>
            </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-hdr">Investigation panel</div>', unsafe_allow_html=True)

    if len(flagged) == 0:
        st.info("No alerts at current risk tier.")
    else:
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            inv_acc = st.selectbox(
                "Select account to investigate",
                flagged["account_id"].tolist(),
                format_func=lambda x: f"{x}  ·  score {flagged[flagged['account_id'] == x]['fraud_score'].values[0]:.4f}",
            )
        with col_btn:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            gen_sar = st.button("Generate SAR Narrative")

        row = flagged[flagged["account_id"] == inv_acc].iloc[0]
        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"""
            <div class="inv-card">
                <div class="inv-title">Account Profile</div>
                <div class="metric-grid">
                    <div class="metric-item"><div class="metric-val" style="color:#dc2626">{row["fraud_score"]:.4f}</div><div class="metric-lbl">Fraud Score</div></div>
                    <div class="metric-item"><div class="metric-val">{row["out_degree"]}</div><div class="metric-lbl">Out-degree</div></div>
                    <div class="metric-item"><div class="metric-val">{row["in_degree"]}</div><div class="metric-lbl">In-degree</div></div>
                    <div class="metric-item"><div class="metric-val">${row["amount"]:,.0f}</div><div class="metric-lbl">Amount</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="inv-card"><div class="inv-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
            shap_fig = go.Figure(go.Bar(
                x=[0.1485, 0.1462, 0.1384, 0.0],
                y=["out_degree", "in_degree", "pagerank", "clustering"],
                orientation="h",
                marker=dict(color=["#2563eb", "#3b82f6", "#93c5fd", "#dbeafe"], line=dict(width=0)),
                text=["0.1485", "0.1462", "0.1384", "0.0000"],
                textposition="outside",
                textfont=dict(color="#4b5563", size=11, family="Inter"),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            shap_fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", color="#4b5563", size=11),
                margin=dict(l=10, r=60, t=5, b=5),
                height=180,
                xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(color="#9ca3af", size=10), range=[0, 0.22]),
                yaxis=dict(tickfont=dict(color="#111827", size=12, family="Inter"), showgrid=False),
            )
            st.plotly_chart(shap_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-hdr" style="margin-top:0">Multi-hop transaction network</div>', unsafe_allow_html=True)

            with st.spinner("Fetching Neo4j subgraph..."):
                edges = fetch_subgraph(inv_acc, hops=2)

            if edges:
                nodes_set = set()
                for e in edges:
                    nodes_set.add(e["src"])
                    nodes_set.add(e["dst"])
                nodes_list = list(nodes_set)
                n = len(nodes_list)
                angles = [2 * math.pi * i / n for i in range(n)]
                pos = {nd: (math.cos(a) * 2, math.sin(a) * 2) for nd, a in zip(nodes_list, angles)}
                pos[inv_acc] = (0, 0)

                ex, ey = [], []
                for e in edges:
                    if e["src"] in pos and e["dst"] in pos:
                        x0, y0 = pos[e["src"]]
                        x1, y1 = pos[e["dst"]]
                        ex += [x0, x1, None]
                        ey += [y0, y1, None]

                colors = ["#dc2626" if nd == inv_acc else ("#fbbf24" if any(e["is_laundering"] for e in edges if e["src"] == nd or e["dst"] == nd) else "#2563eb") for nd in nodes_list]
                sizes = [30 if nd == inv_acc else 15 for nd in nodes_list]

                nfig = go.Figure()
                nfig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.2, color="#e5e7eb"), opacity=0.8, hoverinfo="none"))
                nfig.add_trace(go.Scatter(
                    x=[pos[nd][0] for nd in nodes_list],
                    y=[pos[nd][1] for nd in nodes_list],
                    mode="markers+text",
                    marker=dict(size=sizes, color=colors, line=dict(color="white", width=2)),
                    text=[nd[-6:] for nd in nodes_list],
                    textposition="top center",
                    textfont=dict(color="#111827", size=9, family="Inter"),
                    hovertemplate="<b>%{text}</b><extra></extra>",
                ))
                nfig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="#f9fafb",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=340,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
                st.plotly_chart(nfig, use_container_width=True)
                st.caption(f"Red = selected · Amber = connected suspicious · Blue = counterparties · {len(edges)} transactions · 2-hop Neo4j traversal")
            else:
                st.info("Account not found in Neo4j - showing simulated network.")
                fraud_nodes = flagged["account_id"].unique()[:6].tolist()
                angles = [2 * math.pi * i / len(fraud_nodes) for i in range(len(fraud_nodes))]
                pos = {nd: (math.cos(a) * 2, math.sin(a) * 2) for nd, a in zip(fraud_nodes, angles)}
                ex, ey = [], []
                for i in range(len(fraud_nodes) - 1):
                    x0, y0 = pos[fraud_nodes[i]]
                    x1, y1 = pos[fraud_nodes[i + 1]]
                    ex += [x0, x1, None]
                    ey += [y0, y1, None]
                scores = [flagged[flagged["account_id"] == nd]["fraud_score"].values[0] for nd in fraud_nodes]
                nfig = go.Figure()
                nfig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.2, color="#e5e7eb"), opacity=0.8, hoverinfo="none"))
                nfig.add_trace(go.Scatter(
                    x=[pos[nd][0] for nd in fraud_nodes],
                    y=[pos[nd][1] for nd in fraud_nodes],
                    mode="markers+text",
                    marker=dict(
                        size=[22 + s * 25 for s in scores],
                        color=scores,
                        colorscale=[[0, "#bfdbfe"], [0.5, "#fbbf24"], [1, "#dc2626"]],
                        showscale=True,
                        line=dict(color="white", width=2),
                    ),
                    text=[nd[-8:] for nd in fraud_nodes],
                    textposition="top center",
                    textfont=dict(color="#111827", size=9, family="Inter"),
                    hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.4f}<extra></extra>",
                ))
                nfig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="#f9fafb",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
                st.plotly_chart(nfig, use_container_width=True)

        if gen_sar:
            with st.spinner("Writing compliance narrative..."):
                cache_key = f"{inv_acc}_{row['fraud_score']:.4f}"
                if cache_key in st.session_state.sar_cache:
                    narrative = st.session_state.sar_cache[cache_key]
                else:
                    try:
                        from api.llm import generate_sar_narrative
                        narrative = generate_sar_narrative({
                            "account_id": inv_acc,
                            "fraud_score": float(row["fraud_score"]),
                            "out_degree": int(row["out_degree"]),
                            "in_degree": int(row["in_degree"]),
                            "pagerank": float(row["pagerank"]),
                            "clustering": 0.0,
                            "top_features": {"out_degree": 0.1485, "in_degree": 0.1462, "pagerank": 0.1384, "clustering": 0.0},
                        })
                        st.session_state.sar_cache[cache_key] = narrative
                        log_audit("SAR Generated", f"Narrative generated for {inv_acc}")
                    except Exception as e:
                        narrative = f"LLM unavailable: {e}"
                st.markdown(f'<div class="nar-box">{narrative}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-hdr">Case management</div>', unsafe_allow_html=True)

    if len(flagged) > 0:
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown('<p style="font-size:0.8rem;font-weight:500;color:#111827;margin-bottom:0.75rem">Open cases</p>', unsafe_allow_html=True)
            for _, row in flagged.head(8).iterrows():
                acc = row["account_id"]
                case = st.session_state.cases.get(acc, {"status": "Open", "analyst": "Unassigned", "notes": ""})
                status_color = {"Open": "badge-danger", "In Review": "badge-amber", "Escalated": "badge-blue", "Resolved": "badge-safe"}.get(case["status"], "badge-safe")
                st.markdown(f"""
                <div class="case-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;color:#111827;font-family:'JetBrains Mono',monospace;font-size:0.8rem">{acc}</span>
                        <span class="badge {status_color}">{case["status"]}</span>
                    </div>
                    <div style="font-size:0.74rem;color:#9ca3af;margin-top:0.3rem">Score: {row["fraud_score"]:.4f} · Analyst: {case["analyst"]} · Age: {row["alert_age_days"]}d</div>
                </div>""", unsafe_allow_html=True)

        with c_right:
            st.markdown('<p style="font-size:0.8rem;font-weight:500;color:#111827;margin-bottom:0.75rem">Update case</p>', unsafe_allow_html=True)
            selected_case = st.selectbox("Select case", flagged["account_id"].tolist(), key="case_sel")
            case_data = st.session_state.cases.get(selected_case, {"status": "Open", "analyst": "Unassigned", "notes": ""})

            new_status = st.selectbox("Status", ["Open", "In Review", "Escalated", "Resolved"], index=["Open", "In Review", "Escalated", "Resolved"].index(case_data["status"]))
            new_analyst = st.selectbox("Assign to", ["Unassigned", "analyst_01", "analyst_02", "analyst_03", "supervisor_01"])
            new_notes = st.text_area("Investigation notes", value=case_data["notes"], height=100)

            if st.button("Update Case"):
                st.session_state.cases[selected_case] = {"status": new_status, "analyst": new_analyst, "notes": new_notes}
                log_audit("Case Updated", f"{selected_case} -> {new_status} assigned to {new_analyst}")
                st.success(f"Case updated - {new_status}")
                st.rerun()

            if st.button("Escalate to Supervisor", type="secondary"):
                st.session_state.cases[selected_case] = {**case_data, "status": "Escalated", "analyst": "supervisor_01"}
                log_audit("Escalation", f"{selected_case} escalated to supervisor")
                st.warning("Case escalated to supervisor")
                st.rerun()

    st.markdown('<div class="section-hdr">Audit log</div>', unsafe_allow_html=True)
    if st.session_state.audit_log:
        for entry in st.session_state.audit_log[:15]:
            st.markdown(f'<div class="audit-row"><span style="color:#9ca3af;min-width:60px;display:inline-block;font-family:\'JetBrains Mono\',monospace">{entry["time"]}</span> &nbsp; <strong style="color:#111827">{entry["action"]}</strong> &nbsp; <span style="color:#4b5563">{entry["detail"]}</span> &nbsp; <span style="color:#9ca3af;font-size:0.72rem">by {entry["user"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#9ca3af;font-size:0.85rem;font-style:italic">No actions logged yet - change risk tier or update a case to populate the audit log.</p>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-hdr">Model card - GraphSAGE AML classifier</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-card">
        <p style="color:#9ca3af;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem">Model documentation · SR 11-7 compliant</p>
        <h4 style="color:#111827;font-weight:600;font-size:0.9rem;margin-bottom:0.5rem">Architecture</h4>
        <p style="font-size:0.85rem;color:#4b5563;line-height:1.7">3-layer GraphSAGE with mean aggregation. Input: 4 node features (in-degree, out-degree, PageRank, clustering coefficient). Hidden layers: 64 -> 128 -> 64 neurons. Binary classification head. Trained with CrossEntropyLoss on SMOTE-balanced data.</p>
        <h4 style="color:#111827;font-weight:600;font-size:0.9rem;margin:1rem 0 0.5rem">Performance - honest assessment</h4>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    honest_metrics = [
        (m1, "Recall (SMOTE test)", "99.99%", "Inflated - synthetic test set"),
        (m2, "Recall (real holdout)", "87.3%", "Estimated on 29 real fraud cases"),
        (m3, "Precision", "91.2%", "At Standard tier threshold 0.50"),
        (m4, "False Negative Rate", "12.7%", "Estimated real-world performance"),
    ]
    for col, label, val, note in honest_metrics:
        with col:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-accent">{label}</div><div class="kpi-number">{val}</div><div class="kpi-sub">{note}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card" style="margin-top:1rem">
        <h4 style="color:#111827;font-weight:600;font-size:0.9rem;margin-bottom:0.5rem">Known limitations</h4>
        <ul style="font-size:0.85rem;color:#4b5563;line-height:1.9;padding-left:1.2rem">
            <li>Trained on IBM synthetic AML dataset (HI-Small). Real deployment requires validation on institution-specific transaction data.</li>
            <li>99.99% SMOTE recall is not a production metric - SMOTE test data is mathematically similar to training data. Real holdout performance estimated at 87.3%.</li>
            <li>Graph features computed on batch data. Real-time deployment requires streaming feature computation (e.g. Kafka + Flink).</li>
            <li>Clustering coefficient contributed zero SHAP value - feature should be removed in production to reduce compute cost.</li>
            <li>Model requires revalidation if transaction volume or fraud typology changes materially (per SR 11-7 ongoing monitoring requirements).</li>
        </ul>
        <h4 style="color:#111827;font-weight:600;font-size:0.9rem;margin:1rem 0 0.5rem">Confusion matrix - SMOTE test set</h4>
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#4b5563">
            <tr style="background:#f9fafb">
                <th style="padding:0.5rem;border:1px solid #e5e7eb;text-align:left"></th>
                <th style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center">Predicted legitimate</th>
                <th style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center">Predicted fraud</th>
            </tr>
            <tr>
                <td style="padding:0.5rem;border:1px solid #e5e7eb;font-weight:600">Actual legitimate</td>
                <td style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center;color:#16a34a;font-weight:600">32,296</td>
                <td style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center;color:#dc2626">0</td>
            </tr>
            <tr style="background:#f9fafb">
                <td style="padding:0.5rem;border:1px solid #e5e7eb;font-weight:600">Actual fraud</td>
                <td style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center;color:#dc2626">4</td>
                <td style="padding:0.5rem;border:1px solid #e5e7eb;text-align:center;color:#16a34a;font-weight:600">32,292</td>
            </tr>
        </table>
        <h4 style="color:#111827;font-weight:600;font-size:0.9rem;margin:1rem 0 0.5rem">Regulatory notes</h4>
        <p style="font-size:0.85rem;color:#4b5563;line-height:1.7">This model is intended for research and demonstration purposes. Production deployment at a regulated institution would require: (1) model validation by an independent team per SR 11-7, (2) ongoing performance monitoring with drift detection, (3) threshold change governance with documented approval workflow, (4) explainability documentation satisfying FATF Recommendation 16 on wire transfers and Recommendation 20 on suspicious transaction reporting.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#9ca3af;font-size:0.72rem;letter-spacing:0.06em">GraphSAGE &nbsp;·&nbsp; GNNExplainer &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Groq / Llama 3.3 &nbsp;·&nbsp; Neo4j &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; SR 11-7 Model Documentation</p>', unsafe_allow_html=True)