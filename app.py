import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS — Arms & Escalation Geopolitical Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ── Hide sidebar completely ── */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { width: 0px !important; }

    /* ── Header bar ── */
    .header-bar {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1rem 1.8rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .header-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00c896, #0ea5e9, #d29922);
    }
    .header-left { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
    .header-bar h1 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #f0f6fc;
        font-family: 'Inter', sans-serif;
    }
    .header-bar .subtitle {
        font-size: 0.72rem;
        color: #8b949e;
        width: 100%;
        margin-top: -0.1rem;
    }

    /* ── Pills ── */
    .pill {
        font-size: 0.6rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 3px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    .pill-live { background: rgba(0,200,150,0.15); color: #00c896; }
    .pill-new { background: rgba(14,165,233,0.15); color: #0ea5e9; }
    .pill-critical { background: rgba(248,81,73,0.15); color: #f85149; }
    .pill-high { background: rgba(210,153,34,0.15); color: #d29922; }
    .pill-medium { background: rgba(0,200,150,0.15); color: #00c896; }

    /* ── KPI strip ── */
    .kpi-strip {
        display: flex;
        gap: 0;
        margin-bottom: 0.5rem;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        overflow: hidden;
    }
    .kpi-item {
        flex: 1;
        text-align: center;
        padding: 0.7rem 0.5rem;
        border-right: 1px solid #21262d;
    }
    .kpi-item:last-child { border-right: none; }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f0f6fc;
        line-height: 1;
    }
    .kpi-lbl {
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.15rem;
    }
    .kpi-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        margin-top: 0.1rem;
    }
    .kpi-sub.muted { color: #8b949e; }
    .kpi-sub.red { color: #f85149; }
    .kpi-sub.ylw { color: #d29922; }
    .kpi-sub.grn { color: #00c896; }

    /* ── Panel ── */
    .panel {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .panel-title {
        color: #f0f6fc;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .panel-title .pill { margin-left: auto; }

    /* ── Stat row inside panel ── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #21262d;
        font-size: 0.8rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-row .label { color: #8b949e; }
    .stat-row .value { color: #f0f6fc; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
    .stat-row .value.red { color: #f85149; }
    .stat-row .value.grn { color: #00c896; }

    /* ── Section divider ── */
    .sec-div {
        border-left: 3px solid #00c896;
        padding: 0.5rem 1rem;
        margin: 1.4rem 0 0.8rem 0;
    }
    .sec-div h3 {
        color: #f0f6fc;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .sec-div p { color: #8b949e; font-size: 0.82rem; margin: 0.15rem 0 0 0; }

    /* ── Insight callout ── */
    .callout {
        background: rgba(0,200,150,0.04);
        border: 1px solid rgba(0,200,150,0.12);
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        line-height: 1.6;
        color: #c9d1d9;
        margin: 0.4rem 0;
    }
    .callout strong { color: #58d5a8; }

    /* ── Alert card ── */
    .alert-card {
        border: 1px solid rgba(248,81,73,0.2);
        border-left: 3px solid #f85149;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.78rem;
        background: rgba(248,81,73,0.04);
    }
    .alert-card.warning {
        border-color: rgba(210,153,34,0.3);
        border-left-color: #d29922;
        background: rgba(210,153,34,0.04);
    }

    /* ── Deal card ── */
    .deal-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .deal-card:hover { border-color: #30363d; }

    /* ── Rx card ── */
    .rx-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .rx-card h4 {
        color: #f0f6fc;
        margin: 0 0 0.3rem 0;
        font-size: 0.82rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .rx-card p { color: #8b949e; margin: 0; font-size: 0.76rem; line-height: 1.55; }

    /* ── Model card (for predictive tab) ── */
    .model-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
    }
    .model-card .model-name { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }
    .model-card .model-auc {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .model-card .model-std { color: #8b949e; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }

    /* ── Dark table ── */
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    .dark-table th {
        background: #0d1117;
        color: #8b949e;
        padding: 0.5rem 0.6rem;
        text-align: left;
        border-bottom: 2px solid #21262d;
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .dark-table td {
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }
    .dark-table tr:hover td { background: rgba(0,200,150,0.03); }

    /* ── Tabs ── */
    div[data-testid="stTabs"] button {
        background: transparent !important;
        color: #8b949e !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #00c896 !important;
        border-bottom: 2px solid #00c896 !important;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="stExpander"] { border: 1px solid #21262d; border-radius: 6px; }
    header[data-testid="stHeader"] { background: #0d1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    [data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = pd.read_csv("arms_trade.csv")
    d['Escalation_Flag'] = d['Escalation_Risk'].map({'High': 2, 'Medium': 1, 'Low': 0})
    d['High_Risk_Flag'] = (d['Escalation_Risk'] == 'High').astype(int)
    d['Offensive_Flag'] = (d['Weapon_Class'] == 'Offensive').astype(int)
    d['YearGroup'] = pd.cut(d['Year'], bins=[2004,2009,2014,2019,2025],
                            labels=['2005-09','2010-14','2015-19','2020-24'])
    d['DealSize'] = pd.cut(d['Deal_Value_USD_M'], bins=[0,20,100,500,5000],
                           labels=['Small (<$20M)','Medium ($20-100M)','Large ($100-500M)','Mega (>$500M)'])
    return d

df = load_data()

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
RISK_COLORS = {'High': '#f85149', 'Medium': '#d29922', 'Low': '#00c896'}
CLASS_COLORS = {'Offensive': '#f85149', 'Defensive': '#0ea5e9'}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, -apple-system, sans-serif', color='#c9d1d9', size=11),
    margin=dict(l=40, r=60, t=40, b=35),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#c9d1d9')),
    title_font=dict(size=13, color='#f0f6fc', family='Inter, sans-serif'),
)

def styled_chart(fig, height=400):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor='rgba(201,209,217,0.06)', zerolinecolor='rgba(201,209,217,0.06)')
    fig.update_yaxes(gridcolor='rgba(201,209,217,0.06)', zerolinecolor='rgba(201,209,217,0.06)')
    return fig

# ─────────────────────────────────────────────────────────────
# UTILITY: Dark-themed HTML table
# ─────────────────────────────────────────────────────────────
def render_dark_table(dataframe, max_rows=15):
    """Render a DataFrame as a dark-themed HTML table."""
    html = "<table class='dark-table'><thead><tr>"
    html += "<th></th>" if dataframe.index.name else ""
    for col in dataframe.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for idx, row in dataframe.head(max_rows).iterrows():
        html += f"<tr><td style='color:#8b949e;'>{idx}</td>" if dataframe.index.name or not isinstance(idx, int) else "<tr>"
        for val in row:
            if isinstance(val, float):
                html += f"<td>{val:.2f}</td>" if abs(val) < 100 else f"<td>{val:,.0f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
@st.cache_data
def detect_anomalies(data):
    alerts = []

    # 1. YoY acceleration spikes (>50% increase)
    yearly_by_imp = data.groupby(['Importer', 'Year']).size().unstack(fill_value=0)
    for imp in yearly_by_imp.index:
        series = yearly_by_imp.loc[imp]
        for yr in series.index[1:]:
            prev = series.get(yr - 1, 0)
            curr = series.get(yr, 0)
            if prev >= 3 and curr > prev * 1.5:
                pct = ((curr - prev) / prev * 100)
                alerts.append({
                    'type': 'ACCEL SPIKE', 'severity': 'critical',
                    'text': f"{imp}: {pct:.0f}% YoY surge in {yr} ({prev}→{curr} deals)",
                    'context': 'descriptive'
                })

    # 2. Embargo-flagged deals
    embargo = data[data['UN_Embargo'] == 'Yes']
    if len(embargo) > 0:
        for _, row in embargo.nlargest(3, 'Deal_Value_USD_M').iterrows():
            alerts.append({
                'type': 'EMBARGO FLAG', 'severity': 'critical',
                'text': f"{row['Exporter']}→{row['Importer']}: ${row['Deal_Value_USD_M']}M {row['Weapon_Subtype']} ({row['Year']})",
                'context': 'diagnostic'
            })

    # 3. Outsized deals (>95th percentile)
    p95 = data['Deal_Value_USD_M'].quantile(0.95)
    mega = data[data['Deal_Value_USD_M'] > p95].nlargest(3, 'Deal_Value_USD_M')
    for _, row in mega.iterrows():
        alerts.append({
            'type': 'MEGA DEAL', 'severity': 'high',
            'text': f"${row['Deal_Value_USD_M']}M {row['Weapon_Subtype']}: {row['Exporter']}→{row['Importer']} ({row['Year']})",
            'context': 'descriptive'
        })

    # 4. Unusual corridors
    corridors = data.groupby(['Exporter', 'Importer']).agg(
        Count=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum')).reset_index()
    unusual = corridors[(corridors['Count'] <= 3) & (corridors['Value'] > p95)]
    for _, row in unusual.head(3).iterrows():
        alerts.append({
            'type': 'UNUSUAL CORRIDOR', 'severity': 'high',
            'text': f"{row['Exporter']}→{row['Importer']}: only {row['Count']} deals but ${row['Value']:,.0f}M total",
            'context': 'diagnostic'
        })

    return alerts

def render_alerts(alerts, context_filter, max_show=3):
    filtered = [a for a in alerts if a['context'] == context_filter][:max_show]
    if not filtered:
        return
    for a in filtered:
        sev_cls = '' if a['severity'] == 'critical' else ' warning'
        st.markdown(f"""
        <div class='alert-card{sev_cls}'>
            <span class='pill {"pill-critical" if a["severity"]=="critical" else "pill-high"}'>{a['type']}</span>
            <span style='color:#c9d1d9; margin-left:0.5rem;'>{a['text']}</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# HEADER BAR
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class='header-bar'>
    <div class='header-left'>
        <h1>AEGIS</h1><span class='pill pill-live'>LIVE</span>
        <div class='subtitle'>Arms & Escalation Geopolitical Intelligence System</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# GLOBAL FILTER BAR (replaces sidebar)
# ═════════════════════════════════════════════════════════════
gf1, gf2, gf3 = st.columns([3, 2, 5])
with gf1:
    year_range = st.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()),
                           (int(df['Year'].min()), int(df['Year'].max())), key='global_year')
with gf2:
    risk_filter = st.multiselect("Escalation Risk", ['High', 'Medium', 'Low'],
                                  default=['High', 'Medium', 'Low'], key='global_risk')
with gf3:
    st.markdown("")  # spacer

mask = (
    df['Year'].between(year_range[0], year_range[1]) &
    df['Escalation_Risk'].isin(risk_filter)
)
dff = df[mask].copy()


# ─────────────────────────────────────────────────────────────
# COMPUTED VALUES
# ─────────────────────────────────────────────────────────────
total = len(dff)
total_value = dff['Deal_Value_USD_M'].sum()
high_risk_count = int(dff['High_Risk_Flag'].sum())
high_risk_pct = (high_risk_count / total * 100) if total > 0 else 0
offensive_pct = (dff['Offensive_Flag'].sum() / total * 100) if total > 0 else 0
top_exporter = dff['Exporter'].value_counts().index[0] if total > 0 else 'N/A'
top_exporter_n = int(dff['Exporter'].value_counts().iloc[0]) if total > 0 else 0
accel_count = len(dff[dff['Arms_Import_Trend'] == 'Accelerating'])
accel_pct = (accel_count / total * 100) if total > 0 else 0
conflict_deals = len(dff[dff['Importer_Conflict_Proximity'] == 'Yes'])
conflict_pct = (conflict_deals / total * 100) if total > 0 else 0


# ═════════════════════════════════════════════════════════════
# KPI STRIP
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='kpi-strip'>
    <div class='kpi-item'>
        <div class='kpi-val'>{total:,}</div>
        <div class='kpi-lbl'>Transfers</div>
        <div class='kpi-sub muted'>${total_value:,.0f}M</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{high_risk_count}</div>
        <div class='kpi-lbl'>High Risk</div>
        <div class='kpi-sub red'>{high_risk_pct:.1f}%</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{offensive_pct:.0f}%</div>
        <div class='kpi-lbl'>Offensive</div>
        <div class='kpi-sub ylw'>{int(dff['Offensive_Flag'].sum())} systems</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{top_exporter}</div>
        <div class='kpi-lbl'>Top Exporter</div>
        <div class='kpi-sub muted'>{top_exporter_n} deals</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{accel_pct:.0f}%</div>
        <div class='kpi-lbl'>Accelerating</div>
        <div class='kpi-sub red'>{accel_count} importers</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{conflict_deals}</div>
        <div class='kpi-lbl'>Conflict Zone</div>
        <div class='kpi-sub ylw'>{conflict_pct:.0f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY (auto-generated narrative)
# ═════════════════════════════════════════════════════════════
if total > 0:
    dominant_region = dff.groupby('Importer_Region')['Deal_Value_USD_M'].sum().idxmax()
    dominant_region_val = dff.groupby('Importer_Region')['Deal_Value_USD_M'].sum().max()
    dominant_region_share = (dominant_region_val / total_value * 100) if total_value > 0 else 0

    yearly_vals = dff.groupby('Year')['Deal_Value_USD_M'].sum()
    if len(yearly_vals) >= 4:
        recent_half = yearly_vals.tail(len(yearly_vals) // 2).mean()
        early_half = yearly_vals.head(len(yearly_vals) // 2).mean()
        trend_pct = ((recent_half - early_half) / early_half * 100) if early_half > 0 else 0
        trend_word = "upward" if trend_pct > 5 else "downward" if trend_pct < -5 else "flat"
    else:
        trend_word, trend_pct = "insufficient data", 0

    accel_importers = dff[dff['Arms_Import_Trend'] == 'Accelerating']['Importer'].nunique()
    embargo_deals = len(dff[dff['UN_Embargo'] == 'Yes'])

    top_corridor_df = dff.groupby(['Exporter', 'Importer'])['Deal_Value_USD_M'].sum().nlargest(1)
    if len(top_corridor_df) > 0:
        top_corridor = f"{top_corridor_df.index[0][0]} → {top_corridor_df.index[0][1]}"
        top_corridor_val = top_corridor_df.values[0]
    else:
        top_corridor, top_corridor_val = "N/A", 0

    high_risk_trend = dff[dff['Escalation_Risk'] == 'High'].groupby('Year').size()
    if len(high_risk_trend) >= 4:
        hr_recent = high_risk_trend.tail(3).mean()
        hr_early = high_risk_trend.head(3).mean()
        hr_trend_word = "increasing" if hr_recent > hr_early * 1.1 else "decreasing" if hr_recent < hr_early * 0.9 else "stable"
    else:
        hr_trend_word = "stable"

    bullets = [
        f"<strong>${total_value:,.0f}M</strong> in arms transfers across <strong>{total:,}</strong> deals spanning {int(dff['Year'].min())}–{int(dff['Year'].max())}.",
        f"Transfer volumes show an <strong>{trend_word}</strong> trajectory ({'+' if trend_pct > 0 else ''}{trend_pct:.0f}% period-over-period).",
        f"<strong>{dominant_region}</strong> absorbs {dominant_region_share:.0f}% of total import value — the dominant destination region.",
        f"High-risk classifications are <strong>{hr_trend_word}</strong> — {high_risk_count} deals flagged ({high_risk_pct:.1f}% of total).",
        f"<strong>{accel_importers}</strong> importers show accelerating procurement — a leading buildup indicator.",
        f"Largest corridor: <strong>{top_corridor}</strong> (${top_corridor_val:,.0f}M).{f' {embargo_deals} deals involve UN-embargoed states.' if embargo_deals > 0 else ''}",
    ]

    st.markdown(f"""
    <div class='panel' style='border-left: 3px solid #00c896;'>
        <div class='panel-title'>Executive Summary <span class='pill pill-new'>AUTO-GENERATED</span></div>
        <ul style='margin:0; padding-left:1.2rem; font-size:0.82rem; line-height:1.9; color:#c9d1d9;'>
            {"".join(f"<li>{b}</li>" for b in bullets)}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Anomaly alerts (descriptive context) ──
anomalies = detect_anomalies(dff)
render_alerts(anomalies, 'descriptive', max_show=3)


# ═════════════════════════════════════════════════════════════
# GLOBAL SITUATION MAP + LATEST DEALS
# ═════════════════════════════════════════════════════════════
st.markdown("<div class='sec-div'><h3>Global Arms Flow Map</h3><p>Importer risk heatmap with deal volume overlay — animated by year</p></div>", unsafe_allow_html=True)

map_col, deals_col = st.columns([6.5, 3.5])

with map_col:
    if total > 0 and 'Importer_ISO3' in dff.columns:
        # Aggregate by importer + year
        map_agg = dff.groupby(['Importer', 'Importer_ISO3', 'Year']).agg(
            Total_Value=('Deal_Value_USD_M', 'sum'),
            Avg_Risk=('Escalation_Flag', 'mean'),
            Deal_Count=('Year', 'count'),
            High_Risk_Count=('High_Risk_Flag', 'sum'),
        ).reset_index()

        # Filter out TWN (not in Natural Earth geometry)
        map_agg = map_agg[map_agg['Importer_ISO3'] != 'TWN']

        # Build animated choropleth + bubble using manual frames
        years = sorted(map_agg['Year'].unique())
        global_max_val = map_agg['Total_Value'].max()

        # First frame data
        yr0 = map_agg[map_agg['Year'] == years[0]]
        fig_map = go.Figure(data=[
            go.Choropleth(
                locations=yr0['Importer_ISO3'], z=yr0['Avg_Risk'],
                colorscale=[[0, '#00c896'], [0.5, '#d29922'], [1, '#f85149']],
                zmin=0, zmax=2, showscale=True,
                colorbar=dict(title='Risk', thickness=10, len=0.4, x=1.02),
                text=yr0.apply(lambda r: f"{r['Importer']}<br>${r['Total_Value']:,.0f}M<br>{r['Deal_Count']} deals<br>Risk: {r['Avg_Risk']:.1f}", axis=1),
                hoverinfo='text',
            ),
            go.Scattergeo(
                locations=yr0['Importer_ISO3'], locationmode='ISO-3',
                marker=dict(
                    size=np.clip(yr0['Total_Value'] / max(global_max_val, 1) * 40, 4, 40).values,
                    color='rgba(14,165,233,0.55)',
                    line=dict(width=0.5, color='#0ea5e9'),
                    sizemode='diameter',
                ),
                text=yr0.apply(lambda r: f"{r['Importer']}: ${r['Total_Value']:,.0f}M", axis=1),
                hoverinfo='text', showlegend=False,
            ),
        ])

        # Build frames
        frames = []
        for year in years:
            yr_data = map_agg[map_agg['Year'] == year]
            frames.append(go.Frame(
                data=[
                    go.Choropleth(
                        locations=yr_data['Importer_ISO3'], z=yr_data['Avg_Risk'],
                        colorscale=[[0, '#00c896'], [0.5, '#d29922'], [1, '#f85149']],
                        zmin=0, zmax=2, showscale=True,
                        colorbar=dict(title='Risk', thickness=10, len=0.4, x=1.02),
                        text=yr_data.apply(lambda r: f"{r['Importer']}<br>${r['Total_Value']:,.0f}M<br>{r['Deal_Count']} deals<br>Risk: {r['Avg_Risk']:.1f}", axis=1),
                        hoverinfo='text',
                    ),
                    go.Scattergeo(
                        locations=yr_data['Importer_ISO3'], locationmode='ISO-3',
                        marker=dict(
                            size=np.clip(yr_data['Total_Value'] / max(global_max_val, 1) * 40, 4, 40).values,
                            color='rgba(14,165,233,0.55)',
                            line=dict(width=0.5, color='#0ea5e9'),
                            sizemode='diameter',
                        ),
                        text=yr_data.apply(lambda r: f"{r['Importer']}: ${r['Total_Value']:,.0f}M", axis=1),
                        hoverinfo='text', showlegend=False,
                    ),
                ],
                name=str(year),
            ))
        fig_map.frames = frames

        # Slider & play button
        fig_map.update_layout(
            sliders=[dict(
                active=0,
                steps=[dict(args=[[str(y)], dict(frame=dict(duration=600, redraw=True), mode='immediate')],
                            label=str(y), method='animate') for y in years],
                x=0.05, len=0.9, y=0,
                currentvalue=dict(prefix='Year: ', font=dict(size=12, color='#c9d1d9')),
                font=dict(color='#8b949e'),
                bgcolor='#161b22', activebgcolor='#00c896', bordercolor='#21262d',
            )],
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.0, y=-0.05,
                buttons=[
                    dict(label='▶', method='animate',
                         args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]),
                    dict(label='⏸', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')]),
                ],
                font=dict(color='#c9d1d9'),
                bgcolor='#161b22', bordercolor='#21262d',
            )],
        )

        fig_map.update_geos(
            bgcolor='rgba(0,0,0,0)',
            landcolor='#161b22',
            oceancolor='#0d1117',
            lakecolor='#0d1117',
            coastlinecolor='#21262d',
            countrycolor='#21262d',
            showframe=False,
            projection_type='natural earth',
        )
        fig_map.update_layout(
            **PLOTLY_LAYOUT, height=480,
            geo=dict(bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_map, use_container_width=True)

with deals_col:
    st.markdown("<div class='panel'><div class='panel-title'>Highest-Value Deals <span class='pill pill-critical'>TOP 5</span></div>", unsafe_allow_html=True)

    if total > 0:
        top5_deals = dff.nlargest(5, 'Deal_Value_USD_M')
        for _, deal in top5_deals.iterrows():
            risk_cls = 'pill-critical' if deal['Escalation_Risk'] == 'High' else 'pill-high' if deal['Escalation_Risk'] == 'Medium' else 'pill-medium'

            # Auto-generate implication
            if deal['Escalation_Risk'] == 'High' and deal['Importer_Conflict_Proximity'] == 'Yes':
                impl = "High-risk transfer to active conflict zone — escalation concern."
            elif deal['UN_Embargo'] == 'Yes':
                impl = "Transfer to UN-embargoed state — potential sanctions issue."
            elif deal['Arms_Import_Trend'] == 'Accelerating':
                impl = "Part of accelerating procurement trend — buildup pattern."
            elif deal['Escalation_Risk'] == 'High':
                impl = "Significant arms transfer to politically unstable importer."
            elif deal['Weapon_Class'] == 'Offensive' and deal['Deal_Value_USD_M'] > 200:
                impl = f"Major offensive capability transfer under {deal['Deal_Framework']}."
            else:
                impl = f"{deal['Weapon_Category']} transfer via {deal['Deal_Framework']}."

            st.markdown(f"""
            <div class='deal-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.8rem; font-weight:600; color:#f0f6fc;'>{deal['Exporter']} → {deal['Importer']}</span>
                    <span class='pill {risk_cls}'>{deal['Escalation_Risk']}</span>
                </div>
                <div style='font-size:0.74rem; color:#8b949e; margin-top:0.2rem;'>
                    {deal['Weapon_Subtype']} &bull;
                    <span style='color:#00c896; font-family:JetBrains Mono; font-weight:600;'>${deal['Deal_Value_USD_M']:,.1f}M</span>
                    &bull; {deal['Year']}
                </div>
                <div style='font-size:0.7rem; color:#6e7681; margin-top:0.15rem; font-style:italic;'>{impl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# COUNTRY DRILL-DOWN
# ═════════════════════════════════════════════════════════════
st.markdown("<div class='sec-div'><h3>Country Intelligence Profile</h3><p>Select a country for detailed arms trade analysis</p></div>", unsafe_allow_html=True)

all_countries = sorted(dff['Importer'].unique()) if total > 0 else []
selected_country = st.selectbox("Select Country", ['— Select —'] + all_countries, key='drill_country')

if selected_country != '— Select —' and total > 0:
    cdf = dff[dff['Importer'] == selected_country]
    if len(cdf) > 0:
        cd1, cd2, cd3, cd4 = st.columns(4)

        with cd1:
            suppliers = cdf.groupby('Exporter')['Deal_Value_USD_M'].sum().sort_values(ascending=True).tail(5)
            fig = go.Figure(go.Bar(
                y=suppliers.index, x=suppliers.values, orientation='h',
                marker=dict(color=suppliers.values, colorscale=[[0, '#0c4a3e'], [1, '#00c896']]),
                text=[f'${v:,.0f}M' for v in suppliers.values], textposition='outside', textfont=dict(size=9),
            ))
            fig.update_layout(title=f'Top Suppliers')
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        with cd2:
            risk_trend = cdf.groupby('Year').agg(
                Avg_Risk=('Escalation_Flag', 'mean'), Deals=('Year', 'count')
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=risk_trend['Year'], y=risk_trend['Deals'], name='Deals',
                                  marker_color='rgba(14,165,233,0.3)'), secondary_y=False)
            fig.add_trace(go.Scatter(x=risk_trend['Year'], y=risk_trend['Avg_Risk'], mode='lines+markers',
                                      name='Risk Score', line=dict(color='#f85149', width=2)), secondary_y=True)
            fig.update_layout(title='Risk & Volume Trend', legend=dict(orientation='h', y=-0.2))
            fig.update_yaxes(title_text="Deals", secondary_y=False)
            fig.update_yaxes(title_text="Risk", secondary_y=True)
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        with cd3:
            weapon_mix = cdf.groupby('Weapon_Category').size().reset_index(name='Count')
            fig = go.Figure(go.Pie(
                labels=weapon_mix['Weapon_Category'], values=weapon_mix['Count'],
                hole=0.6, textinfo='percent+label', textfont=dict(size=9),
                marker=dict(colors=px.colors.qualitative.Dark2),
            ))
            fig.update_layout(title='Weapon Mix', showlegend=False)
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        with cd4:
            c_val = cdf['Deal_Value_USD_M'].sum()
            c_hr = int(cdf['High_Risk_Flag'].sum())
            c_hr_pct = (c_hr / len(cdf) * 100)
            c_stab = cdf['Importer_Political_Stability'].mean()
            c_dem = cdf['Importer_Democracy_Index'].mean()
            c_conflict = cdf['Importer_Conflict_Proximity'].mode().iloc[0]
            c_trend = cdf['Arms_Import_Trend'].mode().iloc[0]

            st.markdown(f"""<div class='panel'>
                <div class='panel-title'>{selected_country}</div>
                <div class='stat-row'><span class='label'>Total Deals</span><span class='value'>{len(cdf)}</span></div>
                <div class='stat-row'><span class='label'>Total Value</span><span class='value'>${c_val:,.0f}M</span></div>
                <div class='stat-row'><span class='label'>High Risk</span><span class='value {"red" if c_hr_pct > 40 else ""}'>{c_hr_pct:.1f}%</span></div>
                <div class='stat-row'><span class='label'>Stability</span><span class='value'>{c_stab:.1f}/10</span></div>
                <div class='stat-row'><span class='label'>Democracy</span><span class='value'>{c_dem:.1f}/10</span></div>
                <div class='stat-row'><span class='label'>Conflict Zone</span><span class='value {"red" if c_conflict=="Yes" else "grn"}'>{c_conflict}</span></div>
                <div class='stat-row'><span class='label'>Import Trend</span><span class='value {"red" if c_trend=="Accelerating" else ""}'>{c_trend}</span></div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# IMPORTER WATCHLIST
# ═════════════════════════════════════════════════════════════
if total > 0:
    wl = dff.groupby('Importer').agg(
        Deals=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum'),
        HR=('High_Risk_Flag', 'sum'),
        Conflict=('Importer_Conflict_Proximity', lambda x: (x == 'Yes').any()),
        Accel=('Arms_Import_Trend', lambda x: (x == 'Accelerating').sum()),
    ).reset_index()
    wl['HR_Rate'] = wl['HR'] / wl['Deals']
    wl['Accel_Rate'] = wl['Accel'] / wl['Deals']
    wl['Concern'] = (wl['HR_Rate'] * 50 + wl['Accel_Rate'] * 30 + wl['Conflict'].astype(int) * 20).round(1)
    top5_wl = wl.nlargest(5, 'Concern')

    st.markdown("<div class='panel'><div class='panel-title'>Importer Watchlist <span class='pill pill-critical'>TOP 5 CONCERN</span></div>", unsafe_allow_html=True)
    wl_cols = st.columns(5)
    for col_w, (_, row) in zip(wl_cols, top5_wl.iterrows()):
        risk_cls = 'red' if row['Concern'] > 60 else 'ylw' if row['Concern'] > 40 else 'grn'
        with col_w:
            st.markdown(f"""
            <div style='text-align:center; padding:0.4rem;'>
                <div style='font-size:0.85rem; font-weight:700; color:#f0f6fc;'>{row['Importer']}</div>
                <div class='kpi-val' style='font-size:1.2rem;'>{row['Concern']:.0f}</div>
                <div class='kpi-lbl'>CONCERN</div>
                <div class='kpi-sub {risk_cls}'>{int(row['HR'])} high-risk / {int(row['Accel'])} accel</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# COMMAND OVERVIEW (timeline | donut | regional)
# ═════════════════════════════════════════════════════════════
ov1, ov2, ov3 = st.columns([5, 2, 3])

with ov1:
    yearly = dff.groupby('Year').agg(
        Deals=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum'),
        High_Risk=('High_Risk_Flag', 'sum')
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Deals'], name='Total Deals',
                          marker_color='rgba(0,200,150,0.25)', marker_line=dict(width=0)), secondary_y=False)
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['High_Risk'], name='High Risk',
                          marker_color='rgba(248,81,73,0.5)', marker_line=dict(width=0)), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Value'], name='Value ($M)', mode='lines+markers',
                              line=dict(color='#0ea5e9', width=2), marker=dict(size=5)), secondary_y=True)
    fig.update_layout(title='Arms Transfers Over Time', barmode='overlay',
                      legend=dict(orientation='h', y=-0.2))
    fig.update_yaxes(title_text='Deals', secondary_y=False)
    fig.update_yaxes(title_text='Value ($M)', secondary_y=True)
    st.plotly_chart(styled_chart(fig, 340), use_container_width=True)

with ov2:
    risk_dist = dff['Escalation_Risk'].value_counts()
    fig = go.Figure(go.Pie(
        labels=risk_dist.index, values=risk_dist.values, hole=0.7,
        marker=dict(colors=[RISK_COLORS.get(r, '#484f58') for r in risk_dist.index]),
        textinfo='percent', textfont=dict(size=10),
    ))
    fig.add_annotation(text=f"{high_risk_pct:.0f}%", x=0.5, y=0.5, font=dict(size=22, color='#f85149',
                       family='JetBrains Mono'), showarrow=False)
    fig.update_layout(title='Risk Distribution', showlegend=True,
                      legend=dict(orientation='h', y=-0.15, font=dict(size=9)))
    st.plotly_chart(styled_chart(fig, 340), use_container_width=True)

with ov3:
    region_risk = dff.groupby('Importer_Region').agg(
        Deals=('Year', 'count'), HR=('High_Risk_Flag', 'sum')
    ).reset_index()
    region_risk['HR_Pct'] = (region_risk['HR'] / region_risk['Deals'] * 100).round(1)
    region_risk = region_risk.sort_values('HR_Pct', ascending=False)

    st.markdown("<div class='panel'><div class='panel-title'>Regional Threat Summary</div>", unsafe_allow_html=True)
    for _, row in region_risk.iterrows():
        bar_w = min(row['HR_Pct'], 100)
        cls = 'red' if row['HR_Pct'] > 40 else 'ylw' if row['HR_Pct'] > 25 else ''
        st.markdown(f"""<div class='stat-row'>
            <span class='label'>{row['Importer_Region']}</span>
            <span class='value {cls}'>{row['HR_Pct']:.0f}% HR ({row['Deals']} deals)</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Situation Assessment ──
if total > 0:
    prev_years = dff[dff['Year'] <= dff['Year'].median()]
    recent_years = dff[dff['Year'] > dff['Year'].median()]
    vol_trend = 'expanding' if len(recent_years) > len(prev_years) * 1.05 else 'contracting' if len(recent_years) < len(prev_years) * 0.95 else 'stable'

    primary_corridor_df = dff.groupby(['Exporter', 'Importer'])['Deal_Value_USD_M'].sum().nlargest(1)
    if len(primary_corridor_df) > 0:
        pc_exp, pc_imp = primary_corridor_df.index[0]
        pc_val = primary_corridor_df.values[0]
    else:
        pc_exp, pc_imp, pc_val = 'N/A', 'N/A', 0

    highest_risk_imp = dff.groupby('Importer')['High_Risk_Flag'].mean().idxmax() if total > 0 else 'N/A'
    hr_imp_pct = dff.groupby('Importer')['High_Risk_Flag'].mean().max() * 100 if total > 0 else 0

    offensive_conflict = dff[(dff['Weapon_Class'] == 'Offensive') & (dff['Importer_Conflict_Proximity'] == 'Yes')]
    off_conflict_pct = (len(offensive_conflict) / total * 100) if total > 0 else 0

    st.markdown(f"""<div class='panel' style='border-left: 3px solid #d29922;'>
        <div class='panel-title'>Situation Assessment <span class='pill pill-high'>INTEL BRIEF</span></div>
        <div class='stat-row'><span class='label'>Transfer Volume Trend</span><span class='value'>{vol_trend.upper()}</span></div>
        <div class='stat-row'><span class='label'>Primary Corridor</span><span class='value'>{pc_exp} → {pc_imp} (${pc_val:,.0f}M)</span></div>
        <div class='stat-row'><span class='label'>Highest-Risk Importer</span><span class='value red'>{highest_risk_imp} ({hr_imp_pct:.0f}% HR)</span></div>
        <div class='stat-row'><span class='label'>Offensive → Conflict Zones</span><span class='value {"red" if off_conflict_pct > 15 else "ylw"}'>{off_conflict_pct:.1f}% of all transfers</span></div>
        <div style='font-size:0.78rem; color:#8b949e; margin-top:0.6rem; line-height:1.6;'>
            Global arms transfer activity is <strong style='color:#c9d1d9;'>{vol_trend}</strong>. The primary bilateral corridor
            ({pc_exp}→{pc_imp}) accounts for ${pc_val:,.0f}M. <strong style='color:#f85149;'>{highest_risk_imp}</strong> presents the
            highest concentration of escalation-flagged transfers at {hr_imp_pct:.0f}%. Offensive systems flowing into conflict-adjacent
            zones represent {off_conflict_pct:.1f}% of total volume — {"a significant escalation vector requiring monitoring." if off_conflict_pct > 10 else "within manageable bounds."}
        </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])


# =============================================================
# TAB 1: DESCRIPTIVE
# =============================================================
with tab1:
    # Contextual filter
    ctx_exp = st.multiselect("Filter Exporters", sorted(dff['Exporter'].unique()),
                              default=sorted(dff['Exporter'].unique()), key='tab1_exp')
    dff_t1 = dff[dff['Exporter'].isin(ctx_exp)] if ctx_exp else dff

    # ── Sankey ──
    st.markdown("<div class='sec-div'><h3>Arms Flow Patterns</h3><p>Dollar value flows between exporter and importer regions</p></div>", unsafe_allow_html=True)

    flow = dff_t1.groupby(['Exporter_Region', 'Importer_Region'])['Deal_Value_USD_M'].sum().reset_index()
    flow = flow[flow['Deal_Value_USD_M'] > 0].nlargest(20, 'Deal_Value_USD_M')
    all_labels = list(pd.unique(flow[['Exporter_Region', 'Importer_Region']].values.ravel()))
    src_indices = [all_labels.index(x) for x in flow['Exporter_Region']]
    tgt_indices = [all_labels.index(x) for x in flow['Importer_Region']]
    exp_regions = set(flow['Exporter_Region'].unique())
    node_colors = ['rgba(0,200,150,0.8)' if l in exp_regions else 'rgba(14,165,233,0.8)' for l in all_labels]

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_labels, color=node_colors,
                  line=dict(color='#21262d', width=0.5)),
        link=dict(source=src_indices, target=tgt_indices,
                  value=flow['Deal_Value_USD_M'].values,
                  color='rgba(0,200,150,0.35)')
    ))
    fig.update_layout(title='Exporter Region → Importer Region (by $M value)')
    st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    # Flow insight
    if len(flow) > 0:
        top_flow = flow.nlargest(1, 'Deal_Value_USD_M').iloc[0]
        top3_val = flow.nlargest(3, 'Deal_Value_USD_M')['Deal_Value_USD_M'].sum()
        concentration = (top3_val / flow['Deal_Value_USD_M'].sum() * 100) if flow['Deal_Value_USD_M'].sum() > 0 else 0
        diversification = 'highly concentrated' if concentration > 60 else 'moderately diversified' if concentration > 40 else 'well diversified'

        fa1, fa2 = st.columns([2, 3])
        with fa1:
            st.markdown(f"""<div class='panel'>
                <div class='panel-title'>Flow Concentration</div>
                <div class='stat-row'><span class='label'>Dominant corridor</span>
                    <span class='value'>{top_flow['Exporter_Region']} → {top_flow['Importer_Region']}</span></div>
                <div class='stat-row'><span class='label'>Corridor value</span>
                    <span class='value'>${top_flow['Deal_Value_USD_M']:,.0f}M</span></div>
                <div class='stat-row'><span class='label'>Top 3 share</span>
                    <span class='value {"red" if concentration > 60 else ""}'>{concentration:.0f}%</span></div>
            </div>""", unsafe_allow_html=True)
        with fa2:
            st.markdown(f"""<div class='callout'>
                <strong>Flow Analysis:</strong> Global arms transfers are <strong>{diversification}</strong> —
                top 3 corridors account for {concentration:.0f}% of total flows.
                {"High concentration increases systemic risk — single supplier disruption could destabilise multiple regional security architectures." if concentration > 50 else "Diversified flows reduce single-point-of-failure risk in the global arms supply chain."}
            </div>""", unsafe_allow_html=True)

    render_alerts(anomalies, 'descriptive', 2)

    # ── Key Players & Arsenal ──
    st.markdown("<div class='sec-div'><h3>Key Players & Arsenal</h3><p>Top exporters, weapon hierarchy, and highest-volume importers</p></div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns([3, 4, 3])

    with d1:
        exp_agg = dff_t1.groupby('Exporter').agg(Deals=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum')).reset_index()
        exp_agg = exp_agg.sort_values('Value', ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            y=exp_agg['Exporter'], x=exp_agg['Value'], orientation='h',
            marker=dict(color=exp_agg['Value'], colorscale=[[0, '#0c4a3e'], [1, '#00c896']]),
            text=exp_agg.apply(lambda r: f"${r['Value']:,.0f}M", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Top Exporters ($M)')
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with d2:
        tree_df = dff_t1.groupby(['Weapon_Category', 'Weapon_Subtype', 'Weapon_Class']).size().reset_index(name='Count')
        fig = px.treemap(tree_df, path=['Weapon_Category', 'Weapon_Subtype', 'Weapon_Class'], values='Count',
                         color='Weapon_Class', color_discrete_map=CLASS_COLORS,
                         title='Weapon Hierarchy')
        fig.update_traces(textfont_size=11, textposition='middle center', insidetextfont=dict(size=10),
                          textinfo='label')
        fig.update_layout(uniformtext=dict(minsize=9, mode='hide'))
        st.plotly_chart(styled_chart(fig, 440), use_container_width=True)

    with d3:
        imp_agg = dff_t1.groupby('Importer').agg(Deals=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum'),
                                                   Avg_Risk=('Escalation_Flag', 'mean')).reset_index()
        imp_agg = imp_agg.sort_values('Value', ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            y=imp_agg['Importer'], x=imp_agg['Value'], orientation='h',
            marker=dict(color=imp_agg['Avg_Risk'],
                        colorscale=[[0, '#00c896'], [0.5, '#d29922'], [1, '#f85149']],
                        colorbar=dict(title='Risk', thickness=10)),
            text=imp_agg.apply(lambda r: f"${r['Value']:,.0f}M", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Top Importers (by risk)')
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    # Key players insight
    if len(dff_t1) > 0:
        top3_exporters = dff_t1.groupby('Exporter')['Deal_Value_USD_M'].sum().nlargest(3)
        t1_value = dff_t1['Deal_Value_USD_M'].sum()
        top3_share = (top3_exporters.sum() / t1_value * 100) if t1_value > 0 else 0
        riskiest = imp_agg.sort_values('Avg_Risk', ascending=False).iloc[0] if len(imp_agg) > 0 else None
        off_cats = dff_t1.groupby('Weapon_Category')['Offensive_Flag'].mean().sort_values(ascending=False)
        most_off = off_cats.index[0] if len(off_cats) > 0 else 'N/A'
        most_off_pct = (off_cats.iloc[0] * 100) if len(off_cats) > 0 else 0

        st.markdown(f"""<div class='callout'>
            <strong>Key Finding:</strong> Top 3 exporters ({', '.join(top3_exporters.index)}) control <strong>{top3_share:.0f}%</strong> of transfer value.
            {f"<strong>{riskiest['Importer']}</strong> is the highest average-risk importer (score {riskiest['Avg_Risk']:.2f}/2.0)." if riskiest is not None else ""}
            <strong>{most_off}</strong> is the most offensively-skewed category at {most_off_pct:.0f}%.
        </div>""", unsafe_allow_html=True)

    # ── Offensive/Defensive by Region + Sunburst ──
    e1, e2 = st.columns([3, 2])
    with e1:
        class_region = dff_t1.groupby(['Importer_Region', 'Weapon_Class']).size().reset_index(name='Count')
        fig = px.bar(class_region, x='Importer_Region', y='Count', color='Weapon_Class',
                     color_discrete_map=CLASS_COLORS, barmode='group',
                     title='Offensive vs Defensive by Region')
        fig.update_layout(xaxis_tickangle=-25, xaxis_tickfont_size=10, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    with e2:
        sun_df = dff_t1.groupby(['Exporter_Alliance', 'Deal_Framework', 'Escalation_Risk']).size().reset_index(name='Count')
        fig = px.sunburst(sun_df, path=['Exporter_Alliance', 'Deal_Framework', 'Escalation_Risk'],
                          values='Count', color='Escalation_Risk', color_discrete_map=RISK_COLORS,
                          title='Alliance → Framework → Risk')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    # Offensive/defensive insight
    if len(dff_t1) > 0:
        off_by_region = dff_t1.groupby('Importer_Region')['Offensive_Flag'].mean().sort_values(ascending=False)
        most_off_region = off_by_region.index[0] if len(off_by_region) > 0 else 'N/A'
        most_off_region_pct = (off_by_region.iloc[0] * 100) if len(off_by_region) > 0 else 0
        least_off_region = off_by_region.index[-1] if len(off_by_region) > 0 else 'N/A'
        least_off_region_pct = (off_by_region.iloc[-1] * 100) if len(off_by_region) > 0 else 0

        st.markdown(f"""<div class='callout'>
            <strong>Regional Arms Posture:</strong> <strong>{most_off_region}</strong> receives the most offensively-skewed
            transfers ({most_off_region_pct:.0f}% offensive), while <strong>{least_off_region}</strong> skews most defensive
            ({100 - least_off_region_pct:.0f}% defensive). Offensive concentration in volatile regions amplifies escalation risk.
        </div>""", unsafe_allow_html=True)

    # ── Country Comparison Tool ──
    st.markdown("<div class='sec-div'><h3>Country Comparison</h3><p>Side-by-side analysis of two importers</p></div>", unsafe_allow_html=True)

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        country_a = st.selectbox("Country A", sorted(dff['Importer'].unique()), index=0, key='cmp_a')
    with cmp2:
        opts_b = sorted(dff['Importer'].unique())
        country_b = st.selectbox("Country B", opts_b, index=min(1, len(opts_b) - 1), key='cmp_b')

    if country_a and country_b and country_a != country_b and total > 0:
        df_a = dff[dff['Importer'] == country_a]
        df_b = dff[dff['Importer'] == country_b]

        def _metrics(cdf):
            if len(cdf) == 0:
                return {'HR%': 0, 'Stability': 0, 'Democracy': 0, 'MilSpend': 0, 'Offensive%': 0}
            return {
                'HR%': cdf['High_Risk_Flag'].mean() * 100,
                'Stability': cdf['Importer_Political_Stability'].mean(),
                'Democracy': cdf['Importer_Democracy_Index'].mean(),
                'MilSpend': cdf['Importer_Military_Spend_Pct_GDP'].mean(),
                'Offensive%': cdf['Offensive_Flag'].mean() * 100,
            }

        ma, mb = _metrics(df_a), _metrics(df_b)

        cc1, cc2, cc3 = st.columns([3, 4, 3])

        with cc1:
            trend_a = df_a.groupby('Year')['Deal_Value_USD_M'].sum().reset_index()
            trend_b = df_b.groupby('Year')['Deal_Value_USD_M'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_a['Year'], y=trend_a['Deal_Value_USD_M'],
                                      mode='lines+markers', name=country_a, line=dict(color='#00c896')))
            fig.add_trace(go.Scatter(x=trend_b['Year'], y=trend_b['Deal_Value_USD_M'],
                                      mode='lines+markers', name=country_b, line=dict(color='#0ea5e9')))
            fig.update_layout(title='Deal Value Trend ($M)', legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        with cc2:
            labels = list(ma.keys())
            ra = list(ma.values())
            rb = list(mb.values())
            maxes = [max(a, b, 0.01) for a, b in zip(ra, rb)]
            ra_n = [a / m for a, m in zip(ra, maxes)]
            rb_n = [b / m for b, m in zip(rb, maxes)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=ra_n + [ra_n[0]], theta=labels + [labels[0]],
                                           fill='toself', name=country_a, line=dict(color='#00c896'),
                                           fillcolor='rgba(0,200,150,0.12)'))
            fig.add_trace(go.Scatterpolar(r=rb_n + [rb_n[0]], theta=labels + [labels[0]],
                                           fill='toself', name=country_b, line=dict(color='#0ea5e9'),
                                           fillcolor='rgba(14,165,233,0.12)'))
            fig.update_layout(title='Profile Comparison',
                              polar=dict(radialaxis=dict(range=[0, 1.1], gridcolor='rgba(201,209,217,0.08)'),
                                         angularaxis=dict(gridcolor='rgba(201,209,217,0.08)'),
                                         bgcolor='rgba(0,0,0,0)'),
                              legend=dict(orientation='h', y=-0.1))
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

        with cc3:
            sup_a = df_a.groupby('Exporter')['Deal_Value_USD_M'].sum().nlargest(5)
            sup_b = df_b.groupby('Exporter')['Deal_Value_USD_M'].sum().nlargest(5)
            all_sups = sorted(set(sup_a.index) | set(sup_b.index))
            fig = go.Figure()
            fig.add_trace(go.Bar(y=all_sups, x=[sup_a.get(s, 0) for s in all_sups],
                                  orientation='h', name=country_a, marker_color='#00c896'))
            fig.add_trace(go.Bar(y=all_sups, x=[sup_b.get(s, 0) for s in all_sups],
                                  orientation='h', name=country_b, marker_color='#0ea5e9'))
            fig.update_layout(title='Top Suppliers ($M)', barmode='group',
                              legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)


# =============================================================
# TAB 2: DIAGNOSTIC
# =============================================================
with tab2:

    # ── Correlation heatmap ──
    st.markdown("<div class='sec-div'><h3>Feature Correlations</h3><p>Which numeric features correlate with high escalation risk?</p></div>", unsafe_allow_html=True)

    heat_cols = ['Deal_Value_USD_M', 'Importer_GDP_Per_Capita', 'Importer_Political_Stability',
                 'Importer_Democracy_Index', 'Importer_Military_Spend_Pct_GDP',
                 'Offensive_Flag', 'Delivery_Timeline_Months', 'High_Risk_Flag']
    corr_mat = dff[heat_cols].corr()
    fig = px.imshow(corr_mat, text_auto='.2f',
                    color_continuous_scale=[[0, '#00c896'], [0.5, '#0d1117'], [1, '#f85149']],
                    zmin=-1, zmax=1, title='Feature Correlation Matrix')
    fig.update_traces(textfont_size=10)
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # Correlation insight (NEW — was missing)
    risk_corr_series = corr_mat['High_Risk_Flag'].drop('High_Risk_Flag')
    strongest_pos = risk_corr_series.idxmax()
    strongest_pos_val = risk_corr_series.max()
    strongest_neg = risk_corr_series.idxmin()
    strongest_neg_val = risk_corr_series.min()

    st.markdown(f"""<div class='callout'>
        <strong>Correlation Insight:</strong> The strongest predictor of high escalation risk is
        <strong>{strongest_neg.replace('_', ' ')}</strong> (r = {strongest_neg_val:.2f}) — {"lower values strongly associate with higher risk" if strongest_neg_val < 0 else "higher values associate with risk"}.
        <strong>{strongest_pos.replace('_', ' ')}</strong> shows the strongest positive correlation (r = {strongest_pos_val:.2f}).
        {"Political stability and democracy indices are inversely correlated with risk — less stable, less democratic states receive more high-risk transfers." if 'Stability' in strongest_neg or 'Democracy' in strongest_neg else ""}
    </div>""", unsafe_allow_html=True)

    # ── Risk Profile Analysis ──
    st.markdown("<div class='sec-div'><h3>Risk Profile Analysis</h3><p>High-risk vs Low-risk importer profiles and key correlations</p></div>", unsafe_allow_html=True)

    profile_cols = ['Importer_Political_Stability', 'Importer_Democracy_Index',
                    'Importer_Military_Spend_Pct_GDP', 'Deal_Value_USD_M', 'Offensive_Flag']
    profile_labels = ['Political Stability', 'Democracy Index', 'Military Spend % GDP',
                      'Deal Value ($M)', 'Offensive Weapon Share']

    r1, r2, r3 = st.columns([3, 2, 3])

    with r1:
        high_vals, low_vals = [], []
        for col in profile_cols:
            h = dff[dff['Escalation_Risk'] == 'High'][col].mean()
            l = dff[dff['Escalation_Risk'] == 'Low'][col].mean()
            col_max = max(h, l, 0.01)
            high_vals.append(h / col_max)
            low_vals.append(l / col_max)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=high_vals + [high_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='High Risk', line=dict(color='#f85149'),
                                       fillcolor='rgba(248,81,73,0.12)'))
        fig.add_trace(go.Scatterpolar(r=low_vals + [low_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='Low Risk', line=dict(color='#00c896'),
                                       fillcolor='rgba(0,200,150,0.12)'))
        fig.update_layout(title='Risk Profile Radar',
                          polar=dict(radialaxis=dict(range=[0, 1.1], gridcolor='rgba(201,209,217,0.08)'),
                                     angularaxis=dict(gridcolor='rgba(201,209,217,0.08)'),
                                     bgcolor='rgba(0,0,0,0)'),
                          legend=dict(orientation='h', y=-0.1))
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    with r2:
        gap_data = []
        for col, label in zip(profile_cols, profile_labels):
            h_mean = dff[dff['Escalation_Risk'] == 'High'][col].mean()
            l_mean = dff[dff['Escalation_Risk'] == 'Low'][col].mean()
            t_stat, p_val = stats.ttest_ind(
                dff[dff['Escalation_Risk'] == 'High'][col].dropna(),
                dff[dff['Escalation_Risk'] == 'Low'][col].dropna()
            )
            gap_data.append({'Factor': label, 'High': round(h_mean, 2), 'Low': round(l_mean, 2),
                             'Gap': round(abs(h_mean - l_mean), 2), 'p': round(p_val, 4)})
        gap_df = pd.DataFrame(gap_data)

        st.markdown("<div class='panel'><div class='panel-title'>Gap Analysis <span class='pill pill-new'>T-TEST</span></div>", unsafe_allow_html=True)
        for _, row in gap_df.iterrows():
            sig = '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
            st.markdown(f"""<div class='stat-row'>
                <span class='label'>{row['Factor']}</span>
                <span class='value'>Gap: {row['Gap']:.1f} {sig}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class='callout'>
            <strong>Low political stability</strong> and <strong>low democracy index</strong> are the strongest
            differentiators between high and low risk importers.
        </div>""", unsafe_allow_html=True)

    with r3:
        numeric_cols = ['Deal_Value_USD_M', 'Quantity', 'Delivery_Timeline_Months',
                        'Importer_GDP_Per_Capita', 'Importer_Political_Stability',
                        'Importer_Democracy_Index', 'Importer_Military_Spend_Pct_GDP',
                        'Offensive_Flag', 'High_Risk_Flag']
        corr_m = dff[numeric_cols].corr()
        risk_corr = corr_m['High_Risk_Flag'].drop('High_Risk_Flag').sort_values()

        fig = go.Figure(go.Bar(
            y=risk_corr.index, x=risk_corr.values, orientation='h',
            marker=dict(color=risk_corr.values,
                        colorscale=[[0, '#00c896'], [0.5, '#484f58'], [1, '#f85149']], cmid=0),
            text=[f'{v:.3f}' for v in risk_corr.values], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Correlation with Escalation Risk')
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    # ── Chi-Square ──
    st.markdown("<div class='sec-div'><h3>Statistical Significance</h3><p>Chi-Square tests for categorical factor association with escalation risk</p></div>", unsafe_allow_html=True)

    cat_test_cols = ['Weapon_Category', 'Weapon_Class', 'Deal_Framework', 'Exporter_Alliance',
                     'Importer_Conflict_Proximity', 'Active_Territorial_Dispute',
                     'Natural_Resource_Dependence', 'Arms_Import_Trend', 'UN_Embargo',
                     'Technology_Transfer', 'UNSC_Permanent_Member', 'Importer_Region']
    chi2_results = []
    for col in cat_test_cols:
        if col in dff.columns:
            ct = pd.crosstab(dff[col], dff['Escalation_Risk'])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
                chi2_results.append({'Feature': col, 'Chi2': round(chi2, 2), 'p-value': round(p, 5),
                                     "Cramers_V": round(cramers_v, 3),
                                     'Significant': 'Yes' if p < 0.05 else 'No'})
    chi_df = pd.DataFrame(chi2_results).sort_values("Cramers_V", ascending=False)

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        fig = go.Figure(go.Bar(
            x=chi_df["Cramers_V"], y=chi_df['Feature'], orientation='h',
            marker=dict(color=chi_df["Cramers_V"], colorscale=[[0, '#0ea5e9'], [1, '#f85149']]),
            text=chi_df["Cramers_V"], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title="Cramer's V — Effect Size")
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with ch2:
        # Dark-themed table instead of st.dataframe
        chi_display = chi_df.set_index('Feature')
        st.markdown(f"<div class='panel'>{render_dark_table(chi_display)}</div>", unsafe_allow_html=True)

    # Chi-square insight
    sig_features = chi_df[chi_df['Significant'] == 'Yes']
    top_assoc = chi_df.iloc[0] if len(chi_df) > 0 else None
    n_sig = len(sig_features)
    st.markdown(f"""<div class='callout'>
        <strong>Statistical Finding:</strong> {n_sig} of {len(chi_df)} categorical features show significant
        association with escalation risk (p < 0.05).
        {f"<strong>{top_assoc['Feature']}</strong> has the strongest effect (Cramer's V = {top_assoc['Cramers_V']:.3f})." if top_assoc is not None else ""}
        {f"Features with V > 0.2 ({', '.join(sig_features[sig_features['Cramers_V'] > 0.2]['Feature'].tolist()) or 'none'}) could form rule-based screening criteria." if len(sig_features) > 0 else ""}
    </div>""", unsafe_allow_html=True)

    # ── Risk Factor Combinations ──
    st.markdown("<div class='sec-div'><h3>Risk Factor Combinations</h3><p>Multi-factor profiles with highest escalation rates</p></div>", unsafe_allow_html=True)

    risk_combos = []
    for conflict in ['Yes', 'No']:
        for dispute in ['Yes', 'No']:
            for wclass in ['Offensive', 'Defensive']:
                for trend in ['Accelerating', 'Stable', 'Declining']:
                    subset = dff[(dff['Importer_Conflict_Proximity'] == conflict) &
                                 (dff['Active_Territorial_Dispute'] == dispute) &
                                 (dff['Weapon_Class'] == wclass) &
                                 (dff['Arms_Import_Trend'] == trend)]
                    if len(subset) >= 10:
                        rate = subset['High_Risk_Flag'].mean() * 100
                        risk_combos.append({
                            'Conflict': conflict, 'Dispute': dispute,
                            'Class': wclass, 'Trend': trend,
                            'Count': len(subset), 'High Risk %': round(rate, 1)
                        })

    risk_cdf = pd.DataFrame(risk_combos).sort_values('High Risk %', ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=risk_cdf['High Risk %'],
        y=risk_cdf.apply(lambda r: f"{r['Conflict']}/{r['Dispute']}/{r['Class']}/{r['Trend']}", axis=1),
        orientation='h',
        marker=dict(color=risk_cdf['High Risk %'], colorscale=[[0, '#d29922'], [1, '#f85149']]),
        text=risk_cdf.apply(lambda r: f"{r['High Risk %']}% (n={r['Count']})", axis=1),
        textposition='outside', textfont=dict(size=9)
    ))
    fig.update_layout(title='Conflict / Dispute / Class / Trend → Escalation Rate',
                      xaxis_title='High Risk %')
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    if len(risk_cdf) > 0:
        deadliest = risk_cdf.iloc[0]
        deadliest_rate = deadliest['High Risk %']
        deadliest_label = f"Conflict:{deadliest['Conflict']} / Dispute:{deadliest['Dispute']} / {deadliest['Class']} / {deadliest['Trend']}"
        avg_base_rate = high_risk_pct
        risk_multiplier = (deadliest_rate / avg_base_rate) if avg_base_rate > 0 else 0

        st.markdown(f"""<div class='panel' style='border-left: 3px solid #f85149;'>
            <div class='panel-title'>Diagnostic Summary <span class='pill pill-critical'>KEY FINDING</span></div>
            <div style='font-size:0.82rem; color:#c9d1d9; line-height:1.6;'>
                The deadliest risk profile is <strong style='color:#f85149;'>{deadliest_label}</strong> at
                <strong style='color:#f85149;'>{deadliest_rate:.0f}%</strong> high-risk rate — that's
                <strong style='color:#f0f6fc;'>{risk_multiplier:.1f}x</strong> the baseline of {avg_base_rate:.0f}%.
                Transfers matching top 3 profiles should trigger mandatory enhanced due diligence.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Embargo ──
    embargo_df = dff[dff['UN_Embargo'] == 'Yes']
    if len(embargo_df) > 0:
        st.markdown("<div class='sec-div'><h3>Embargo Circumvention</h3><p>Arms flowing to UN-embargoed destinations</p></div>", unsafe_allow_html=True)
        emb_by_exp = embargo_df.groupby('Exporter').agg(Deals=('Year', 'count'), Value=('Deal_Value_USD_M', 'sum')).reset_index()
        emb_by_exp = emb_by_exp.sort_values('Value', ascending=True)
        fig = go.Figure(go.Bar(
            y=emb_by_exp['Exporter'], x=emb_by_exp['Value'], orientation='h',
            marker_color='#f85149',
            text=emb_by_exp.apply(lambda r: f"${r['Value']:,.0f}M ({r['Deals']})", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Arms to Embargoed Destinations by Exporter')
        st.plotly_chart(styled_chart(fig, 280), use_container_width=True)

        # Embargo insight (NEW — was missing)
        top_emb_exp = emb_by_exp.iloc[-1] if len(emb_by_exp) > 0 else None
        emb_total = embargo_df['Deal_Value_USD_M'].sum()
        st.markdown(f"""<div class='callout'>
            <strong>Embargo Analysis:</strong> ${emb_total:,.0f}M in arms transfers reached UN-embargoed states
            across {len(embargo_df)} deals.
            {f"<strong>{top_emb_exp['Exporter']}</strong> is the largest supplier (${top_emb_exp['Value']:,.0f}M, {int(top_emb_exp['Deals'])} deals)." if top_emb_exp is not None else ""}
            These transfers represent potential sanctions violations requiring enhanced monitoring.
        </div>""", unsafe_allow_html=True)

    render_alerts(anomalies, 'diagnostic', 3)


# =============================================================
# TAB 3: PREDICTIVE
# =============================================================
with tab3:
    st.markdown("<div class='sec-div'><h3>Escalation Risk Classification</h3><p>ML models predicting High vs Non-High escalation risk</p></div>", unsafe_allow_html=True)

    @st.cache_data
    def run_predictive_models(data):
        df_ml = data.copy()
        cat_features = ['Exporter_Alliance', 'Weapon_Category', 'Weapon_Class', 'Deal_Framework',
                        'Importer_Conflict_Proximity', 'Active_Territorial_Dispute',
                        'Natural_Resource_Dependence', 'Arms_Import_Trend', 'UN_Embargo',
                        'Technology_Transfer', 'UNSC_Permanent_Member', 'Importer_Region']
        for c in cat_features:
            le = LabelEncoder()
            df_ml[c + '_enc'] = le.fit_transform(df_ml[c])
        feature_cols = ['Deal_Value_USD_M', 'Quantity', 'Delivery_Timeline_Months',
                        'Importer_GDP_Per_Capita', 'Importer_Political_Stability',
                        'Importer_Democracy_Index', 'Importer_Military_Spend_Pct_GDP'] + \
                       [c + '_enc' for c in cat_features]
        X = df_ml[feature_cols]
        y = df_ml['High_Risk_Flag']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
        }
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            model.fit(X_scaled, y)
            importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
            results[name] = {'auc_mean': scores.mean(), 'auc_std': scores.std(),
                             'importance': pd.Series(importance, index=feature_cols).sort_values(ascending=False)}
        roc_data = {}
        for name, model in models.items():
            y_prob = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        return results, roc_data, feature_cols

    results, roc_data, feature_cols = run_predictive_models(df)

    # Model cards
    model_colors = {'Logistic Regression': '#00c896', 'Random Forest': '#0ea5e9', 'Gradient Boosting': '#d29922'}
    mc1, mc2, mc3 = st.columns(3)
    for col_widget, (name, res) in zip([mc1, mc2, mc3], results.items()):
        color = model_colors[name]
        with col_widget:
            st.markdown(f"""
            <div class='model-card'>
                <div class='model-name'>{name}</div>
                <div class='model-auc' style='color:{color};'>{res['auc_mean']:.3f}</div>
                <div class='model-std'>AUC &plusmn; {res['auc_std']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ROC curves
    fig = go.Figure()
    for name, rdata in roc_data.items():
        fig.add_trace(go.Scatter(x=rdata['fpr'], y=rdata['tpr'], mode='lines',
                                 name=f"{name} ({rdata['auc']:.3f})",
                                 line=dict(color=model_colors[name], width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                             line=dict(color='#30363d', dash='dash', width=1)))
    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    # Feature importance
    st.markdown("<div class='sec-div'><h3>Feature Importance</h3><p>Which features drive escalation risk predictions?</p></div>", unsafe_allow_html=True)

    f1, f2 = st.columns([1, 1])
    with f1:
        selected_model = st.selectbox("Model:", list(results.keys()), index=1)
        imp = results[selected_model]['importance'].head(12)
        fig = go.Figure(go.Bar(
            y=imp.index[::-1], x=imp.values[::-1], orientation='h',
            marker=dict(color=imp.values[::-1], colorscale=[[0, '#0c4a3e'], [1, '#f85149']]),
            text=[f'{v:.4f}' for v in imp.values[::-1]], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title=f'{selected_model} — Top 12 Features')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    with f2:
        all_imp = pd.DataFrame()
        for name, res in results.items():
            all_imp[name] = res['importance'] / res['importance'].max()
        all_imp['Mean'] = all_imp.mean(axis=1)
        all_imp = all_imp.sort_values('Mean', ascending=False).head(12)

        fig = go.Figure()
        for name in results.keys():
            fig.add_trace(go.Bar(name=name, y=all_imp.index[::-1], x=all_imp[name].values[::-1],
                                 orientation='h', marker_color=model_colors[name], opacity=0.75))
        fig.update_layout(title='Consensus Ranking (All Models)', barmode='group', xaxis_title='Normalized Importance')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    st.markdown("""<div class='callout'>
        <strong>Model Insight:</strong> <strong>Political stability, democracy index, and conflict proximity</strong> consistently
        emerge as the strongest predictors across all three models. Weapon class and arms import trend are secondary signals.
        These features should form the core of any rule-based early warning system.
    </div>""", unsafe_allow_html=True)


# =============================================================
# TAB 4: PRESCRIPTIVE
# =============================================================
with tab4:
    # Contextual filter
    ctx_region = st.multiselect("Filter Regions", sorted(dff['Importer_Region'].unique()),
                                 default=sorted(dff['Importer_Region'].unique()), key='tab4_region')
    dff_t4 = dff[dff['Importer_Region'].isin(ctx_region)] if ctx_region else dff

    st.markdown("<div class='sec-div'><h3>Risk Assessment</h3><p>Interactive risk simulator and regional threat profiling</p></div>", unsafe_allow_html=True)

    p1, p2 = st.columns([2, 3])

    with p1:
        st.markdown("**Escalation Risk Simulator**")
        sim_stability = st.slider("Political Stability", 1.0, 10.0, 5.0, 0.5, key='s1')
        sim_democracy = st.slider("Democracy Index", 1.0, 10.0, 5.0, 0.5, key='s2')
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_conflict = st.selectbox("Conflict", ['Yes', 'No'], key='s3')
            sim_weapon = st.selectbox("Weapon Class", ['Offensive', 'Defensive'], key='s5')
        with sc2:
            sim_dispute = st.selectbox("Dispute", ['Yes', 'No'], key='s4')
            sim_trend = st.selectbox("Arms Trend", ['Accelerating', 'Stable', 'Declining'], key='s6')
        sim_milspend = st.slider("Military Spend % GDP", 0.5, 8.0, 2.5, 0.5, key='s7')
        sim_resource = st.selectbox("Resource Dependence", ['High', 'Medium', 'Low'], key='s8')

        risk_score = 0
        risk_score += (10 - sim_stability) * 3.0
        risk_score += (10 - sim_democracy) * 1.5
        if sim_conflict == 'Yes': risk_score += 12
        if sim_dispute == 'Yes': risk_score += 8
        if sim_weapon == 'Offensive': risk_score += 5
        if sim_trend == 'Accelerating': risk_score += 7
        elif sim_trend == 'Declining': risk_score -= 3
        if sim_milspend > 4.0: risk_score += 6
        elif sim_milspend > 2.5: risk_score += 3
        if sim_resource == 'High': risk_score += 4
        elif sim_resource == 'Medium': risk_score += 2
        risk_score = min(100, max(0, risk_score))

        risk_color = '#00c896' if risk_score < 28 else '#d29922' if risk_score < 45 else '#f85149'
        risk_label = 'LOW' if risk_score < 28 else 'ELEVATED' if risk_score < 45 else 'CRITICAL'

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': risk_label, 'font': {'size': 14, 'color': risk_color, 'family': 'Inter'}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#30363d'),
                bar=dict(color=risk_color),
                bgcolor='rgba(0,0,0,0)',
                steps=[
                    dict(range=[0, 28], color='rgba(0,200,150,0.06)'),
                    dict(range=[28, 45], color='rgba(210,153,34,0.06)'),
                    dict(range=[45, 100], color='rgba(248,81,73,0.06)'),
                ],
                threshold=dict(line=dict(color='#f85149', width=2), thickness=0.75, value=risk_score)
            ),
            number=dict(suffix='/100', font=dict(size=36, color=risk_color, family='JetBrains Mono'))
        ))
        st.plotly_chart(styled_chart(fig, 250), use_container_width=True)

    with p2:
        region_full = dff_t4.groupby('Importer_Region').agg(
            Total_Deals=('Year', 'count'), High_Risk_Deals=('High_Risk_Flag', 'sum'),
            Total_Value=('Deal_Value_USD_M', 'sum'),
            Avg_Stability=('Importer_Political_Stability', 'mean'),
            Avg_Democracy=('Importer_Democracy_Index', 'mean'),
            Offensive_Pct=('Offensive_Flag', 'mean'),
            Accel_Count=('Arms_Import_Trend', lambda x: (x == 'Accelerating').sum())
        ).reset_index()
        region_full['High_Risk_Pct'] = (region_full['High_Risk_Deals'] / region_full['Total_Deals'] * 100).round(1)
        region_full['Offensive_Pct'] = (region_full['Offensive_Pct'] * 100).round(1)
        region_full = region_full.sort_values('High_Risk_Pct', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=region_full['Importer_Region'], y=region_full['High_Risk_Pct'],
                             name='High Risk %', marker_color='#f85149'))
        fig.add_trace(go.Bar(x=region_full['Importer_Region'], y=region_full['Offensive_Pct'],
                             name='Offensive %', marker_color='rgba(14,165,233,0.5)'))
        fig.update_layout(title='Regional Threat Profile', barmode='group',
                          yaxis_title='%', xaxis_tickangle=-20, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

        # Dark-themed table instead of st.dataframe
        region_display = region_full.set_index('Importer_Region').rename(columns={
            'Total_Deals': 'Deals', 'High_Risk_Deals': 'High Risk', 'Total_Value': 'Value ($M)',
            'Avg_Stability': 'Stability', 'Avg_Democracy': 'Democracy',
            'Offensive_Pct': 'Offensive %', 'Accel_Count': 'Accelerating', 'High_Risk_Pct': 'Risk %'
        })
        st.markdown(f"<div class='panel'>{render_dark_table(region_display)}</div>", unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown("<div class='sec-div'><h3>Strategic Recommendations</h3><p>Evidence-based policy interventions</p></div>", unsafe_allow_html=True)

    conflict_risk_rate = dff[dff['Importer_Conflict_Proximity'] == 'Yes']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity'] == 'Yes']) > 0 else 0
    no_conflict_rate = dff[dff['Importer_Conflict_Proximity'] == 'No']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity'] == 'No']) > 0 else 0
    accel_risk_rate = dff[dff['Arms_Import_Trend'] == 'Accelerating']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Arms_Import_Trend'] == 'Accelerating']) > 0 else 0
    offensive_risk_rate = dff[dff['Weapon_Class'] == 'Offensive']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Weapon_Class'] == 'Offensive']) > 0 else 0
    low_stab_rate = dff[dff['Importer_Political_Stability'] < 4]['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Political_Stability'] < 4]) > 0 else 0

    recommendations = [
        ("Arms Embargo Enforcement", f"Conflict-proximate importers: {conflict_risk_rate:.0f}% high-risk vs {no_conflict_rate:.0f}% non-conflict. Strengthen multilateral monitoring.", "CRITICAL"),
        ("Acceleration Monitoring", f"Accelerating trends show {accel_risk_rate:.0f}% high-risk rate. Deploy real-time tracking and flag >20% YoY acceleration.", "CRITICAL"),
        ("Offensive Transfer Controls", f"Offensive systems: {offensive_risk_rate:.0f}% escalation rate. Stricter end-use certificates for combat aircraft, missiles.", "HIGH"),
        ("Governance-Linked Licensing", f"Stability <4.0: {low_stab_rate:.0f}% risk. Binding governance thresholds in export frameworks.", "HIGH"),
        ("Diplomatic Corridors", "Territorial disputes are top escalation drivers. Prioritise mediation for top dispute dyads.", "MEDIUM"),
        ("Predictive Peacekeeping", "Use ML models as quarterly early-warning for UN DPPA. Shift to anticipatory posture.", "HIGH"),
    ]

    for i in range(0, len(recommendations), 2):
        rc1, rc2 = st.columns(2)
        for col_w, j in zip([rc1, rc2], [i, i + 1]):
            if j < len(recommendations):
                title, desc, priority = recommendations[j]
                pill_cls = 'pill-critical' if priority == 'CRITICAL' else 'pill-high' if priority == 'HIGH' else 'pill-medium'
                with col_w:
                    st.markdown(f"""<div class='rx-card'>
                        <h4>{title} <span class='pill {pill_cls}'>{priority}</span></h4>
                        <p>{desc}</p>
                    </div>""", unsafe_allow_html=True)

    # ── Impact matrix ──
    st.markdown("<div class='sec-div'><h3>Impact vs Feasibility</h3><p>Intervention prioritisation by expected impact and complexity</p></div>", unsafe_allow_html=True)

    impact_data = pd.DataFrame({
        'Intervention': ['Embargo Enforcement', 'Acceleration Monitoring', 'Offensive Controls',
                         'Governance Licensing', 'Diplomatic Corridors', 'Predictive Peacekeeping'],
        'Risk Reduction %': [8.5, 6.2, 5.0, 4.5, 3.8, 7.0],
        'Complexity': [4, 2, 3, 4, 3, 2],
        'Time (months)': [12, 4, 8, 18, 24, 6]
    })

    fig = px.scatter(impact_data, x='Complexity', y='Risk Reduction %',
                     size='Time (months)', text='Intervention',
                     color='Risk Reduction %',
                     color_continuous_scale=[[0, '#d29922'], [1, '#00c896']],
                     title='Intervention Prioritisation (size = time to impact)')
    fig.update_traces(textposition='top center', textfont=dict(size=10, family='Inter'))
    fig.update_layout(xaxis_title='Complexity (1=Easy, 5=Hard)', yaxis_title='Risk Reduction %')
    st.plotly_chart(styled_chart(fig, 380), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.7rem; padding:0.5rem 0;'>
    AEGIS — Arms & Escalation Geopolitical Intelligence System &bull; Synthetic Data (SIPRI methodology) &bull; Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
