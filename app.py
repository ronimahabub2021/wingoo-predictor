#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WinGo Predictor — Streamlit Dashboard v4.0
Complete, error-free, production-ready UI
"""

import time
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st

# ── MUST be the very first Streamlit call ──
st.set_page_config(
    page_title="WinGo Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_autorefresh import st_autorefresh
from wingo_engine import (
    ALGO_ORDER, ALGO_WEIGHTS, MIN_DATA, TOTAL_ALGOS,
    SessionState, fetch_latest, fetch_history_pages,
    load_data, log_prediction, multi_layer_prediction,
    save_to_csv,
)

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

  .big-card {
      background: linear-gradient(135deg, #1a1f2e 0%, #262b3d 100%);
      border-radius: 16px; padding: 18px 22px;
      border: 1px solid #2e3548;
      box-shadow: 0 6px 24px rgba(0,0,0,0.4);
      margin-bottom: 10px;
  }
  .big-card h4 {
      color: #8b95b0; font-size: 0.75rem; margin: 0 0 6px 0;
      text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
  }
  .big-card .value {
      color: #ffffff; font-size: 1.75rem; font-weight: 700; margin: 0;
  }
  .big-card .sub { color: #6b7690; font-size: 0.75rem; margin-top: 4px; }

  .signal-big   { background: linear-gradient(135deg,#0d3b25,#145c39);
                  border:2px solid #22c55e; border-radius:16px;
                  padding:22px; text-align:center;
                  box-shadow: 0 0 30px rgba(34,197,94,0.3); }
  .signal-small { background: linear-gradient(135deg,#3b0d1a,#5c1428);
                  border:2px solid #ef4444; border-radius:16px;
                  padding:22px; text-align:center;
                  box-shadow: 0 0 30px rgba(239,68,68,0.3); }
  .signal-wait  { background: linear-gradient(135deg,#2a2a1a,#3d3d26);
                  border:2px solid #eab308; border-radius:16px;
                  padding:22px; text-align:center;
                  box-shadow: 0 0 30px rgba(234,179,8,0.25); }
  .signal-text  { font-size: 2.8rem; font-weight: 800; margin: 0;
                  line-height: 1; }
  .signal-label { font-size: 0.8rem; color: #cbd5e1; letter-spacing: 2px;
                  text-transform: uppercase; margin-bottom: 6px; }
  .signal-meta  { color: #cbd5e1; font-size: 0.9rem; margin-top: 10px; }

  .conf-bar-bg   { background:#1a1f2e; border-radius:10px; height:20px;
                   overflow:hidden; margin-top:10px;
                   border:1px solid #2e3548; }
  .conf-bar-fill { height:100%; border-radius:10px;
                   background:linear-gradient(90deg,#22c55e,#4ade80);
                   transition: width 0.4s ease; }

  .pill { display:inline-block; padding:3px 10px; margin:2px 4px 2px 0;
          border-radius:20px; font-size:0.72rem; font-weight:600;
          border:1px solid; }
  .pill-big   { background:#0d3b25; color:#4ade80; border-color:#22c55e; }
  .pill-small { background:#3b0d1a; color:#f87171; border-color:#ef4444; }
  .pill-dim   { background:#1a1f2e; color:#8b95b0; border-color:#2e3548; }

  .algo-row { display:flex; align-items:center; padding:6px 10px;
              border-bottom:1px solid #1e2436; font-size:0.86rem; }
  .algo-name { color:#cbd5e1; width:130px; font-weight:600; }
  .algo-vote { width:80px; }
  .algo-conf { color:#94a3b8; width:60px; text-align:right; }
  .algo-w    { color:#64748b; width:60px; text-align:right; }
  .algo-rsn  { color:#64748b; margin-left:16px; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION BOOTSTRAP
# ════════════════════════════════════════════════════════════
if "session" not in st.session_state:
    st.session_state.session = SessionState()
if "pending" not in st.session_state:
    st.session_state.pending = None
if "last_issue_seen" not in st.session_state:
    st.session_state.last_issue_seen = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "boot_done" not in st.session_state:
    st.session_state.boot_done = False
    st.session_state.boot_message = ""
    st.session_state.boot_error = ""

    # ── Initial load ──
    try:
        df_check = load_data()
        if len(df_check) < MIN_DATA:
            with st.spinner("📥 First-time setup: fetching history…"):
                items = fetch_history_pages(pages=4)

                if not items:
                    st.session_state.boot_error = (
                        "⚠️ API থেকে কোনো data আসেনি। Streamlit Cloud এর "
                        "IP block হতে পারে। Logs চেক করুন।")
                else:
                    added = 0
                    for it in reversed(items):
                        try:
                            iss = str(it.get("issueNumber")
                                      or it.get("issue") or "")
                            num = int(it.get("number")
                                      or it.get("num") or 0)
                            if not iss:
                                continue
                            size = "Big" if num >= 5 else "Small"
                            if save_to_csv(iss, num, size):
                                added += 1
                        except Exception:
                            pass
                    st.session_state.boot_message = \
                        f"✅ Loaded {added} new history rows"

        # Fetch latest once
        latest_boot = fetch_latest()
        if latest_boot:
            iss = str(latest_boot.get("issueNumber")
                      or latest_boot.get("issue") or "")
            num = int(latest_boot.get("number")
                      or latest_boot.get("num") or 0)
            if iss:
                size = "Big" if num >= 5 else "Small"
                save_to_csv(iss, num, size)
                st.session_state.last_issue_seen = iss
    except Exception as e:
        st.session_state.boot_error = f"Boot error: {e}"


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 WinGo Predictor")
    st.caption("14-algo ensemble · Live dashboard")

    auto_refresh = st.toggle("🔄 Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh interval (sec)", 3, 30, 6, 1)

    st.divider()
    st.markdown("#### ⚙️ Display")
    show_votes = st.toggle("Algorithm votes", value=True)
    show_history = st.toggle("History table", value=True)
    show_chart = st.toggle("Charts", value=True)

    st.divider()
    sess = st.session_state.session
    st.markdown("#### 📊 Session")
    st.metric("Predictions", sess.total_pred)
    st.metric("Accuracy", f"{sess.accuracy}%")
    st.metric("Recent-20", f"{sess.recent_accuracy}%")
    st.metric("Streak", sess.streak,
              delta=sess.best_streak - sess.streak)

    st.divider()
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.rerun()
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.session = SessionState()
        st.session_state.pending = None
        st.session_state.last_prediction = None
        st.rerun()

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")


# ════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ════════════════════════════════════════════════════════════
if auto_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="wingo_refresh")


# ════════════════════════════════════════════════════════════
#  FETCH LATEST + EVALUATE PREVIOUS
# ════════════════════════════════════════════════════════════
network_ok = True
latest = None
try:
    latest = fetch_latest()
    if latest is None:
        network_ok = False
except Exception as e:
    network_ok = False
    st.error(f"Network: {e}")

if latest:
    try:
        issue = str(latest.get("issueNumber")
                    or latest.get("issue") or "")
        num = int(latest.get("number") or latest.get("num") or 0)
        size = "Big" if num >= 5 else "Small"

        # Evaluate previous prediction
        if (st.session_state.pending and
                st.session_state.last_issue_seen and
                issue != st.session_state.last_issue_seen):
            p = st.session_state.pending
            if p["pred"] not in ("WAIT", "ERROR", ""):
                hit = (p["pred"] == size)
                st.session_state.session.update(p["pred"], size,
                                                 p.get("votes"))
                try:
                    log_prediction(p["issue"], p["pred"], p["conf"],
                                   size, hit)
                except Exception:
                    pass
                st.session_state.last_prediction = {
                    "pred": p["pred"],
                    "actual": f"{num} ({size})",
                    "hit": hit,
                    "issue": p["issue"],
                }

        # Save new draw
        if issue and issue != st.session_state.last_issue_seen:
            save_to_csv(issue, num, size)
            st.session_state.last_issue_seen = issue
    except Exception as e:
        st.error(f"Update error: {e}")


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("# 🎯 WinGo Predictor")
    st.caption("14-algorithm ensemble · Live market dashboard · "
               "Auto-weight learning")
with h2:
    status = "🟢 Online" if network_ok else "🔴 Offline"
    st.markdown(
        f"<div style='text-align:right;padding-top:16px;"
        f"font-size:1.05rem'>{status}</div>",
        unsafe_allow_html=True)

if st.session_state.boot_message:
    st.success(st.session_state.boot_message)
    st.session_state.boot_message = ""

if st.session_state.boot_error:
    st.warning(st.session_state.boot_error)
    st.session_state.boot_error = ""


# ════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════
try:
    df = load_data()
except Exception as e:
    st.error(f"Data load error: {e}")
    df = pd.DataFrame(columns=["Issue", "Number", "Size", "Time"])

st.caption(f"📊 Loaded **{len(df)}** history rows")


# ════════════════════════════════════════════════════════════
#  TOP METRICS
# ════════════════════════════════════════════════════════════
sess = st.session_state.session
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="big-card">
      <h4>Latest Issue</h4>
      <p class="value">{st.session_state.last_issue_seen or '—'}</p>
      <div class="sub">rows: {len(df)}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    if len(df):
        last = df.iloc[-1]
        col = "#4ade80" if last["Size"] == "Big" else "#f87171"
        st.markdown(f"""
        <div class="big-card">
          <h4>Last Result</h4>
          <p class="value" style="color:{col}">
            {int(last['Number'])} ({last['Size']})</p>
          <div class="sub">{last['Time']}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="big-card">
          <h4>Last Result</h4>
          <p class="value">—</p>
          <div class="sub">no data yet</div>
        </div>""", unsafe_allow_html=True)

with c3:
    acc_col = ("#4ade80" if sess.accuracy >= 55
               else "#eab308" if sess.accuracy >= 50
               else "#f87171")
    st.markdown(f"""
    <div class="big-card">
      <h4>Session Accuracy</h4>
      <p class="value" style="color:{acc_col}">{sess.accuracy}%</p>
      <div class="sub">{sess.correct}/{sess.total_pred} correct</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="big-card">
      <h4>Recent-20</h4>
      <p class="value">{sess.recent_accuracy}%</p>
      <div class="sub">streak: {sess.streak} (best {sess.best_streak})</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="big-card">
      <h4>Bayesian Prior</h4>
      <p class="value">{round(sess.prior_big*100,1)}%</p>
      <div class="sub">Big bias</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PREVIOUS VERDICT
# ════════════════════════════════════════════════════════════
lp = st.session_state.last_prediction
if lp:
    if lp["hit"]:
        st.success(f"✔ **CORRECT** — predicted **{lp['pred']}** "
                   f"· actual **{lp['actual']}** (issue {lp['issue']})")
    else:
        st.error(f"✘ **MISSED** — predicted **{lp['pred']}** "
                 f"· actual **{lp['actual']}** (issue {lp['issue']})")


# ════════════════════════════════════════════════════════════
#  SIGNAL PANEL
# ════════════════════════════════════════════════════════════
st.markdown("## 🚦 Next-Round Signal")

result = None
try:
    result = multi_layer_prediction(df, st.session_state.session)
except Exception as e:
    st.error(f"Prediction error: {e}")
    with st.expander("🔍 Traceback"):
        st.code(traceback.format_exc())

if result is None:
    st.warning("Prediction engine returned nothing.")
elif result["pred"] == "WAIT":
    needed = result.get("needed")
    pct = min(int(len(df) / max(MIN_DATA, 1) * 100), 100)
    st.markdown(f"""
    <div class="signal-wait">
      <div class="signal-label">STATUS</div>
      <p class="signal-text" style="color:#eab308">WAIT</p>
      <div class="signal-meta">{result.get('logic','')}</div>
      <div class="conf-bar-bg" style="margin-top:16px">
        <div class="conf-bar-fill" style="width:{pct}%;
             background:linear-gradient(90deg,#eab308,#facc15)"></div>
      </div>
      <div class="signal-meta">{pct}% · need {needed or 0} more rounds</div>
    </div>""", unsafe_allow_html=True)
else:
    pred = result["pred"]
    conf = result["final_confidence"]
    cls = "signal-big" if pred == "Big" else "signal-small"
    pcol = "#4ade80" if pred == "Big" else "#f87171"
    st.markdown(f"""
    <div class="{cls}">
      <div class="signal-label">Prediction</div>
      <p class="signal-text" style="color:{pcol}">{pred.upper()}</p>
      <div class="signal-meta">
        Confidence <b>{conf}%</b> · Agreement <b>{result['agreement']}</b>
        · Big <b>{result['big_score']}</b> vs Small <b>{result['small_score']}</b>
      </div>
      <div class="conf-bar-bg" style="margin-top:16px">
        <div class="conf-bar-fill" style="width:{conf}%;
             background:linear-gradient(90deg,{pcol},{pcol}dd)"></div>
      </div>
      <div class="signal-meta" style="margin-top:12px">
        <b>Logic:</b> {result['logic']}
      </div>
      <div class="signal-meta" style="font-size:0.8rem;color:#94a3b8">
        {result['stats']}
      </div>
    </div>""", unsafe_allow_html=True)

    st.session_state.pending = {
        "issue": st.session_state.last_issue_seen,
        "pred": result.get("pred", "WAIT"),
        "conf": result.get("final_confidence", 0),
        "votes": result.get("votes", {}),
    }


# ════════════════════════════════════════════════════════════
#  ALGO VOTES
# ════════════════════════════════════════════════════════════
if show_votes and result:
    st.markdown("## 🧠 Algorithm Votes")
    votes = result.get("votes", {})
    rows_html = []
    for algo in ALGO_ORDER:
        v = votes.get(algo)
        if not v:
            rows_html.append(
                f'<div class="algo-row">'
                f'<div class="algo-name">{algo}</div>'
                f'<div class="algo-vote">'
                f'<span class="pill pill-dim">—</span></div>'
                f'<div class="algo-conf">-</div>'
                f'<div class="algo-w">{ALGO_WEIGHTS[algo]}</div>'
                f'<div class="algo-rsn">no signal</div></div>')
            continue
        vp_cls = "pill-big" if v["pred"] == "Big" else "pill-small"
        rows_html.append(
            f'<div class="algo-row">'
            f'<div class="algo-name">{algo}</div>'
            f'<div class="algo-vote">'
            f'<span class="pill {vp_cls}">{v["pred"]}</span></div>'
            f'<div class="algo-conf">{v["conf"]}%</div>'
            f'<div class="algo-w">{v.get("w",1):.2f}</div>'
            f'<div class="algo-rsn">{v.get("reason","")}</div></div>')
    st.markdown('<div class="big-card" style="padding:10px 18px">'
                + "".join(rows_html) + '</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════════
if show_chart and PLOTLY_OK and len(df) >= 20:
    st.markdown("## 📈 Trends")
    ch1, ch2 = st.columns(2)

    with ch1:
        tail = df.tail(100).copy()
        tail["Big"] = (tail["Size"] == "Big").astype(int)
        tail["RollBig"] = tail["Big"].rolling(20, min_periods=5).mean() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(tail))), y=tail["RollBig"],
            mode="lines", line=dict(color="#4ade80", width=2)))
        fig.add_hline(y=50, line_dash="dash", line_color="#64748b")
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#cbd5e1"),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(gridcolor="#1e2436", range=[0, 100]),
            showlegend=False, title="Big % (20-round rolling)")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        nums = df.tail(50)["Number"].tolist()
        counts = [nums.count(i) for i in range(10)]
        colors = ["#f87171"] * 5 + ["#4ade80"] * 5
        fig2 = go.Figure(go.Bar(
            x=list(range(10)), y=counts,
            marker_color=colors, text=counts, textposition="outside"))
        fig2.update_layout(
            height=280, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#cbd5e1"),
            xaxis=dict(gridcolor="#1e2436", title="Number"),
            yaxis=dict(gridcolor="#1e2436", title="Count"),
            showlegend=False, title="Number distribution (last 50)")
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  HISTORY TABLE
# ════════════════════════════════════════════════════════════
if show_history and len(df):
    st.markdown("## 📜 Recent History")
    hist = df.tail(30).iloc[::-1].copy()
    hist["Size"] = hist["Size"].apply(
        lambda s: f"🟢 {s}" if s == "Big" else f"🔴 {s}")
    st.dataframe(hist[["Issue", "Number", "Size", "Time"]],
                 use_container_width=True, hide_index=True, height=360)


# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.divider()
st.caption("⚠️ WinGo draws are random/house-controlled. No guarantee "
           "of profit. For analysis/entertainment only.")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")