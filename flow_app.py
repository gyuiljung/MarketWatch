#!/usr/bin/env python3
"""
Global Flow Monitor — Streamlit Wrapper
Streamlit static file serving으로 기존 HTML 대시보드를 그대로 서빙.
HTML은 fetch()로 JSON을 로드 — 기존 동작 100% 동일.

실행: streamlit run flow_app.py
"""
import json
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Global Flow Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import global_flow_monitor as gfm

PROJECT_DIR = Path(__file__).parent
STATIC_DIR = PROJECT_DIR / 'static'
STATIC_DIR.mkdir(exist_ok=True)


@st.cache_data(ttl=43200, show_spinner=False)
def run_full_pipeline():
    """파이프라인 실행 → JSON 생성 → static/ 에 복사."""
    # 1. Fetch
    category_data, errors, meta = gfm.fetch_all_data(days=400)
    summaries = gfm.compute_group_summaries(category_data)

    # 2. Signals
    composite = gfm.generate_composite_signals(category_data)

    # 3. TE (price + flow)
    close_series = meta.get('close_series', {})
    volume_series = meta.get('volume_series', {})
    te_windows = {'2W': 10, '1M': 21, '3M': 63, '6M': 126}
    te_price, te_flow = {}, {}

    if close_series:
        grp_ret = gfm.build_group_returns(close_series, window=300)
        grp_flow = (gfm.build_group_flow_returns(close_series, volume_series, window=300)
                    if volume_series else pd.DataFrame())
        for label, window in te_windows.items():
            if len(grp_ret.columns) >= 3:
                g = grp_ret.iloc[-window:] if len(grp_ret) >= window else grp_ret
                n = len(g)
                if n >= 15:
                    bins = 4 if n < 30 else 6 if n < 60 else 8
                    te_price[label] = gfm.compute_group_te(g, bins=bins, max_lag=3, n_surrogates=50, top_n=8)
            if len(grp_flow.columns) >= 3:
                g = grp_flow.iloc[-window:] if len(grp_flow) >= window else grp_flow
                n = len(g)
                if n >= 15:
                    bins = 4 if n < 30 else 6 if n < 60 else 8
                    te_flow[label] = gfm.compute_group_te(g, bins=bins, max_lag=3, n_surrogates=50, top_n=8)

    group_te = {'price': te_price, 'flow': te_flow}
    buying_eff = gfm.compute_buying_efficiency()

    # 4. 기존 export → JSON 파일 생성 (프로젝트 루트)
    gfm.export_json_v2(category_data, summaries, composite, meta, buying_eff, group_te=group_te)

    # 5. static/ 에 HTML + JSON 복사
    shutil.copy2(PROJECT_DIR / 'global_flow_dashboard.html',
                 STATIC_DIR / 'global_flow_dashboard.html')
    shutil.copy2(PROJECT_DIR / 'flow_monitor_latest.json',
                 STATIC_DIR / 'flow_monitor_latest.json')

    return True


# ── Main ──
status = st.empty()
status.info("📡 데이터 로딩 중... 첫 실행 시 2-3분, 이후 즉시")

run_full_pipeline()
status.empty()

# static/ 에서 서빙되는 HTML을 iframe으로 로드
# Streamlit static serving: /_app/static/{filename}
st.components.v1.iframe(
    src="/_app/static/global_flow_dashboard.html",
    height=2800,
    scrolling=True,
)
