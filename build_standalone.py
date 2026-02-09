#!/usr/bin/env python3
"""
단일 HTML 파일 빌드 — JSON을 인라인으로 임베딩.
파일 하나만 공유하면 브라우저에서 바로 열림 (서버 불필요).

실행: python build_standalone.py
옵션: python build_standalone.py --skip-fetch  (기존 JSON 재사용)
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_DIR = Path(__file__).parent
HTML_SRC = PROJECT_DIR / 'global_flow_dashboard.html'
JSON_SRC = PROJECT_DIR / 'flow_monitor_latest.json'


def run_pipeline():
    """데이터 수집 + JSON 생성."""
    sys.path.insert(0, str(PROJECT_DIR))
    import global_flow_monitor as gfm
    import pandas as pd

    print("데이터 수집 중...")
    category_data, errors, meta = gfm.fetch_all_data(days=400)
    summaries = gfm.compute_group_summaries(category_data)
    composite = gfm.generate_composite_signals(category_data)

    close_series = meta.get('close_series', {})
    volume_series = meta.get('volume_series', {})
    te_windows = {'2W': 10, '1M': 21, '3M': 63, '6M': 126}
    te_price, te_flow = {}, {}

    if close_series:
        grp_ret = gfm.build_group_returns(close_series, window=300)
        grp_flow = (gfm.build_group_flow_returns(close_series, volume_series, window=300)
                    if volume_series else pd.DataFrame())
        for label, window in te_windows.items():
            for src, dst in [(grp_ret, te_price), (grp_flow, te_flow)]:
                if len(src.columns) >= 3:
                    g = src.iloc[-window:] if len(src) >= window else src
                    n = len(g)
                    if n >= 10:
                        bins = 3 if n < 15 else 4 if n < 30 else 6 if n < 60 else 8
                        dst[label] = gfm.compute_group_te(g, bins=bins, max_lag=2, n_surrogates=50, top_n=8)

    group_te = {'price': te_price, 'flow': te_flow}
    buying_eff = gfm.compute_buying_efficiency()
    gfm.export_json_v2(category_data, summaries, composite, meta, buying_eff, group_te=group_te)
    print("JSON 생성 완료")


def build_standalone(output_path=None):
    """HTML + JSON → 단일 HTML 파일."""
    html = HTML_SRC.read_text(encoding='utf-8')
    json_text = JSON_SRC.read_text(encoding='utf-8')

    # fetch() 블록을 인라인 JSON으로 교체
    inline_js = f"""// ===================== MAIN: Inline JSON =====================
const data = {json_text};
(function(data) {{
    showTimestamp(data.generated_at, data.data_date);

    if (data.categories) {{
      renderFlowScoreboard(data.categories);
      renderCategoryDetail(data.categories);
    }}

    if (data.top_movers) renderTopMovers(data.top_movers);
    if (data.kpi) renderKPI(data.kpi);
    renderSignals(data.signals);
    if (data.group_te) renderGroupTE(data.group_te);
    renderBuyingEfficiency(data.buying_efficiency);

    // Footer
    const footer = document.getElementById('footer-note');
    if (footer) {{
      footer.innerHTML = `데이터 연동 완료 | ${{data.generated_at || ''}} | ${{data.data_date || 'N/A'}}<br>` + footer.innerHTML;
    }}
}})(data);"""

    # fetch 블록 전체를 교체 (// MAIN 주석부터 끝 catch까지)
    pattern = r'// =+ MAIN: Load JSON =+\n.*?\.catch\([^}]*\{[^}]*\}[^)]*\);'
    new_html = re.sub(pattern, inline_js, html, flags=re.DOTALL)

    if new_html == html:
        print("WARNING: fetch 패턴 매칭 실패, 수동 교체 시도")
        old_block = "fetch('./flow_monitor_latest.json?t=' + Date.now())"
        if old_block in html:
            # 전체 fetch~catch 블록 찾기
            start = html.index("// ===================== MAIN: Load JSON")
            end = html.index("});", html.index(".catch(", start)) + 3
            new_html = html[:start] + inline_js + html[end:]

    if output_path is None:
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = PROJECT_DIR / f'flow_dashboard_{date_str}.html'

    Path(output_path).write_text(new_html, encoding='utf-8')
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"=> {output_path} ({size_kb:.0f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Build standalone HTML dashboard')
    parser.add_argument('--skip-fetch', action='store_true', help='기존 JSON 재사용')
    parser.add_argument('-o', '--output', type=str, help='출력 파일 경로')
    args = parser.parse_args()

    if not args.skip_fetch:
        run_pipeline()
    elif not JSON_SRC.exists():
        print("JSON 없음 - 데이터 수집 실행")
        run_pipeline()
    else:
        print("기존 JSON 재사용 (--skip-fetch)")

    out = build_standalone(args.output)
    print(f"\n파일 하나만 공유하면 됩니다: {out}")
    print("브라우저에서 바로 열기 가능 (서버 불필요)")


if __name__ == '__main__':
    main()
