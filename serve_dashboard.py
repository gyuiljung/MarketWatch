#!/usr/bin/env python3
"""
Global Flow Monitor — 원커맨드 대시보드
1) yfinance 데이터 수집 + TE 계산 + JSON 생성
2) HTTP 서버 자동 시작 (포트 8080)
3) 브라우저 자동 열기

실행: python serve_dashboard.py
옵션: python serve_dashboard.py --port 8080 --no-open --skip-fetch
"""
import argparse
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

# Windows cp949 인코딩 문제 방지
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_DIR = Path(__file__).parent
SERVE_DIR = PROJECT_DIR / '_dashboard'
JSON_NAME = 'flow_monitor_latest.json'
HTML_NAME = 'global_flow_dashboard.html'


def run_pipeline():
    """전체 파이프라인 — global_flow_monitor.py 로직 재사용."""
    sys.path.insert(0, str(PROJECT_DIR))
    import global_flow_monitor as gfm
    import pandas as pd

    print("📡 데이터 수집 중...")
    category_data, errors, meta = gfm.fetch_all_data(days=400)
    summaries = gfm.compute_group_summaries(category_data)
    composite = gfm.generate_composite_signals(category_data)

    # TE (price + flow)
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
                    if n >= 15:
                        bins = 4 if n < 30 else 6 if n < 60 else 8
                        dst[label] = gfm.compute_group_te(g, bins=bins, max_lag=3, n_surrogates=50, top_n=8)

    group_te = {'price': te_price, 'flow': te_flow}
    buying_eff = gfm.compute_buying_efficiency()

    # JSON 생성
    gfm.export_json_v2(category_data, summaries, composite, meta, buying_eff, group_te=group_te)
    print("✅ JSON 생성 완료")


def prepare_serve_dir():
    """서빙 디렉토리에 HTML + JSON 복사."""
    SERVE_DIR.mkdir(exist_ok=True)
    shutil.copy2(PROJECT_DIR / HTML_NAME, SERVE_DIR / HTML_NAME)
    shutil.copy2(PROJECT_DIR / JSON_NAME, SERVE_DIR / JSON_NAME)
    print(f"📁 서빙 디렉토리: {SERVE_DIR}")


def start_server(port):
    """HTTP 서버 시작."""
    os.chdir(str(SERVE_DIR))
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # 로그 억제

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
        print(f"\n{'='*50}")
        print(f"  📊 대시보드: http://localhost:{port}/{HTML_NAME}")
        print(f"  종료: Ctrl+C")
        print(f"{'='*50}\n")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='Flow Monitor Dashboard')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--no-open', action='store_true', help='브라우저 자동 열기 비활성화')
    parser.add_argument('--skip-fetch', action='store_true', help='기존 JSON 재사용 (데이터 수집 건너뜀)')
    args = parser.parse_args()

    if not args.skip_fetch:
        run_pipeline()
    elif not (PROJECT_DIR / JSON_NAME).exists():
        print("⚠️  JSON 없음 — 데이터 수집 실행")
        run_pipeline()
    else:
        print("⏭️  기존 JSON 재사용 (--skip-fetch)")

    prepare_serve_dir()

    if not args.no_open:
        url = f"http://localhost:{args.port}/{HTML_NAME}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        start_server(args.port)
    except OSError as e:
        if 'Address already in use' in str(e) or '10048' in str(e):
            print(f"⚠️  포트 {args.port} 사용 중. --port 옵션으로 변경하세요.")
        else:
            raise


if __name__ == '__main__':
    main()
