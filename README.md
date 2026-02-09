# Global Flow Monitor

ETF 기반 글로벌 자금 흐름 모니터링 대시보드. 65개 ETF의 가격/거래대금 데이터로 시장 구조를 분석합니다.

## 주요 기능

- **Flow Scoreboard** — 5개 카테고리(Regional, US Sectors, Thematic, Risk/Macro, Bonds/FX) 실시간 성과
- **Dual Transfer Entropy** — 가격 수익률 TE + 거래대금(Dollar Volume) 흐름 TE (2W/1M/3M/6M)
- **Top Movers** — 상승/하락 상위 종목
- **Composite Signals** — Risk-On/Off, 안전자산 선호도 등 종합 시그널
- **Buying Efficiency** — 개인 매수 효율 지표

## 실행 방법

### 1. 단일 HTML 빌드 (공유용)

```bash
python build_standalone.py
```

- yfinance에서 최신 데이터 수집 → JSON 생성 → HTML에 인라인 임베딩
- `flow_dashboard_YYYYMMDD.html` 파일 하나 생성
- **서버 없이 브라우저에서 바로 열기 가능**
- 이 파일만 카톡/메일/슬랙으로 공유하면 됨

```bash
# 기존 데이터 재사용 (빠름)
python build_standalone.py --skip-fetch

# 출력 경로 지정
python build_standalone.py -o my_dashboard.html
```

### 2. 로컬 서버 (개발/실시간 확인용)

```bash
python serve_dashboard.py
```

- 데이터 수집 + HTTP 서버 시작 (기본 포트 8080)
- 브라우저 자동 열림: `http://localhost:8080/global_flow_dashboard.html`

```bash
# 기존 데이터로 서버만 시작
python serve_dashboard.py --skip-fetch

# 포트 변경
python serve_dashboard.py --port 9090
```

### 3. 데이터만 생성

```bash
python global_flow_monitor.py
```

- `flow_monitor_latest.json` + 카테고리별 CSV 생성
- `global_flow_dashboard.html`과 같은 폴더에서 `python -m http.server 8080`로 확인

## 데이터 업데이트 주기

- yfinance 기반 — 미국 장 마감 후 (한국 시간 오전 6시 이후) 최신 데이터 반영
- 주말/공휴일: 마지막 거래일 데이터 표시
- **매번 수동 실행 필요** (`build_standalone.py` 또는 `serve_dashboard.py`)

## Transfer Entropy (TE) 설명

그룹 간 정보 흐름의 방향과 강도를 측정합니다.

| 윈도우 | 기간 | Bins | 용도 |
|--------|------|------|------|
| 2W | 10일 | 3 | 최근 단기 동향 |
| 1M | 21일 | 4 | 단기 흐름 |
| 3M | 63일 | 8 | 중기 구조 |
| 6M | 126일 | 8 | 장기 구조 |

- **Price TE**: 가격 수익률 기반 정보 전이 (파란색 계열)
- **Flow TE**: 거래대금(Close x Volume) 변화율 기반 자금 흐름 전이 (시안색 계열)
- Leverage/Volatility 그룹은 TE 계산에서 제외 (bull+bear 평균 = 노이즈)

## 카테고리 구성 (65 ETFs)

| 카테고리 | 그룹 | 대표 ETF |
|----------|------|----------|
| Regional | US Broad, DM Broad, EM Broad, Asia, China, EM ex-China | SPY, EFA, EEM, EWJ, FXI, INDA |
| US Sectors | Tech, Financials, Healthcare, Energy, Industrials, Real Estate, Consumer | XLK, XLF, XLV, XLE, XLI, XLRE, XLY |
| Thematic | Semis, Innovation, Clean Energy, Biotech, Cybersecurity, Blockchain | SMH, ARKK, ICLN, XBI, CIBR, BITO |
| Risk/Macro | Safe Haven, Volatility, Leverage, Commodities | GLD, ^VIX, TQQQ/SQQQ/SOXL/SOXS, USO |
| Bonds/FX | US Treasury, IG Credit, HY Credit, EM Debt, FX | TLT, LQD, HYG, EMB, UUP |

## 파일 구조

```
MarketWatch/
  global_flow_monitor.py    # 핵심 엔진 (데이터 수집 + 분석 + JSON 생성)
  global_flow_dashboard.html # 대시보드 UI (fetch로 JSON 로드)
  build_standalone.py        # 단일 HTML 빌더 (JSON 인라인 임베딩)
  serve_dashboard.py         # 로컬 HTTP 서버
  flow_app.py                # Streamlit 래퍼
  flow_monitor_latest.json   # 생성된 데이터 (gitignore)
```

## 요구사항

```
pip install yfinance pandas numpy
```
