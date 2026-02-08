#!/usr/bin/env python3
"""
글로벌 플로우 모니터 v2.0
========================
글로벌 자금 흐름 히트맵 대시보드.

v2.0 변경사항:
    - 5개 카테고리 (~47 티커): US Sector, Regional, Thematic, Risk/Macro, Bonds/FX
    - 그룹 헤더 + 하위 티커 계층 구조
    - yf.download() 배치 수집 (순차 60s → 배치 10-15s)
    - 히트맵 스타일 JSON v2 → HTML 대시보드 연동
    - 새 시그널: Credit Spread (HYG-LQD), US Exceptionalism (SPY vs EFA)
    - 콘솔: 그룹 평균 서머리 테이블 (--detail 로 개별 티커)

v1.4 변경사항:
    - DI-04: --export 시 JSON 파일 생성 → HTML 대시보드 자동 연동
    - IA-08: pykrx 기반 개인 매수 효율(Buying Efficiency) 자동 계산

설치:
    pip install yfinance pandas tabulate requests

실행:
    python global_flow_monitor.py              # 전체 대시보드 (그룹 서머리)
    python global_flow_monitor.py --detail     # 개별 티커 전체 출력
    python global_flow_monitor.py --export     # JSON v2 + CSV 내보내기
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import argparse
import sys
import os
import json

# IA-08: pykrx — optional (개인 매수 효율 계산용)
try:
    from pykrx import stock as krx_stock
    HAS_PYKRX = True
except ImportError:
    HAS_PYKRX = False

# ============================================================
# 설정: 5개 카테고리 티커 구성 (v2.0)
# ============================================================

CATEGORIES = {
    'us_sectors': {
        'label': 'US Sectors (GICS 11)',
        'icon': '🏛️',
        'groups': {
            'Tech': {
                'XLK': {'name': 'Technology'},
                'XLC': {'name': 'Communication'},
            },
            'Cyclical': {
                'XLF': {'name': 'Financials'},
                'XLY': {'name': 'Consumer Disc.'},
                'XLI': {'name': 'Industrials'},
            },
            'Defensive': {
                'XLV': {'name': 'Health Care'},
                'XLP': {'name': 'Consumer Staples'},
                'XLU': {'name': 'Utilities'},
            },
            'Commodity': {
                'XLE': {'name': 'Energy'},
                'XLB': {'name': 'Materials'},
            },
            'Rate-Sensitive': {
                'XLRE': {'name': 'Real Estate'},
            },
        },
    },
    'regional': {
        'label': 'Regional Flow',
        'icon': '🌍',
        'groups': {
            'DM Americas': {
                'SPY': {'name': 'S&P 500'},
            },
            'DM Europe': {
                'EFA': {'name': 'EAFE (Europe/Asia)'},
                'EWG': {'name': 'MSCI Germany'},
            },
            'DM Asia': {
                'EWJ': {'name': 'MSCI Japan'},
            },
            'EM Asia': {
                'FXI': {'name': 'China Large Cap'},
                'EWY': {'name': 'MSCI Korea'},
                'EWT': {'name': 'MSCI Taiwan'},
                'INDA': {'name': 'MSCI India'},
            },
            'EM Latam': {
                'EWZ': {'name': 'MSCI Brazil'},
            },
            'EM Broad': {
                'EEM': {'name': 'iShares EM'},
                'VWO': {'name': 'Vanguard EM'},
            },
        },
    },
    'thematic': {
        'label': 'Thematic / Growth',
        'icon': '🚀',
        'groups': {
            'Semiconductor': {
                'SMH': {'name': 'VanEck Semi'},
                'SOXX': {'name': 'iShares Semi'},
            },
            'Biotech': {
                'XBI': {'name': 'Biotech SPDR'},
            },
            'Disruptive': {
                'ARKK': {'name': 'ARK Innovation'},
            },
            'China Tech': {
                'KWEB': {'name': 'China Internet'},
            },
            'Leverage': {
                'SOXL': {'name': 'Semi 3x Bull'},
                'SOXS': {'name': 'Semi 3x Bear'},
            },
        },
    },
    'risk_macro': {
        'label': 'Risk / Macro',
        'icon': '⚡',
        'groups': {
            'Volatility': {
                '^VIX': {'name': 'VIX'},
            },
            'Safe Haven': {
                'GLD': {'name': 'Gold'},
                'TLT': {'name': '20Y+ Treasury'},
                'SHY': {'name': '1-3Y Treasury'},
            },
            'Commodities': {
                'USO': {'name': 'WTI Oil'},
            },
            'FX': {
                'UUP': {'name': 'USD Index'},
                'FXY': {'name': 'Yen ETF'},
            },
            'Credit': {
                'HYG': {'name': 'High Yield Corp'},
                'LQD': {'name': 'Inv Grade Corp'},
            },
            'Inflation': {
                'TIP': {'name': 'TIPS Bond'},
            },
        },
    },
    'bonds_fx': {
        'label': 'Bonds & FX',
        'icon': '🏦',
        'groups': {
            'Aggregate': {
                'AGG': {'name': 'US Agg Bond'},
            },
        },
    },
}

# 필수 티커 — 복합 시그널 판단에 반드시 필요
REQUIRED_TICKERS = [
    'SMH', 'SOXX',       # 반도체 로테이션
    'EWY', 'EEM',         # EM 상대 성과
    '^VIX', 'GLD', 'USO', 'UUP', 'FXY',  # 리스크 시그널
    'HYG', 'LQD',         # 크레딧 스프레드
    'SPY', 'EFA',          # US Exceptionalism
]


def get_all_tickers():
    """CATEGORIES에서 전체 고유 티커 목록 추출."""
    tickers = {}  # ticker -> {name, category, group}
    for cat_key, cat in CATEGORIES.items():
        for grp_name, grp_tickers in cat['groups'].items():
            for ticker, info in grp_tickers.items():
                if ticker not in tickers:
                    tickers[ticker] = {
                        'name': info['name'],
                        'category': cat_key,
                        'group': grp_name,
                        'memberships': [],
                    }
                tickers[ticker]['memberships'].append((cat_key, grp_name))
    return tickers


def check_required_tickers(fetched_set):
    """필수 티커 수집 여부 확인."""
    return [t for t in REQUIRED_TICKERS if t not in fetched_set]


def print_missing_warning(missing):
    """필수 티커 누락 시 상단 경고."""
    if not missing:
        return
    width = 70
    print("\n" + "!" * width)
    print("  ⛔ 필수 데이터 누락 경고 — 복합 시그널 신뢰도 저하")
    print("!" * width)
    for t in missing:
        print(f"  ❌ {t} — 데이터 수집 실패")
    print("  → 해당 티커에 의존하는 시그널이 평가되지 않았습니다.")
    print("!" * width)


# ============================================================
# 데이터 수집 — 배치 (v2.0)
# ============================================================

def fetch_all_data(days=35):
    """yf.download() 배치 수집 → 카테고리별 분할 반환.

    Returns:
        (category_data, errors, meta)
        category_data: {cat_key: DataFrame}  — 각 DF에 Ticker,Name,Group,1D%,5D%,...
        errors: list of failed tickers
        meta: {'last_date', 'first_date', 'd5_refs'}
    """
    all_tickers = get_all_tickers()
    ticker_list = list(all_tickers.keys())

    end = datetime.now()
    start = end - timedelta(days=days)

    meta = {'last_date': None, 'first_date': None, 'd5_refs': {}}
    errors = []

    # 배치 다운로드
    print(f"  📡 {len(ticker_list)}개 티커 배치 수집 중...")
    try:
        raw = yf.download(
            ticker_list,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            group_by='ticker',
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"  ⚠️  배치 다운로드 실패: {e}")
        return {}, ticker_list, meta

    # 개별 티커 처리
    rows = []  # (ticker, row_dict)
    for ticker in ticker_list:
        try:
            # 단일 티커일 때와 복수 티커일 때 컬럼 구조가 다름
            if len(ticker_list) == 1:
                hist = raw
            else:
                if ticker not in raw.columns.get_level_values(0):
                    errors.append(ticker)
                    continue
                hist = raw[ticker].dropna(how='all')

            if hist is None or len(hist) < 2:
                errors.append(ticker)
                continue

            close = hist['Close'].dropna()
            if len(close) < 2:
                errors.append(ticker)
                continue

            last = close.iloc[-1]
            prev = close.iloc[-2]
            first = close.iloc[0]

            # 날짜 추적
            last_date = close.index[-1]
            if meta['last_date'] is None or last_date > meta['last_date']:
                meta['last_date'] = last_date
            first_date = close.index[0]
            if meta['first_date'] is None or first_date < meta['first_date']:
                meta['first_date'] = first_date

            # 5D 참조
            if len(close) >= 6:
                d5_price = close.iloc[-6]
                d5_ref_date = close.index[-6].strftime('%m-%d')
            else:
                d5_price = first
                d5_ref_date = close.index[0].strftime('%m-%d') + '*'
            meta['d5_refs'][ticker] = d5_ref_date

            # 수익률
            d1_ret = round((last - prev) / prev * 100, 2)
            d5_ret = round((last - d5_price) / d5_price * 100, 2)
            d30_ret = round((last - first) / first * 100, 2)

            # Z-score
            z_5d = None
            if len(close) >= 10:
                rolling_5d = close.pct_change(5) * 100
                rolling_5d = rolling_5d.dropna()
                if len(rolling_5d) >= 5 and rolling_5d.std() > 0:
                    z_5d = round((d5_ret - rolling_5d.mean()) / rolling_5d.std(), 2)

            # 거래량 변화 + 평균 일일 달러 거래량
            vol = hist.get('Volume')
            vol_change = 0
            avg_dol_vol = 0  # 평균 일일 달러 거래량 ($)
            if vol is not None and len(vol) >= 5:
                vol_recent = vol.iloc[-5:].mean()
                close_recent = close.iloc[-5:].mean()
                avg_dol_vol = round(float(vol_recent * close_recent), 0)
                vol_prior = vol.iloc[-10:-5].mean() if len(vol) >= 10 else vol_recent
                if vol_prior > 0:
                    vol_change = round((vol_recent - vol_prior) / vol_prior * 100, 0)
            elif vol is not None and len(vol) >= 2:
                avg_dol_vol = round(float(vol.iloc[-2:].mean() * close.iloc[-2:].mean()), 0)

            info = all_tickers[ticker]
            row = {
                'Ticker': ticker,
                'Name': info['name'],
                'Last': round(float(last), 2),
                '1D %': d1_ret,
                '5D %': d5_ret,
                '5D Ref': d5_ref_date,
                '5D Z': z_5d,
                '30D %': d30_ret,
                'Vol Δ%': vol_change,
                'AvgDolVol': avg_dol_vol,
            }
            rows.append((ticker, row))

        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:50]}")

    # 카테고리별 분할
    category_data = {}
    for cat_key, cat in CATEGORIES.items():
        cat_rows = []
        for grp_name, grp_tickers in cat['groups'].items():
            for ticker in grp_tickers:
                for t, row in rows:
                    if t == ticker:
                        r = dict(row)
                        r['Group'] = grp_name
                        cat_rows.append(r)
                        break
        if cat_rows:
            category_data[cat_key] = pd.DataFrame(cat_rows)

    return category_data, errors, meta


# ============================================================
# 그룹 서머리 (v2.0)
# ============================================================

def compute_top_movers(category_data, n=5):
    """전체 티커 중 5D% 기준 top/bottom N 추출.

    제외: ^VIX (지표), SOXL/SOXS (레버리지/인버스 — 왜곡)

    Returns:
        {'gainers': [dict, ...], 'losers': [dict, ...]}
    """
    EXCLUDE = {'^VIX', 'SOXL', 'SOXS'}

    all_rows = []
    # cat_key → (icon, label) 매핑
    cat_meta = {k: (v['icon'], v['label']) for k, v in CATEGORIES.items()}

    for cat_key, df in category_data.items():
        icon, label = cat_meta.get(cat_key, ('', ''))
        for _, row in df.iterrows():
            ticker = row['Ticker']
            if ticker in EXCLUDE:
                continue
            d5 = row.get('5D %')
            if d5 is None or pd.isna(d5):
                continue
            all_rows.append({
                'Ticker': ticker,
                'Name': row.get('Name', ''),
                '5D %': round(float(d5), 2),
                '1D %': round(float(row.get('1D %', 0)), 2) if not pd.isna(row.get('1D %')) else None,
                '30D %': round(float(row.get('30D %', 0)), 2) if not pd.isna(row.get('30D %')) else None,
                'group': row.get('Group', ''),
                'cat_icon': icon,
            })

    if not all_rows:
        return {'gainers': [], 'losers': []}

    sorted_rows = sorted(all_rows, key=lambda r: r['5D %'], reverse=True)
    gainers = sorted_rows[:n]
    losers = sorted_rows[-n:][::-1]  # worst first (가장 나쁜 것부터)

    return {'gainers': gainers, 'losers': losers}


def compute_group_summaries(category_data):
    """카테고리별 그룹 평균 수익률 + 추정 편출입 계산.

    Returns:
        {cat_key: {group_name: {
            '1D %': avg, '5D %': avg, '30D %': avg,
            'trading_impact': 추정 5D 편출입 ($, 양수=유입/음수=유출),
            'avg_dol_vol': 그룹 일평균 달러거래량 합계 ($)
        }}}

    est_flow = Σ(ticker_AvgDolVol) × 5 × (group_5D% / 100)
    → 가격×거래량 기반 추정. 실제 ETF 펀드플로우와 다름.
    """
    summaries = {}
    for cat_key, df in category_data.items():
        cat_summary = {}
        for grp_name in df['Group'].unique():
            grp = df[df['Group'] == grp_name]
            d5_avg = round(grp['5D %'].mean(), 2)
            grp_dol_vol = grp['AvgDolVol'].sum() if 'AvgDolVol' in grp.columns else 0
            est_flow = round(grp_dol_vol * 5 * (d5_avg / 100), 0)
            cat_summary[grp_name] = {
                '1D %': round(grp['1D %'].mean(), 2),
                '5D %': d5_avg,
                '30D %': round(grp['30D %'].mean(), 2),
                'trading_impact': est_flow,
                'avg_dol_vol': round(grp_dol_vol, 0),
            }
        summaries[cat_key] = cat_summary
    return summaries


# ============================================================
# 시그널 (v2.0 — 기존 4개 + 새로운 2개)
# ============================================================

def compute_rotation_score(category_data):
    """섹터 로테이션 강도: Defensive - Tech/Semi.

    v2.0: 새 그룹명 기반. Leverage 그룹 제외.
    """
    # Semiconductor from thematic
    thematic = category_data.get('thematic')
    if thematic is None:
        return None

    semi = thematic[thematic['Group'] == 'Semiconductor']
    # Defensive from us_sectors
    sectors = category_data.get('us_sectors')
    if sectors is None:
        return None

    defensive = sectors[sectors['Group'] == 'Defensive']

    if len(semi) == 0 or len(defensive) == 0:
        return None

    semi_avg = semi['5D %'].mean()
    defensive_avg = defensive['5D %'].mean()
    score = round(defensive_avg - semi_avg, 2)

    # Leverage 포함 비교 (검증용)
    semi_all = thematic[thematic['Group'].isin(['Semiconductor', 'Leverage'])]
    semi_all_no_bear = semi_all[~semi_all['Ticker'].isin(['SOXS'])]
    score_meta = None
    if len(semi_all_no_bear) > len(semi):
        raw = round(defensive_avg - semi_all_no_bear['5D %'].mean(), 2)
        dev = abs(raw - score) / abs(score) * 100 if score != 0 else 0
        score_meta = {'raw_with_leverage': raw, 'deviation_pct': round(dev, 0)}

    return score, score_meta


def compute_em_relative(category_data):
    """EWY vs EEM 상대 성과 (IA-03 보정 유지)."""
    EWY_WEIGHT_IN_EEM = 0.12
    regional = category_data.get('regional')
    if regional is None:
        return None

    korea = regional[regional['Ticker'] == 'EWY']
    em_broad = regional[regional['Ticker'] == 'EEM']
    taiwan = regional[regional['Ticker'] == 'EWT']

    relatives = []
    if len(korea) > 0 and len(em_broad) > 0:
        ewy_5d = korea['5D %'].values[0]
        eem_5d = em_broad['5D %'].values[0]
        ewy_30d = korea['30D %'].values[0]
        eem_30d = em_broad['30D %'].values[0]

        eem_ex_kr_5d = (eem_5d - ewy_5d * EWY_WEIGHT_IN_EEM) / (1 - EWY_WEIGHT_IN_EEM)
        eem_ex_kr_30d = (eem_30d - ewy_30d * EWY_WEIGHT_IN_EEM) / (1 - EWY_WEIGHT_IN_EEM)

        diff_5d_raw = ewy_5d - eem_5d
        diff_5d_adj = ewy_5d - eem_ex_kr_5d
        diff_30d_adj = ewy_30d - eem_ex_kr_30d

        interp = '한국 EM 대비 강세' if diff_5d_adj > 0 else '한국 EM 대비 약세'
        relatives.append({
            'Pair': 'EWY vs EEM(보정)',
            '5D 상대%': round(diff_5d_adj, 2),
            '30D 상대%': round(diff_30d_adj, 2),
            '해석': f'{interp} (보정 전 {diff_5d_raw:+.1f}%)'
        })

    if len(korea) > 0 and len(taiwan) > 0:
        diff_5d = korea['5D %'].values[0] - taiwan['5D %'].values[0]
        diff_30d = korea['30D %'].values[0] - taiwan['30D %'].values[0]
        relatives.append({
            'Pair': 'EWY vs EWT',
            '5D 상대%': round(diff_5d, 2),
            '30D 상대%': round(diff_30d, 2),
            '해석': '한국 대만 대비 강세' if diff_5d > 0 else '한국 대만 대비 약세 (반도체 비중↑)'
        })

    return pd.DataFrame(relatives) if relatives else None


def _get_ticker_row(category_data, ticker):
    """카테고리 데이터에서 특정 티커 행 찾기."""
    for df in category_data.values():
        match = df[df['Ticker'] == ticker]
        if len(match) > 0:
            return match.iloc[0]
    return None


def compute_risk_dashboard(category_data):
    """리스크 상태 종합 판단 — Z-score 적응형."""
    risk_df = category_data.get('risk_macro')
    if risk_df is None:
        return []

    signals = []
    for _, row in risk_df.iterrows():
        ticker = row['Ticker']
        d5 = row['5D %']
        z = row.get('5D Z')
        z_tag = f" Z={z:+.1f}" if z is not None else ""

        if ticker == '^VIX':
            level = row['Last']
            if z is not None and z > 2:
                signals.append(('🔴', f"VIX {level:.0f} — 30일 대비 급등{z_tag}"))
            elif level > 25:
                signals.append(('🔴', f"VIX {level:.0f} — 공포 구간{z_tag}"))
            elif z is not None and z > 1:
                signals.append(('🟡', f"VIX {level:.0f} — 30일 대비 상승{z_tag}"))
            elif level > 20:
                signals.append(('🟡', f"VIX {level:.0f} — 경계 구간{z_tag}"))
            else:
                signals.append(('🟢', f"VIX {level:.0f} — 안정{z_tag}"))

        elif ticker == 'GLD':
            if z is not None and z > 2:
                signals.append(('🔴', f"금 5D {d5:+.1f}% — 30일 대비 이상 급등{z_tag}"))
            elif d5 > 2:
                signals.append(('🔴', f"금 5D {d5:+.1f}% — 안전자산 수요 급증{z_tag}"))
            elif z is not None and z > 1:
                signals.append(('🟡', f"금 5D {d5:+.1f}% — 30일 대비 상승{z_tag}"))
            elif d5 > 0.5:
                signals.append(('🟡', f"금 5D {d5:+.1f}% — 안전자산 소폭 선호{z_tag}"))

        elif ticker == 'USO':
            if z is not None and z > 2:
                signals.append(('🔴', f"유가 5D {d5:+.1f}% — 30일 대비 이상 급등{z_tag}"))
            elif d5 > 5:
                signals.append(('🔴', f"유가 5D {d5:+.1f}% — 지정학/공급 리스크{z_tag}"))
            elif z is not None and z < -2:
                signals.append(('🟢', f"유가 5D {d5:+.1f}% — 30일 대비 이상 급락{z_tag}"))
            elif d5 < -5:
                signals.append(('🟢', f"유가 5D {d5:+.1f}% — 수요 약화 우려{z_tag}"))

        elif ticker == 'UUP':
            if z is not None and z > 2:
                signals.append(('🔴', f"달러 5D {d5:+.1f}% — 30일 대비 이상 강세{z_tag}"))
            elif d5 > 1:
                signals.append(('🔴', f"달러 5D {d5:+.1f}% — EM 자금유출 압력{z_tag}"))
            elif z is not None and z < -2:
                signals.append(('🟢', f"달러 5D {d5:+.1f}% — 30일 대비 이상 약세{z_tag}"))
            elif d5 < -1:
                signals.append(('🟢', f"달러 5D {d5:+.1f}% — EM 자금유입 우호적{z_tag}"))

        elif ticker == 'FXY':
            if z is not None and z > 2:
                signals.append(('🟡', f"엔화 5D {d5:+.1f}% — 30일 대비 이상 강세{z_tag}"))
            elif d5 > 2:
                signals.append(('🟡', f"엔화 5D {d5:+.1f}% — 캐리트레이드 청산 주의{z_tag}"))

    return signals


def generate_composite_signals(category_data):
    """복합 시그널 생성 — 기존 4개 + 새 2개."""
    signals = []

    # Helper
    def get_row(ticker):
        return _get_ticker_row(category_data, ticker)

    # 1. 반도체 로테이션 + 달러 강세 = 한국 매도 압력
    rot_result = compute_rotation_score(category_data)
    rot_score = rot_result[0] if rot_result is not None else None
    uup = get_row('UUP')

    if rot_score is not None and rot_score > 3 and uup is not None and uup['5D %'] > 0.5:
        signals.append({
            'Level': '🔴 HIGH',
            'Signal': '반도체 로테이션 + 달러 강세 동시 발생',
            'Implication': '외인 한국 현물 순매도 가속 예상. 선물 숏 동반 가능.',
            'Data': f'로테이션 스코어: {rot_score}, 달러 5D: +{uup["5D %"]:.1f}%'
        })

    # 2. EM 유입 + 한국 언더퍼폼 = 반도체 기피
    em_rel = compute_em_relative(category_data)
    if em_rel is not None and len(em_rel) > 0:
        eem_row = em_rel[em_rel['Pair'] == 'EWY vs EEM(보정)']
        if len(eem_row) > 0 and eem_row['5D 상대%'].values[0] < -2:
            signals.append({
                'Level': '🟡 MED',
                'Signal': 'EM 자금 유입 중이지만 한국은 소외',
                'Implication': '글로벌 EM 로테이션에서 한국=반도체 인식으로 비중 축소.',
                'Data': f'EWY vs EEM(보정) 5D: {eem_row["5D 상대%"].values[0]:+.1f}%'
            })

    # 3. 금+유가 동시 급등 — IA-05 교차 분류
    gld = get_row('GLD')
    uso = get_row('USO')
    if gld is not None and uso is not None:
        gld_5d = gld['5D %']
        uso_5d = uso['5D %']
        if gld_5d > 1.5 and uso_5d > 3:
            tlt = get_row('TLT')
            uup_r = get_row('UUP')
            fxi = get_row('FXI')
            tlt_5d = tlt['5D %'] if tlt is not None else 0
            uup_5d = uup_r['5D %'] if uup_r is not None else 0
            fxi_5d = fxi['5D %'] if fxi is not None else 0

            cross = f'TLT 5D: {tlt_5d:+.1f}%, USD 5D: {uup_5d:+.1f}%, FXI 5D: {fxi_5d:+.1f}%'
            if tlt_5d > 1 and uup_5d > 0:
                signals.append({
                    'Level': '🔴 HIGH',
                    'Signal': '금 + 유가 동시 상승 — 지정학 리스크 (TLT 동반 상승 확인)',
                    'Implication': '안전자산 동반 상승 — 리스크오프 확인. 중동/대만 긴장 점검 필요.',
                    'Data': f'금 5D: +{gld_5d:.1f}%, 유가 5D: +{uso_5d:.1f}% | {cross}'
                })
            elif uup_5d < -0.5:
                signals.append({
                    'Level': '🟡 MED',
                    'Signal': '금 + 유가 동시 상승 — 달러 약세 주도 (명목 상승)',
                    'Implication': '달러 약세가 원자재 가격을 밀어올림. EM 자금유입에는 우호적.',
                    'Data': f'금 5D: +{gld_5d:.1f}%, 유가 5D: +{uso_5d:.1f}% | {cross}'
                })
            elif fxi_5d > 2:
                signals.append({
                    'Level': '🟡 MED',
                    'Signal': '금 + 유가 동시 상승 — 중국 경기부양 기대 (FXI 동반 상승)',
                    'Implication': 'FXI 강세 동반 — 중국 부양책 기대. 한국 수출주에 긍정적 가능성.',
                    'Data': f'금 5D: +{gld_5d:.1f}%, 유가 5D: +{uso_5d:.1f}% | {cross}'
                })
            else:
                signals.append({
                    'Level': '🟡 MED',
                    'Signal': '금 + 유가 동시 상승 — 인플레이션 기대 상승',
                    'Implication': '안전자산 동반 없이 원자재만 상승. 인플레 기대 우세, 금리 경로 주시.',
                    'Data': f'금 5D: +{gld_5d:.1f}%, 유가 5D: +{uso_5d:.1f}% | {cross}'
                })

    # 4. VIX 급등 + 엔화 강세 = 캐리트레이드 청산
    vix = get_row('^VIX')
    fxy = get_row('FXY')
    if vix is not None and fxy is not None:
        vix_val = vix['Last']
        fxy_5d = fxy['5D %']
        data_str = f'VIX: {vix_val:.0f}, 엔화 5D: {fxy_5d:+.1f}%'
        if vix_val > 28 and fxy_5d > 3:
            signals.append({
                'Level': '🔴 HIGH',
                'Signal': 'VIX 급등 + 엔화 급등 — 캐리트레이드 청산 진행',
                'Implication': '엔 캐리 청산 확인 수준. 한국 포함 EM 전반 자금 이탈.',
                'Data': data_str
            })
        elif vix_val > 22 and fxy_5d > 1.5:
            signals.append({
                'Level': '🟡 MED',
                'Signal': 'VIX 상승 + 엔화 강세 — 캐리트레이드 청산 주의',
                'Implication': '아직 청산 확인 수준 아님 (HIGH: VIX 28+, FXY +3%). 모니터링.',
                'Data': data_str
            })

    # 5. [NEW] Credit Spread: HYG-LQD 5D 스프레드
    hyg = get_row('HYG')
    lqd = get_row('LQD')
    if hyg is not None and lqd is not None:
        spread_5d = hyg['5D %'] - lqd['5D %']
        if spread_5d < -1.5:
            signals.append({
                'Level': '🔴 HIGH',
                'Signal': f'크레딧 스프레드 확대 — HYG vs LQD 5D: {spread_5d:+.1f}%',
                'Implication': '하이일드 급락 / 투자등급 상대 강세 → 신용 리스크 확대. 위험 자산 전반 경계.',
                'Data': f'HYG 5D: {hyg["5D %"]:+.1f}%, LQD 5D: {lqd["5D %"]:+.1f}%'
            })
        elif spread_5d < -0.8:
            signals.append({
                'Level': '🟡 MED',
                'Signal': f'크레딧 스프레드 소폭 확대 — HYG vs LQD 5D: {spread_5d:+.1f}%',
                'Implication': '하이일드 약세 시작. 추세 지속 시 리스크오프 전환 가능.',
                'Data': f'HYG 5D: {hyg["5D %"]:+.1f}%, LQD 5D: {lqd["5D %"]:+.1f}%'
            })

    # 6. [NEW] US Exceptionalism: SPY vs EFA
    spy = get_row('SPY')
    efa = get_row('EFA')
    if spy is not None and efa is not None:
        us_ex_5d = spy['5D %'] - efa['5D %']
        if us_ex_5d > 3:
            signals.append({
                'Level': '🟡 MED',
                'Signal': f'US Exceptionalism — SPY vs EFA 5D: +{us_ex_5d:.1f}%',
                'Implication': '미국 독주 → 비미국 자산에서 자금 이탈 압력. EM 로테이션 리스크.',
                'Data': f'SPY 5D: {spy["5D %"]:+.1f}%, EFA 5D: {efa["5D %"]:+.1f}%'
            })

    # 시그널 부재 시 커버리지 명시
    if not signals:
        signals.append({
            'Level': '🟢 LOW',
            'Signal': '주요 복합 경고 시그널 없음 (6/6 조건 미해당)',
            'Implication': '감시 중인 6개 시나리오 해당 없음. '
                          '미감시 영역: 위안화, 한국 정치/대북, 글로벌 유동성.',
            'Data': ''
        })

    return signals


# ============================================================
# IA-08: 개인 매수 효율 (Buying Efficiency) — 그대로 유지
# ============================================================

def compute_buying_efficiency(date_str=None):
    """KRX 데이터 기반 개인 매수 효율 계산."""
    if not HAS_PYKRX:
        return None
    try:
        if date_str is None:
            today = datetime.now()
            end_dt = today.strftime('%Y%m%d')
            start_dt = (today - timedelta(days=7)).strftime('%Y%m%d')
        else:
            end_dt = date_str.replace('-', '')
            start_dt = (datetime.strptime(end_dt, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')

        kospi = krx_stock.get_index_ohlcv(start_dt, end_dt, "1001")
        if kospi is None or len(kospi) < 2:
            return {'error': '코스피 데이터 부족', 'date': end_dt}

        last_date = kospi.index[-1]
        kospi_close = kospi['종가'].iloc[-1]
        kospi_prev = kospi['종가'].iloc[-2]
        kospi_change = round(kospi_close - kospi_prev, 2)

        trade_date = last_date.strftime('%Y%m%d')
        trading = krx_stock.get_market_trading_value_by_investor(
            trade_date, trade_date, "KOSPI"
        )
        if trading is None or len(trading) == 0:
            return {'error': '투자자 매매 데이터 없음', 'date': trade_date}

        if '개인' in trading.index:
            individual_net = trading.loc['개인', '순매수']
            individual_net_trillion = round(individual_net / 1_000_000_000_000, 2)
        else:
            return {'error': '개인 투자자 데이터 없음', 'date': trade_date}

        if abs(individual_net_trillion) < 0.01:
            efficiency = None
        else:
            efficiency = round(kospi_change / individual_net_trillion, 1)

        foreign_net_trillion = None
        if '외국인' in trading.index:
            foreign_net = trading.loc['외국인', '순매수']
            foreign_net_trillion = round(foreign_net / 1_000_000_000_000, 2)

        return {
            'date': last_date.strftime('%Y-%m-%d'),
            'kospi_close': kospi_close,
            'kospi_change': kospi_change,
            'individual_net_buy': individual_net_trillion,
            'foreign_net_buy': foreign_net_trillion,
            'efficiency': efficiency,
            'error': None,
        }
    except Exception as e:
        return {'error': f'KRX 데이터 조회 실패: {str(e)[:80]}'}


# ============================================================
# JSON v2 출력
# ============================================================

def export_json_v2(category_data, summaries, composite, meta, buying_eff=None):
    """계층적 JSON v2 출력.

    {version: 2, categories: {cat_key: {label, icon, groups, summary}}, signals, kpi, ...}
    """

    def nan_to_none(val):
        if pd.isna(val):
            return None
        return val

    def row_to_dict(row):
        return {k: nan_to_none(v) for k, v in row.items()}

    data = {
        'version': 2,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M KST'),
        'data_date': None,
        'categories': {},
        'signals': [],
        'kpi': {},
        'buying_efficiency': None,
    }

    # 데이터 기준일
    if meta.get('last_date') is not None:
        ld = meta['last_date']
        weekdays_ko = ['월', '화', '수', '목', '금', '토', '일']
        data['data_date'] = ld.strftime('%Y-%m-%d') + f' ({weekdays_ko[ld.weekday()]})'

    # 카테고리별 구조화
    for cat_key, cat_config in CATEGORIES.items():
        df = category_data.get(cat_key)
        if df is None:
            continue

        cat_out = {
            'label': cat_config['label'],
            'icon': cat_config['icon'],
            'groups': {},
            'summary': summaries.get(cat_key, {}),
        }

        for grp_name in cat_config['groups']:
            grp_rows = df[df['Group'] == grp_name]
            if len(grp_rows) > 0:
                cat_out['groups'][grp_name] = [row_to_dict(r) for _, r in grp_rows.iterrows()]

        data['categories'][cat_key] = cat_out

    # 복합 시그널
    if composite:
        for s in composite:
            data['signals'].append({
                'level': s.get('Level', ''),
                'signal': s.get('Signal', ''),
                'implication': s.get('Implication', ''),
                'data': s.get('Data', ''),
            })

    # KPI
    rot_result = compute_rotation_score(category_data)
    if rot_result is not None:
        data['kpi']['rotation_score'] = rot_result[0]

    em_rel = compute_em_relative(category_data)
    if em_rel is not None and len(em_rel) > 0:
        eem_row = em_rel[em_rel['Pair'] == 'EWY vs EEM(보정)']
        if len(eem_row) > 0:
            data['kpi']['em_relative_5d'] = eem_row['5D 상대%'].values[0]

    vix = _get_ticker_row(category_data, '^VIX')
    if vix is not None:
        data['kpi']['vix'] = float(vix['Last'])

    hyg = _get_ticker_row(category_data, 'HYG')
    lqd = _get_ticker_row(category_data, 'LQD')
    if hyg is not None and lqd is not None:
        data['kpi']['hyg_lqd_spread_5d'] = round(hyg['5D %'] - lqd['5D %'], 2)

    spy = _get_ticker_row(category_data, 'SPY')
    efa = _get_ticker_row(category_data, 'EFA')
    if spy is not None and efa is not None:
        data['kpi']['spy_vs_efa_5d'] = round(spy['5D %'] - efa['5D %'], 2)

    # v1 호환 레이어
    v1 = {'rotation': [], 'em': [], 'risk': []}
    v1_map = {
        'us_sectors': 'rotation',
        'thematic': 'rotation',
        'regional': 'em',
        'risk_macro': 'risk',
        'bonds_fx': 'risk',
    }
    for cat_key, df in category_data.items():
        target = v1_map.get(cat_key, 'risk')
        cols = ['Group', 'Ticker', 'Name', '1D %', '5D %', '5D Ref', '30D %', 'Vol Δ%']
        available = [c for c in cols if c in df.columns]
        records = df[available].to_dict('records')
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
        v1[target].extend(records)
    data['_v1_compat'] = v1

    # Top Movers
    data['top_movers'] = compute_top_movers(category_data, n=5)

    # 매수 효율
    if buying_eff is not None and buying_eff.get('error') is None:
        data['buying_efficiency'] = buying_eff

    # 저장
    filepath = 'flow_monitor_latest.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON v2: {filepath}")

    return filepath


# ============================================================
# 콘솔 출력
# ============================================================

def print_header(title, data_date=None):
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    run_time = datetime.now().strftime('%Y-%m-%d %H:%M KST')
    if data_date is not None:
        weekdays_ko = ['월', '화', '수', '목', '금', '토', '일']
        wd = weekdays_ko[data_date.weekday()]
        print(f"  실행: {run_time}  |  데이터 기준: {data_date.strftime('%Y-%m-%d')} ({wd})")
    else:
        print(f"  {run_time}")
    print("═" * width)


def print_group_summary(summaries):
    """그룹 평균 서머리 테이블 출력."""
    rows = []
    for cat_key, cat_config in CATEGORIES.items():
        grp_data = summaries.get(cat_key, {})
        for grp_name, vals in grp_data.items():
            rows.append({
                'Category': cat_config['label'][:16],
                'Group': grp_name,
                '1D %': vals['1D %'],
                '5D %': vals['5D %'],
                '30D %': vals['30D %'],
            })

    if rows:
        df = pd.DataFrame(rows)
        print(tabulate(df, headers='keys', tablefmt='simple', showindex=False,
                       numalign='right', floatfmt='+.2f'))


def print_detail_tables(category_data):
    """개별 티커 전체 출력 (--detail)."""
    for cat_key, cat_config in CATEGORIES.items():
        df = category_data.get(cat_key)
        if df is None:
            continue
        print(f"\n── {cat_config['icon']} {cat_config['label']} {'─' * 40}")
        cols = ['Group', 'Ticker', 'Name', 'Last', '1D %', '5D %', '5D Z', '30D %', 'Vol Δ%']
        available = [c for c in cols if c in df.columns]
        print(tabulate(df[available], headers='keys', tablefmt='simple',
                       showindex=False, numalign='right', floatfmt='.2f'))


def print_signals(title, signals):
    print(f"\n── {title} {'─' * max(1, 50 - len(title))}")
    for s in signals:
        if isinstance(s, tuple):
            print(f"  {s[0]} {s[1]}")
        elif isinstance(s, dict):
            print(f"  {s['Level']}  {s['Signal']}")
            print(f"         → {s['Implication']}")
            if s.get('Data'):
                print(f"         📊 {s['Data']}")
            print()


# ============================================================
# 메인 대시보드 (v2.0)
# ============================================================

def run_dashboard(detail=False, export=False):
    """메인 대시보드 실행."""

    # 배치 수집
    category_data, errors, meta = fetch_all_data(days=35)

    if not category_data:
        print("  ⚠️  데이터 수집 실패. 네트워크를 확인하세요.")
        return

    # 필수 티커 확인
    fetched_set = set()
    for df in category_data.values():
        fetched_set.update(df['Ticker'].values)
    missing = check_required_tickers(fetched_set)
    print_missing_warning(missing)

    # 그룹 서머리 계산
    summaries = compute_group_summaries(category_data)

    # 헤더
    print_header("글로벌 플로우 모니터 v2.0 — 히트맵 대시보드", meta.get('last_date'))

    # 서머리 또는 디테일
    if detail:
        print_detail_tables(category_data)
    else:
        print(f"\n── 📊 그룹 평균 서머리 {'─' * 40}")
        print_group_summary(summaries)
        print(f"\n  ℹ️  개별 티커 확인: --detail 플래그 사용")

    # 로테이션 스코어
    rot_result = compute_rotation_score(category_data)
    if rot_result is not None:
        rot_score, score_meta = rot_result
        direction = "→ Growth→Value 진행" if rot_score > 0 else "→ Value→Growth 복귀"
        bar = "█" * min(abs(int(rot_score)), 20)
        print(f"\n  📊 로테이션 스코어 (5D): {rot_score:+.1f}  {direction}")
        print(f"     [{bar}]")
        if score_meta is not None:
            raw = score_meta['raw_with_leverage']
            dev = score_meta['deviation_pct']
            print(f"     검증: 레버리지 포함 시 {raw:+.1f} (편차 {dev:.0f}%)")

    # EM 상대 성과
    em_rel = compute_em_relative(category_data)
    if em_rel is not None:
        print(f"\n── 한국 상대 성과 {'─' * 40}")
        print(tabulate(em_rel, headers='keys', tablefmt='simple', showindex=False))

    # 리스크 시그널
    risk_signals = compute_risk_dashboard(category_data)
    if risk_signals:
        print_signals("리스크 상태 판단", risk_signals)

    # 복합 시그널
    composite = generate_composite_signals(category_data)
    print_signals("복합 시그널 — 외인 행동 예측", composite)

    # 매수 효율
    buying_eff = None
    buying_eff = compute_buying_efficiency()
    if buying_eff is not None:
        print(f"\n── 개인 매수 효율 (Buying Efficiency) {'─' * 22}")
        if buying_eff.get('error'):
            print(f"  ⚠️  {buying_eff['error']}")
        else:
            kospi_chg = buying_eff['kospi_change']
            ind_net = buying_eff['individual_net_buy']
            eff = buying_eff['efficiency']
            fgn_net = buying_eff.get('foreign_net_buy')

            chg_color = "▲" if kospi_chg > 0 else "▼" if kospi_chg < 0 else "─"
            print(f"  📅 기준일: {buying_eff['date']}")
            print(f"  📊 코스피: {buying_eff['kospi_close']:,.0f} ({chg_color}{kospi_chg:+.1f}p)")
            print(f"  🧑 개인 순매수: {ind_net:+.2f}조원")
            if fgn_net is not None:
                print(f"  🌏 외인 순매수: {fgn_net:+.2f}조원")
            if eff is not None:
                if eff > 0:
                    label = "개인 매수 → 지수 상승 (효율적)"
                elif eff > -5:
                    label = "개인 매수 → 외인 매도 상쇄 중"
                else:
                    label = "개인 매수에도 지수 하락 — 방어 실패"
                print(f"  💡 매수 효율: {eff:+.1f} ({label})")
            else:
                print(f"  💡 매수 효율: 계산 불가 (순매수 ≈ 0)")
    elif not HAS_PYKRX:
        print(f"\n── 개인 매수 효율 {'─' * 33}")
        print(f"  ℹ️  pykrx 미설치. 'pip install pykrx'로 설치 시 자동 활성화.")

    # 에러
    if errors:
        print(f"\n  ⚠️  수집 실패 ({len(errors)}개): {', '.join(str(e) for e in errors[:10])}")

    # 내보내기
    if export:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        for cat_key, df in category_data.items():
            filename = f"flow_monitor_{cat_key}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"  💾 CSV: {filename}")
        export_json_v2(category_data, summaries, composite, meta, buying_eff)

    # 푸터
    if meta.get('last_date'):
        ld = meta['last_date']
        weekdays_ko = ['월', '화', '수', '목', '금', '토', '일']
        wd = weekdays_ko[ld.weekday()]
        date_note = f"  데이터 기준: {ld.strftime('%Y-%m-%d')} ({wd})"
    else:
        date_note = "  데이터 기준: 수집 실패"

    print("\n" + "═" * 70)
    print(date_note)
    print(f"  총 {len(fetched_set)}개 티커 수집 완료")
    print("  데이터 소스: Yahoo Finance (15분 지연)" +
          (" + KRX (pykrx)" if HAS_PYKRX else ""))
    print("═" * 70 + "\n")


# ============================================================
# 진입점
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='글로벌 플로우 모니터 v2.0')
    parser.add_argument('--detail', action='store_true', help='개별 티커 전체 출력')
    parser.add_argument('--export', action='store_true', help='JSON v2 + CSV 내보내기')

    args = parser.parse_args()
    run_dashboard(detail=args.detail, export=args.export)
