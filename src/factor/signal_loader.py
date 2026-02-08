"""
V8 Signal Loader
================
v8 config 파일에서 현재 시그널 상세 정보를 로딩.
v8_bridge가 전체 프리뷰를 실행하는 반면,
이 모듈은 config CSV에서 팩터 구성만 빠르게 로딩.
"""
import pandas as pd
from pathlib import Path


def load_v8_config(v5_path: str, config_files: dict) -> dict:
    """
    V8 config CSV 파일에서 팩터 구성 로딩.

    Args:
        v5_path: v5 프로젝트 루트
        config_files: {'kospi': 'output_v3/final_kospi_...csv', '3ybm': '...csv'}

    Returns:
        {
            'kospi': DataFrame (팩터 설정),
            '3ybm': DataFrame
        }
    """
    v5_root = Path(v5_path)
    result = {}

    for label, rel_path in config_files.items():
        path = v5_root / rel_path
        if path.exists():
            df = pd.read_csv(path)
            result[label] = df
            print(f"  V8 config loaded: {label} ({len(df)} factors)")
        else:
            print(f"  WARNING: V8 config not found: {path}")
            result[label] = pd.DataFrame()

    return result


def get_active_factors(config_df: pd.DataFrame) -> list:
    """활성 팩터 목록 반환"""
    if config_df.empty:
        return []
    return config_df['factor'].tolist()


def get_factor_by_theme(config_df: pd.DataFrame) -> dict:
    """팩터를 테마별로 그룹핑 (v5 FACTOR_THEMES 기반)"""
    themes = {}
    for _, row in config_df.iterrows():
        factor = row['factor']
        # 간단한 테마 추론
        theme = _infer_theme(factor)
        if theme not in themes:
            themes[theme] = []
        themes[theme].append({
            'factor': factor,
            'signal_type': row.get('signal_type', 'hysteresis'),
            'direction': row.get('direction', 'momentum'),
            'score': row.get('combined_score', 1.0),
        })
    return themes


def _infer_theme(factor: str) -> str:
    """팩터명에서 테마 추론"""
    factor_upper = factor.upper()
    theme_keywords = {
        'FX_KRW': ['NDF', 'JPY-KRW', 'EUR-KRW', 'CNY-KRW', 'USD-KRW'],
        'FX_GLOBAL': ['달러인덱스', 'USD-EUR'],
        'RATE_KR': ['TP-IRS', 'TP ICAP', 'TP-CRS', '국고채', '금투협'],
        'RATE_CREDIT': ['회사채', 'BBB', 'AA-'],
        'LIQUIDITY': ['융자잔고', '예탁금', '저축성예금', '대출'],
        'EQUITY_KR': ['중형주', '코스피', '코스닥'],
        'COMMODITY': ['고무', '아연', '구리', 'SHFE', 'LME', '두바이유'],
        'GLOBAL_EQUITY': ['S&P', 'TSX', 'NASDAQ', '엔비디아'],
    }
    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if kw.upper() in factor_upper:
                return theme
    return 'OTHER'
