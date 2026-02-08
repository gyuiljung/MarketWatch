"""
V8 Signal Bridge
================
v5 프로젝트의 preview_signal_v8.py를 호출하여
현재 V8 시그널 + 팩터 상세를 가져오는 브릿지.

의존성: v5/src/preview_signal_v8.py, v5/src/config.py
"""
import sys
import pandas as pd
from pathlib import Path


def _restore_src_modules(saved_src, saved_src_children, v5_str, saved_path=None):
    """sys.modules에서 v5의 src를 제거하고 MarketWatch의 src를 복원."""
    import importlib

    # v5가 로드한 src.* 모듈 제거
    to_remove = [k for k in sys.modules if k == 'src' or k.startswith('src.')]
    for k in to_remove:
        sys.modules.pop(k, None)

    # MarketWatch의 src 모듈 복원
    if saved_src is not None:
        sys.modules['src'] = saved_src
    for k, v in saved_src_children.items():
        sys.modules[k] = v

    # sys.path 원본 복원
    if saved_path is not None:
        sys.path[:] = saved_path
    elif v5_str in sys.path:
        sys.path.remove(v5_str)

    importlib.invalidate_caches()


def get_v8_signals(v5_path: str, target_date: str = None) -> dict:
    """
    V8 시그널을 v5 프로젝트에서 가져옴.

    Args:
        v5_path: v5 프로젝트 루트 경로
        target_date: 조회일 (None이면 오늘)

    Returns:
        {
            'kospi': {'signal': float, 't1_total': float, 't2_total': float,
                      'factors': list, 'near_threshold': list, 'theme_summary': dict},
            '3ybm':  { ... }
        }
    """
    v5_path = Path(v5_path)

    if not v5_path.exists():
        print(f"  WARNING: v5 path not found: {v5_path}")
        return {}

    import os
    import importlib

    v5_str = str(v5_path)
    original_cwd = os.getcwd()

    preview_module_path = v5_path / 'src' / 'preview_signal_v8.py'
    if not preview_module_path.exists():
        print(f"  WARNING: v5 preview_signal not found: {preview_module_path}")
        return {}

    # MarketWatch의 src와 v5의 src 충돌 방지:
    # 1) sys.modules에서 기존 src.* 전부 보존 후 제거
    # 2) sys.path에서 MarketWatch 경로 임시 제거, v5 경로 추가
    # 3) import cache 무효화
    saved_src = sys.modules.pop('src', None)
    saved_src_children = {k: v for k, v in sys.modules.items() if k.startswith('src.')}
    for k in saved_src_children:
        sys.modules.pop(k, None)

    # MarketWatch 프로젝트 루트를 sys.path에서 임시 제거
    mw_root = str(Path(__file__).parent.parent.parent)
    saved_path = sys.path.copy()
    sys.path = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(mw_root)]
    sys.path.insert(0, v5_str)

    os.chdir(v5_str)
    importlib.invalidate_caches()

    try:
        from src.preview_signal_v8 import preview_signal
    except ImportError as e:
        print(f"  WARNING: Cannot import v5 preview_signal: {e}")
        os.chdir(original_cwd)
        _restore_src_modules(saved_src, saved_src_children, v5_str, saved_path)
        return {}

    results = {}

    for sheet in ['kospi', '3y']:
        label = 'kospi' if sheet == 'kospi' else '3ybm'
        try:
            result = preview_signal(sheet=sheet, target_date=target_date)
            results[label] = {
                'signal': result['total_signal'],
                't1_total': result['t1_total'],
                't2_total': result['t2_total'],
                'factors': result['t1_factors'] + result['t2_factors'],
                'near_threshold': result.get('near_threshold', []),
                'near_exit': result.get('near_exit', []),
                'theme_summary': result.get('theme_summary', {}),
            }
        except Exception as e:
            print(f"  WARNING: v8 signal load failed for {label}: {e}")
            results[label] = {
                'signal': None,
                't1_total': 0,
                't2_total': 0,
                'factors': [],
                'near_threshold': [],
                'near_exit': [],
                'theme_summary': {},
            }

    # cwd 복원 + sys.modules 복원
    os.chdir(original_cwd)
    _restore_src_modules(saved_src, saved_src_children, v5_str, saved_path)

    return results


def get_v8_signal_summary(v8_signals: dict) -> str:
    """V8 시그널 요약 문자열 생성"""
    lines = []
    for label in ['kospi', '3ybm']:
        data = v8_signals.get(label, {})
        sig = data.get('signal')
        if sig is not None:
            direction = "롱" if sig > 0.6 else ("숏" if sig < 0.4 else "중립")
            t1 = data.get('t1_total', 0)
            t2 = data.get('t2_total', 0)
            n_active = sum(1 for f in data.get('factors', []) if abs(f.get('raw_signal', 0)) > 0.01)
            lines.append(f"  {label.upper()}: {sig:.2f} ({direction}) | T+1: {t1:+.3f}, T+2: {t2:+.3f} | Active: {n_active}")
        else:
            lines.append(f"  {label.upper()}: N/A")
    return '\n'.join(lines)
