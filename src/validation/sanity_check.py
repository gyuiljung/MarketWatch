"""
Sanity Check
============
알려진 사실 검증 (market-watch 기반).

원칙:
1. Raw 숫자 보여주기
2. 경제적으로 말이 되는가?
3. Baseline 대비 개선 있는가?
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

KNOWN_FACTS = {
    'corr': {
        ('VIX', 'SPX'): {'sign': -1, 'min_abs': 0.5, 'desc': 'VIX-SPX 강한 역상관'},
        ('USDJPY', 'USDKRW'): {'sign': 1, 'min_abs': 0.1, 'desc': '달러 강세 시 둘 다 상승'},
        ('SPX', 'USDKRW'): {'sign': -1, 'min_abs': 0.0, 'desc': '위험회피 시 원화 약세'},
        ('GOLD', 'DXY'): {'sign': -1, 'min_abs': 0.0, 'desc': '달러 강세 시 금 약세'},
    },
    'lead_lag': {
        ('SPX', 'NKY'): {'leader': 'SPX', 'min_improvement': 0.05, 'desc': '미국→일본 시차'},
        ('SPX', 'KOSPI'): {'leader': 'SPX', 'min_improvement': 0.05, 'desc': '미국→한국 시차'},
    },
    'volatility': {
        'VIX': {'min_std': 5.0, 'desc': 'VIX는 변동성이 커야 함'},
        'SPX': {'max_std': 3.0, 'desc': 'SPX는 상대적으로 안정'},
    }
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    expected: str
    actual: str
    details: str


class SanityChecker:
    """결과 검증기"""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.results: List[CheckResult] = []

    def check_correlation(self, asset1: str, asset2: str) -> Optional[CheckResult]:
        key = (asset1, asset2)
        reverse_key = (asset2, asset1)

        fact = KNOWN_FACTS['corr'].get(key) or KNOWN_FACTS['corr'].get(reverse_key)
        if not fact:
            return None
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return None

        actual_corr = self.returns[asset1].corr(self.returns[asset2])
        sign_ok = (actual_corr * fact['sign']) > 0
        magnitude_ok = abs(actual_corr) >= fact['min_abs']

        return CheckResult(
            name=f"Corr({asset1}, {asset2})",
            passed=sign_ok and magnitude_ok,
            expected=f"sign={'+' if fact['sign']>0 else '-'}, |r|>={fact['min_abs']}",
            actual=f"r={actual_corr:.3f}",
            details=fact['desc']
        )

    def check_lead_lag(self, asset1: str, asset2: str, max_lag: int = 5) -> Optional[CheckResult]:
        key = (asset1, asset2)
        reverse_key = (asset2, asset1)

        fact = KNOWN_FACTS['lead_lag'].get(key) or KNOWN_FACTS['lead_lag'].get(reverse_key)
        if not fact:
            return None

        pair = key if key in KNOWN_FACTS['lead_lag'] else reverse_key
        a1, a2 = pair
        if a1 not in self.returns.columns or a2 not in self.returns.columns:
            return None

        x, y = self.returns[a1], self.returns[a2]
        corr_0 = x.corr(y)
        best_corr, best_lag = corr_0, 0

        for lag in range(1, max_lag + 1):
            corr_lead = x.shift(lag).corr(y)
            if abs(corr_lead) > abs(best_corr):
                best_corr, best_lag = corr_lead, lag

            corr_follow = x.corr(y.shift(lag))
            if abs(corr_follow) > abs(best_corr):
                best_corr, best_lag = corr_follow, -lag

        improvement = abs(best_corr) - abs(corr_0)
        expected_lag_sign = 1 if fact['leader'] == a1 else -1
        direction_ok = (best_lag * expected_lag_sign) > 0 if best_lag != 0 else False
        improvement_ok = improvement >= fact['min_improvement']

        actual_leader = a1 if best_lag > 0 else (a2 if best_lag < 0 else "동시")

        return CheckResult(
            name=f"LeadLag({a1}, {a2})",
            passed=direction_ok and improvement_ok,
            expected=f"{fact['leader']} leads, improve>={fact['min_improvement']}",
            actual=f"{actual_leader} (lag={best_lag}, improve={improvement:.3f})",
            details=fact['desc']
        )

    def check_volatility(self, asset: str) -> Optional[CheckResult]:
        fact = KNOWN_FACTS['volatility'].get(asset)
        if not fact or asset not in self.returns.columns:
            return None

        actual_std = self.returns[asset].std() * 100
        passed = True
        if 'min_std' in fact:
            passed = passed and (actual_std >= fact['min_std'])
        if 'max_std' in fact:
            passed = passed and (actual_std <= fact['max_std'])

        expected_parts = []
        if 'min_std' in fact:
            expected_parts.append(f"std>={fact['min_std']}%")
        if 'max_std' in fact:
            expected_parts.append(f"std<={fact['max_std']}%")

        return CheckResult(
            name=f"Vol({asset})", passed=passed,
            expected=" & ".join(expected_parts),
            actual=f"std={actual_std:.2f}%",
            details=fact['desc']
        )

    def run_all(self) -> List[CheckResult]:
        self.results = []

        for (a1, a2) in KNOWN_FACTS['corr'].keys():
            r = self.check_correlation(a1, a2)
            if r:
                self.results.append(r)

        for (a1, a2) in KNOWN_FACTS['lead_lag'].keys():
            r = self.check_lead_lag(a1, a2)
            if r:
                self.results.append(r)

        for asset in KNOWN_FACTS['volatility'].keys():
            r = self.check_volatility(asset)
            if r:
                self.results.append(r)

        return self.results

    def get_summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        return f"{passed}/{total} passed ({passed/total*100:.0f}%)" if total else "No checks"
