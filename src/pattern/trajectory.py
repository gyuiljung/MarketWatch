"""
Pattern Trajectory
==================
유사 과거 시점 이후 수익률 궤적 분석 및 요약.
"""
import numpy as np
import pandas as pd
from typing import List


def compute_trajectories(prices: pd.DataFrame, similar_dates: List[pd.Timestamp],
                         assets: List[str] = None, horizon: int = 60) -> dict:
    """
    유사 시점들의 이후 수익률 궤적 계산.

    Returns:
        {date: {asset: [cumulative return % for each day]}}
    """
    if assets is None:
        assets = ['SPX', 'NKY', 'KOSPI', 'USDJPY', 'USDKRW', 'VIX']

    assets = [a for a in assets if a in prices.columns]
    trajectories = {}

    for date in similar_dates:
        if date not in prices.index:
            continue
        idx = prices.index.get_loc(date)

        traj = {}
        for asset in assets:
            base_price = prices.iloc[idx][asset]
            returns = []
            for d in range(min(horizon + 1, len(prices) - idx)):
                price = prices.iloc[idx + d][asset]
                if pd.notna(base_price) and pd.notna(price) and base_price != 0:
                    returns.append((price / base_price - 1) * 100)
                else:
                    returns.append(np.nan)
            traj[asset] = returns

        trajectories[date] = traj

    return trajectories


def summarize_trajectories(similar_df: pd.DataFrame) -> str:
    """유사 시점 수익률 요약"""
    lines = []

    for horizon in ['5d', '20d', '60d']:
        lines.append(f"\n  {horizon} Average Returns:")
        for asset in ['SPX', 'NKY', 'KOSPI', 'USDJPY', 'USDKRW', 'VIX']:
            col = f'{horizon}_{asset}'
            if col in similar_df.columns:
                avg = similar_df[col].mean()
                std = similar_df[col].std()
                if pd.notna(avg):
                    lines.append(f"    {asset}: {avg:+.2f}% (±{std:.2f}%)")

    return '\n'.join(lines)


def check_direction_consistency(similar_df: pd.DataFrame) -> List[str]:
    """Forward return 방향 일치율 검증"""
    checks = []

    for horizon in ['5d', '20d']:
        for asset in ['SPX', 'VIX', 'USDJPY']:
            col = f'{horizon}_{asset}'
            if col in similar_df.columns:
                vals = similar_df[col].dropna()
                if len(vals) >= 3:
                    positive = (vals > 0).sum()
                    total = len(vals)
                    ratio = positive / total
                    direction = "UP" if ratio > 0.5 else "DOWN"
                    consistency = max(ratio, 1 - ratio) * 100

                    status = "PASS" if consistency >= 60 else "WARN"
                    checks.append(f"  [{status}] {horizon} {asset}: {direction} ({consistency:.0f}% consistent)")

    return checks
