"""
L3: Impulse Response
====================
Hub 자산 충격이 연결 자산에 전파되는 패턴 분석.
market_monitor ImpulseResponseAnalyzer 기반.
"""
import numpy as np
import pandas as pd


class ImpulseResponseAnalyzer:
    """
    Hub asset → Connected assets 충격 전파 분석.
    - Lead/Lag cross-correlation
    - Conditional response (hub moves > 1 std)
    """

    def __init__(self, config: dict):
        analysis = config.get('analysis', {})
        self.window = analysis.get('impulse_window', 60)
        self.max_lag = analysis.get('impulse_max_lag', 5)

    def compute_lead_lag(self, returns: pd.DataFrame, hub: str, neighbors: list) -> dict:
        """Compute lead-lag cross-correlations"""
        recent = returns.iloc[-self.window:]
        results = {}

        for neighbor in neighbors:
            if neighbor not in recent.columns or hub not in recent.columns:
                continue

            lag_corrs = {}
            for lag in range(-self.max_lag, self.max_lag + 1):
                if lag < 0:
                    x = recent[hub].iloc[:lag].values
                    y = recent[neighbor].iloc[-lag:].values
                elif lag > 0:
                    x = recent[hub].iloc[lag:].values
                    y = recent[neighbor].iloc[:-lag].values
                else:
                    x = recent[hub].values
                    y = recent[neighbor].values

                if len(x) > 10:
                    corr = np.corrcoef(x, y)[0, 1]
                    lag_corrs[lag] = corr

            if lag_corrs:
                best_lag = max(lag_corrs.keys(), key=lambda k: abs(lag_corrs[k]))
                results[neighbor] = {
                    'lag_correlations': lag_corrs,
                    'best_lag': best_lag,
                    'best_corr': lag_corrs[best_lag],
                    'contemporaneous': lag_corrs.get(0, 0),
                    'hub_leads': best_lag < 0,
                }

        return results

    def compute_conditional_response(self, returns: pd.DataFrame, hub: str,
                                     neighbors: list, std_threshold: float = 1.0) -> dict:
        """When hub moves > threshold std, what happens to neighbors?"""
        recent = returns.iloc[-self.window:]

        if hub not in recent.columns:
            return {}

        hub_std = recent[hub].std()
        big_move_mask = recent[hub].abs() > (hub_std * std_threshold)
        big_move_dates = recent[big_move_mask].index

        results = {'hub': hub, 'n_events': len(big_move_dates), 'std_threshold': std_threshold}

        for neighbor in neighbors:
            if neighbor not in recent.columns:
                continue

            up_responses = []
            down_responses = []

            for event_date in big_move_dates:
                idx = recent.index.get_loc(event_date)
                hub_ret = recent[hub].iloc[idx]

                response = {
                    'date': event_date,
                    'hub_ret': hub_ret,
                    'same_day': recent[neighbor].iloc[idx],
                }

                response['past_5d'] = recent[neighbor].iloc[max(0, idx-5):idx].mean() if idx >= 5 else np.nan
                response['future_5d'] = recent[neighbor].iloc[idx+1:idx+6].mean() if idx + 5 < len(recent) else np.nan

                if hub_ret > 0:
                    up_responses.append(response)
                else:
                    down_responses.append(response)

            neighbor_result = {}
            if up_responses:
                up_df = pd.DataFrame(up_responses)
                neighbor_result['up'] = {
                    'n': len(up_df),
                    'past_5d_avg': up_df['past_5d'].mean(),
                    'same_day_avg': up_df['same_day'].mean(),
                    'future_5d_avg': up_df['future_5d'].mean(),
                }
            if down_responses:
                down_df = pd.DataFrame(down_responses)
                neighbor_result['down'] = {
                    'n': len(down_df),
                    'past_5d_avg': down_df['past_5d'].mean(),
                    'same_day_avg': down_df['same_day'].mean(),
                    'future_5d_avg': down_df['future_5d'].mean(),
                }
            results[neighbor] = neighbor_result

        return results
