"""
L2: Transfer Entropy (Improved)
===============================
인과 흐름 분석 - 개선 버전.

개선점 (vs market_monitor v2.3):
1. 10-bin quantile 적응형 이산화 (6-bin rank → 10-bin quantile)
2. Lag 1~5 스캔, 최대 TE lag 자동 선택 (고정 lag=1 → multi-lag)
3. 100 surrogates (30 → 100, p-value 안정성)
4. Net TE flow + best lag 보고
"""
import numpy as np
import pandas as pd


class TransferEntropyCalculator:
    """
    All-pairs Transfer Entropy with multi-lag scanning and surrogate testing.
    """

    def __init__(self, config: dict):
        te_cfg = config.get('transfer_entropy', {})
        self.bins = te_cfg.get('bins', 10)
        self.max_lag = te_cfg.get('max_lag', 5)
        self.n_surrogates = te_cfg.get('n_surrogates', 100)
        self.alpha = te_cfg.get('alpha', 0.05)
        self.top_n = te_cfg.get('top_n', 15)

    def _discretize_quantile(self, returns: pd.DataFrame) -> np.ndarray:
        """적응형 quantile 이산화 (자산별 분포 반영)"""
        n_assets = len(returns.columns)
        T = len(returns)
        result = np.zeros((n_assets, T), dtype=int)

        for i, col in enumerate(returns.columns):
            series = returns[col].values
            valid = ~np.isnan(series)
            if valid.sum() == 0:
                continue
            # Quantile-based binning (각 자산의 분포에 적응)
            valid_vals = series[valid]
            edges = np.percentile(valid_vals, np.linspace(0, 100, self.bins + 1))
            edges[-1] += 1e-10  # 마지막 빈 포함
            bins = np.digitize(valid_vals, edges[1:])
            bins = np.minimum(bins, self.bins - 1)
            result[i, valid] = bins

        return result

    def _compute_te_single_lag(self, discretized: np.ndarray, lag: int) -> np.ndarray:
        """특정 lag에서 모든 pairwise TE 계산"""
        n_assets, T = discretized.shape
        n = T - lag
        results = np.zeros((n_assets, n_assets))

        for tgt in range(n_assets):
            tgt_d = discretized[tgt]
            tgt_future = tgt_d[lag:]
            tgt_past = tgt_d[:-lag]

            # P(Y_t, Y_{t-lag})
            joint_yy = np.zeros((self.bins, self.bins))
            for t in range(n):
                joint_yy[tgt_future[t], tgt_past[t]] += 1
            joint_yy /= n

            marginal_y = np.bincount(tgt_past[:n], minlength=self.bins).astype(float) / n

            for src in range(n_assets):
                if src == tgt:
                    continue

                src_past = discretized[src, :-lag]

                # P(Y_t, Y_{t-lag}, X_{t-lag})
                joint_yyx = np.zeros((self.bins, self.bins, self.bins))
                for t in range(n):
                    joint_yyx[tgt_future[t], tgt_past[t], src_past[t]] += 1
                joint_yyx /= n

                # P(Y_{t-lag}, X_{t-lag})
                marginal_yx = np.zeros((self.bins, self.bins))
                for t in range(n):
                    marginal_yx[tgt_past[t], src_past[t]] += 1
                marginal_yx /= n

                # TE = sum p(y_t, y_past, x_past) * log2(p(y_t, y_past, x_past) * p(y_past) / (p(y_past, x_past) * p(y_t, y_past)))
                te = 0.0
                for yt in range(self.bins):
                    for yp in range(self.bins):
                        for xp in range(self.bins):
                            p_joint = joint_yyx[yt, yp, xp]
                            if p_joint > 1e-10:
                                p_yy = joint_yy[yt, yp]
                                p_yx = marginal_yx[yp, xp]
                                p_y = marginal_y[yp]
                                if p_yy > 1e-10 and p_yx > 1e-10 and p_y > 1e-10:
                                    te += p_joint * np.log2(
                                        (p_joint * p_y) / (p_yx * p_yy)
                                    )

                results[src, tgt] = max(0, te)

        return results

    def _compute_te_best_lag(self, discretized: np.ndarray) -> tuple:
        """Lag 1~max_lag 스캔, 최대 TE lag 자동 선택"""
        n_assets = discretized.shape[0]
        best_te = np.zeros((n_assets, n_assets))
        best_lags = np.ones((n_assets, n_assets), dtype=int)

        for lag in range(1, self.max_lag + 1):
            te_matrix = self._compute_te_single_lag(discretized, lag)
            improved = te_matrix > best_te
            best_te[improved] = te_matrix[improved]
            best_lags[improved] = lag

        return best_te, best_lags

    def compute_all_pairs(self, returns: pd.DataFrame) -> dict:
        """모든 쌍 TE 계산 (multi-lag + surrogate testing)"""
        assets = list(returns.columns)
        n_assets = len(assets)

        # Quantile 이산화
        discretized = self._discretize_quantile(returns)

        # Best-lag TE matrix
        te_raw, te_lags = self._compute_te_best_lag(discretized)

        # Surrogate distribution
        te_surrogates = np.zeros((self.n_surrogates, n_assets, n_assets))
        for s in range(self.n_surrogates):
            if s % 20 == 0:
                print(f"    Surrogate {s}/{self.n_surrogates}", end='\r')
            perm = np.random.permutation(len(returns))
            shuffled = discretized[:, perm]
            te_surrogates[s], _ = self._compute_te_best_lag(shuffled)

        print(f"    Surrogate {self.n_surrogates}/{self.n_surrogates} Done")

        # Z-scores and p-values
        te_mean = np.mean(te_surrogates, axis=0)
        te_std = np.std(te_surrogates, axis=0) + 1e-10
        te_z = (te_raw - te_mean) / te_std
        p_values = np.mean(te_surrogates >= te_raw[np.newaxis, :, :], axis=0)

        # Build results
        results = {}
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i != j:
                    results[f'{src}\u2192{tgt}'] = {
                        'te_raw': te_raw[i, j],
                        'te_z': te_z[i, j],
                        'p_value': p_values[i, j],
                        'significant': p_values[i, j] < self.alpha,
                        'best_lag': int(te_lags[i, j]),
                        'surrogate_mean': te_mean[i, j],
                        'surrogate_std': te_std[i, j],
                    }

        return results

    def compute_net_flow(self, returns: pd.DataFrame) -> dict:
        """Net Information Flow for all unique pairs"""
        all_te = self.compute_all_pairs(returns)
        assets = list(returns.columns)

        results = {}
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i < j:
                    key_ab = f'{a}\u2192{b}'
                    key_ba = f'{b}\u2192{a}'

                    if key_ab in all_te and key_ba in all_te:
                        te_ab = all_te[key_ab]
                        te_ba = all_te[key_ba]

                        net_flow = te_ab['te_z'] - te_ba['te_z']

                        results[f'{a}\u21c4{b}'] = {
                            'te_ab': te_ab,
                            'te_ba': te_ba,
                            'net_flow_z': net_flow,
                            'dominant': a if net_flow > 0 else b,
                            'dominant_direction': f'{a}\u2192{b}' if net_flow > 0 else f'{b}\u2192{a}',
                            'flow_strength': abs(net_flow),
                            'best_lag_ab': te_ab['best_lag'],
                            'best_lag_ba': te_ba['best_lag'],
                            'both_significant': te_ab['significant'] and te_ba['significant'],
                            'any_significant': te_ab['significant'] or te_ba['significant'],
                        }

        return results

    def get_top_significant(self, all_te: dict, n: int = None) -> list:
        if n is None:
            n = self.top_n
        sig_pairs = [(k, v) for k, v in all_te.items() if v['significant']]
        sig_pairs.sort(key=lambda x: x[1]['te_z'], reverse=True)
        return sig_pairs[:n]

    def get_top_net_flows(self, net_flow: dict, n: int = None) -> list:
        if n is None:
            n = self.top_n
        flows = list(net_flow.items())
        flows.sort(key=lambda x: x[1]['flow_strength'], reverse=True)
        return flows[:n]
