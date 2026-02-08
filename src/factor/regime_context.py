"""
Regime Context
==============
매크로 레짐(turbulence, network hub, sync level)과
V8 시그널을 교차하여 해석.

"V8 시그널이 현재 매크로 환경에서 어떤 의미인가?"
"""


def interpret_signal_in_regime(macro_state: dict, v8_signals: dict) -> dict:
    """
    매크로 레짐 × V8 시그널 교차 분석.

    Args:
        macro_state: {
            'turbulence_status': str,       # 'calm'/'normal'/'elevated'/'extreme'
            'extreme_assets': list,
            'sync': dict,                   # multi-scale sync values
            'top_hub': str,
            'regime_shift': bool,
        }
        v8_signals: {
            'kospi': {'signal': float, ...},
            '3ybm': {'signal': float, ...}
        }

    Returns:
        {
            'kospi': {'context': str, 'caution_level': int, 'details': list},
            '3ybm': {'context': str, 'caution_level': int, 'details': list}
        }
    """
    results = {}

    for label in ['kospi', '3ybm']:
        v8 = v8_signals.get(label, {})
        signal = v8.get('signal')
        if signal is None:
            results[label] = {'context': 'V8 시그널 없음', 'caution_level': 0, 'details': []}
            continue

        caution = 0
        details = []

        # Signal direction
        if signal > 0.6:
            sig_direction = 'LONG'
        elif signal < 0.4:
            sig_direction = 'SHORT'
        else:
            sig_direction = 'NEUTRAL'

        # 1. Turbulence context
        turb_status = macro_state.get('turbulence_status', 'normal')
        extreme_assets = macro_state.get('extreme_assets', [])

        if turb_status == 'extreme':
            caution += 2
            details.append(f"VIX/변동성 EXTREME → {sig_direction} 시그널에 주의")
            if label == 'kospi' and sig_direction == 'LONG':
                details.append("변동성 극단에서 롱 → 역방향 리스크")
                caution += 1

        elif turb_status == 'elevated':
            caution += 1
            details.append(f"변동성 elevated ({len(extreme_assets)} 자산 extreme)")

        # 2. VIX specific
        if 'VIX' in extreme_assets:
            caution += 1
            details.append("VIX EXTREME → 시장 공포 구간")

        # 3. Sync context
        sync = macro_state.get('sync', {})
        short_sync = sync.get(f'sync_5d', 0)
        regime_shift = macro_state.get('regime_shift', False)

        if regime_shift:
            caution += 1
            details.append("단기 동기화 급등 → 레짐 전환 가능성")

        if short_sync > 0.25:
            details.append(f"고동기화 ({short_sync:.3f}) → 분산 불가 구간")
            if label == '3ybm':
                details.append("채권도 동조화 → 안전자산 헤지 약화")

        # 4. Hub context
        top_hub = macro_state.get('top_hub', '')
        if top_hub == 'VIX':
            caution += 1
            details.append("VIX가 Hub → 위험 회피 주도 시장")
        elif top_hub in ['GOLD', 'DXY']:
            details.append(f"{top_hub}가 Hub → 매크로/통화 주도 시장")

        # Summary
        if caution >= 3:
            context = f"{sig_direction} 시그널 + 고위험 환경 → 포지션 축소 권고"
        elif caution >= 2:
            context = f"{sig_direction} 시그널 + 경계 환경 → 사이즈 조절"
        elif caution >= 1:
            context = f"{sig_direction} 시그널 + 소폭 경계"
        else:
            context = f"{sig_direction} 시그널, 매크로 안정"

        results[label] = {
            'context': context,
            'caution_level': caution,
            'details': details,
            'signal_direction': sig_direction,
            'signal_value': signal,
        }

    return results
