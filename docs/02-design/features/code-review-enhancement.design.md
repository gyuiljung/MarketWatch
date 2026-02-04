# Design: Market Watch 코드 재점검 및 완성도 향상

## 1. 목표 아키텍처

### 1.1 디렉토리 구조

```
market-monitor/
├── src/
│   └── market_monitor/
│       ├── __init__.py          # 패키지 초기화, 버전 정보
│       ├── cli.py               # CLI 엔트리포인트
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py        # 설정 로더/검증
│       │   ├── constants.py     # 상수 정의
│       │   └── exceptions.py    # 커스텀 예외
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py        # DataLoader 클래스들
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── network.py       # NetworkAnalyzer
│       │   ├── transfer_entropy.py
│       │   ├── volatility.py
│       │   ├── tail_dependence.py
│       │   ├── impulse.py
│       │   └── timeline.py
│       ├── report/
│       │   ├── __init__.py
│       │   ├── generator.py     # ReportGenerator
│       │   └── visualizer.py    # Visualizer
│       └── utils/
│           ├── __init__.py
│           └── logging.py       # 로깅 설정
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # pytest fixtures
│   ├── test_loader.py
│   ├── test_network.py
│   ├── test_transfer_entropy.py
│   └── fixtures/
│       └── sample_data.xlsx     # 테스트용 샘플 데이터
├── config/
│   ├── default.yaml
│   └── clustered.yaml
├── docs/
│   └── ... (PDCA 문서)
├── run_monitor.py               # 간단한 CLI wrapper
├── requirements.txt
├── setup.py                     # 패키지 설치 설정
└── README.md
```

---

## 2. 모듈별 상세 설계

### 2.1 core/config.py

```python
"""설정 관리 모듈"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

@dataclass
class AnalysisConfig:
    """분석 파라미터 설정"""
    window: int = 60
    step: int = 5
    rv_window: int = 5
    lookback: int = 252
    windows: List[int] = field(default_factory=lambda: [5, 20, 60])

@dataclass
class TransferEntropyConfig:
    """Transfer Entropy 설정"""
    bins: int = 6
    lag: int = 1
    n_surrogates: int = 30
    alpha: float = 0.05
    top_n: int = 15

@dataclass
class ThresholdConfig:
    """임계값 설정"""
    rv_extreme: float = 90.0
    rv_elevated: float = 75.0
    sync_warning: float = 0.20
    network_sync_danger: Optional[float] = None
    hub_influence_danger: Optional[float] = None

@dataclass
class VisualizationConfig:
    """시각화 설정"""
    figsize: tuple = (24, 20)
    dpi: int = 150
    heatmap_days: int = 90
    colors: Dict[str, str] = field(default_factory=dict)

@dataclass
class Config:
    """전체 설정"""
    assets: Dict[str, str]
    rate_assets: List[str]
    categories: Dict[str, List[str]]
    category_colors: Dict[str, str]
    analysis: AnalysisConfig
    transfer_entropy: TransferEntropyConfig
    thresholds: ThresholdConfig
    visualization: VisualizationConfig

    # Clustered mode only
    core_assets: Optional[Dict[str, str]] = None
    clusters: Optional[Dict[str, dict]] = None

class ConfigLoader:
    """설정 파일 로더 및 검증기"""

    REQUIRED_KEYS = ['assets', 'rate_assets', 'categories']

    @classmethod
    def load(cls, path: Path) -> Config:
        """YAML 설정 파일 로드"""
        ...

    @classmethod
    def validate(cls, raw_config: dict) -> None:
        """설정 유효성 검증"""
        ...

    @classmethod
    def get_default(cls) -> Config:
        """기본 설정 반환"""
        ...
```

### 2.2 core/exceptions.py

```python
"""커스텀 예외 정의"""

class MarketMonitorError(Exception):
    """기본 예외 클래스"""
    pass

class ConfigError(MarketMonitorError):
    """설정 관련 오류"""
    pass

class DataLoadError(MarketMonitorError):
    """데이터 로딩 오류"""
    pass

class AnalysisError(MarketMonitorError):
    """분석 중 오류"""
    pass

class InsufficientDataError(AnalysisError):
    """데이터 부족 오류"""
    pass
```

### 2.3 core/constants.py

```python
"""상수 정의"""

# Transfer Entropy
TE_DEFAULT_BINS = 6
TE_DEFAULT_LAG = 1
TE_MIN_SAMPLES = 60

# Tail Dependence
TAIL_CORR_COEFFICIENT = 0.35  # expected = q + TAIL_CORR_COEFFICIENT * |corr|
TAIL_CRISIS_THRESHOLD = 0.20
TAIL_ELEVATED_THRESHOLD = 0.15
TAIL_DIVERSIFIED_THRESHOLD = -0.10

# Network
MST_MIN_NODES = 3
CORRELATION_THRESHOLD = 0.1

# Visualization colors (Dark theme)
COLORS = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'text': '#e6edf3',
    'grid': '#30363d',
    'danger': '#f85149',
    'warning': '#d29922',
    'safe': '#3fb950',
    'accent': '#58a6ff',
}
```

### 2.4 data/loader.py

```python
"""데이터 로더 모듈"""
from typing import Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ..core.config import Config
from ..core.exceptions import DataLoadError, InsufficientDataError

logger = logging.getLogger(__name__)

@dataclass
class LoadedData:
    """로드된 데이터 컨테이너"""
    prices: pd.DataFrame
    returns: pd.DataFrame
    assets: List[str]
    period: Tuple[pd.Timestamp, pd.Timestamp]
    working_days: int

class BaseDataLoader:
    """데이터 로더 기본 클래스"""

    def __init__(self, filepath: Path, config: Config):
        self.filepath = filepath
        self.config = config
        self._data: Optional[LoadedData] = None

    def load(self) -> LoadedData:
        """데이터 로드 및 전처리"""
        raise NotImplementedError

    def _read_excel(self) -> pd.DataFrame:
        """Excel 파일 읽기"""
        logger.info(f"Loading data from {self.filepath}")
        try:
            df = pd.read_excel(self.filepath, header=None)
            return df
        except FileNotFoundError:
            raise DataLoadError(f"File not found: {self.filepath}")
        except Exception as e:
            raise DataLoadError(f"Failed to read Excel: {e}")

    def _calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """수익률 계산 (금리는 diff, 나머지는 log return)"""
        returns = np.log(prices / prices.shift(1))
        for col in self.config.rate_assets:
            if col in returns.columns:
                returns[col] = prices[col].diff()
        return returns

    def _filter_working_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """영업일 필터링 (USDKRW 기준)"""
        if 'USDKRW' in df.columns:
            working = df['USDKRW'][df['USDKRW'] != 0].index
            return df.loc[working]
        return df

class DataLoader(BaseDataLoader):
    """기본 데이터 로더 (v1.0 호환)"""

    def load(self) -> LoadedData:
        df = self._read_excel()
        # ... 구현
        return self._data

class ClusteredDataLoader(BaseDataLoader):
    """클러스터 기반 데이터 로더 (v2.3)"""

    def __init__(self, filepath: Path, config: Config):
        super().__init__(filepath, config)
        self.core_assets: List[str] = []
        self.cluster_reps: List[str] = []

    def load(self) -> LoadedData:
        df = self._read_excel()
        # ... 구현
        return self._data

    def get_network_assets(self) -> List[str]:
        """네트워크 분석용 자산 목록"""
        return self.core_assets + self.cluster_reps
```

### 2.5 analysis/network.py

```python
"""네트워크 분석 모듈"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import networkx as nx
import logging

from ..core.config import Config
from ..core.constants import MST_MIN_NODES, CORRELATION_THRESHOLD

logger = logging.getLogger(__name__)

@dataclass
class NetworkSnapshot:
    """네트워크 스냅샷 결과"""
    corr: pd.DataFrame
    mst: nx.Graph
    betweenness: Dict[str, float]
    eigenvector: Dict[str, float]
    top_hub_bt: str
    top_hub_ev: str
    top3_bt: List[Tuple[str, float]]
    top3_ev: List[Tuple[str, float]]
    neighbors: List[str]
    hub_avg_corr: float
    hub_influence: float
    network_sync: float

class NetworkAnalyzer:
    """네트워크 토폴로지 분석기"""

    def __init__(self, config: Config):
        self.config = config
        self.windows = config.analysis.windows

    @staticmethod
    def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Mantegna distance 변환"""
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)

    @staticmethod
    def build_mst(corr: pd.DataFrame) -> nx.Graph:
        """최소신장트리 구축"""
        dist = NetworkAnalyzer.corr_to_distance(corr)
        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j:
                    G.add_edge(a1, a2, weight=dist.iloc[i, j], corr=corr.iloc[i, j])
        return nx.minimum_spanning_tree(G)

    @staticmethod
    def calc_network_sync(corr: pd.DataFrame) -> float:
        """평균 상관계수 (Network Synchronization)"""
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return float(corr.values[mask].mean())

    def compute_snapshot(
        self,
        returns: pd.DataFrame,
        network_assets: List[str],
        window: Optional[int] = None
    ) -> NetworkSnapshot:
        """현재 네트워크 스냅샷 계산"""
        if window is None:
            window = self.windows[-1]

        cols = [c for c in network_assets if c in returns.columns]
        if len(cols) < MST_MIN_NODES:
            raise InsufficientDataError(f"Need at least {MST_MIN_NODES} assets")

        recent = returns[cols].iloc[-window:]
        corr = recent.corr()
        mst = self.build_mst(corr)

        bt = nx.betweenness_centrality(mst)
        ev = self._compute_eigenvector(corr)

        # ... 나머지 구현

        return NetworkSnapshot(...)

    def _compute_eigenvector(self, corr: pd.DataFrame) -> Dict[str, float]:
        """Eigenvector centrality 계산"""
        G_full = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i, j]) > CORRELATION_THRESHOLD:
                    G_full.add_edge(a1, a2, weight=abs(corr.iloc[i, j]))

        try:
            return nx.eigenvector_centrality_numpy(G_full, weight='weight')
        except nx.NetworkXError as e:
            logger.warning(f"Eigenvector centrality failed: {e}")
            return {n: 0.0 for n in corr.columns}

    def compute_multi_scale_sync(
        self,
        returns: pd.DataFrame,
        network_assets: List[str]
    ) -> Dict[str, float]:
        """다중 스케일 동기화 계산"""
        ...

    def compute_timeseries(
        self,
        returns: pd.DataFrame,
        network_assets: List[str]
    ) -> pd.DataFrame:
        """시계열 지표 계산"""
        ...
```

### 2.6 analysis/transfer_entropy.py

```python
"""Transfer Entropy 분석 모듈"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

from ..core.config import Config
from ..core.constants import TE_DEFAULT_BINS, TE_DEFAULT_LAG, TE_MIN_SAMPLES

logger = logging.getLogger(__name__)

@dataclass
class TEResult:
    """Transfer Entropy 결과"""
    te_raw: float
    te_z: float
    p_value: float
    significant: bool
    surrogate_mean: float
    surrogate_std: float

@dataclass
class NetFlowResult:
    """Net Flow 결과"""
    te_ab: TEResult
    te_ba: TEResult
    net_flow_z: float
    dominant: str
    dominant_direction: str
    flow_strength: float
    both_significant: bool
    any_significant: bool

class TransferEntropyCalculator:
    """Transfer Entropy 계산기"""

    def __init__(self, config: Config):
        te_cfg = config.transfer_entropy
        self.bins = te_cfg.bins
        self.lag = te_cfg.lag
        self.n_surrogates = te_cfg.n_surrogates
        self.alpha = te_cfg.alpha
        self.top_n = te_cfg.top_n

    def compute_all_pairs(self, returns: pd.DataFrame) -> Dict[str, TEResult]:
        """모든 페어의 TE 계산"""
        assets = list(returns.columns)
        n_assets = len(assets)

        if len(returns) < TE_MIN_SAMPLES:
            logger.warning(f"Insufficient data: {len(returns)} < {TE_MIN_SAMPLES}")
            return {}

        discretized = self._discretize_all(returns)
        te_raw = self._compute_te_matrix(discretized)

        # Surrogate testing
        te_surrogates = self._compute_surrogates(discretized)
        te_mean = np.mean(te_surrogates, axis=0)
        te_std = np.std(te_surrogates, axis=0) + 1e-10
        te_z = (te_raw - te_mean) / te_std
        p_values = np.mean(te_surrogates >= te_raw[np.newaxis, :, :], axis=0)

        results = {}
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i != j:
                    key = f'{src}→{tgt}'
                    results[key] = TEResult(
                        te_raw=te_raw[i, j],
                        te_z=te_z[i, j],
                        p_value=p_values[i, j],
                        significant=p_values[i, j] < self.alpha,
                        surrogate_mean=te_mean[i, j],
                        surrogate_std=te_std[i, j]
                    )

        return results

    def _discretize_all(self, returns: pd.DataFrame) -> np.ndarray:
        """전체 자산 이산화"""
        ...

    def _compute_te_matrix(self, discretized: np.ndarray) -> np.ndarray:
        """TE 매트릭스 계산"""
        ...

    def _compute_surrogates(self, discretized: np.ndarray) -> np.ndarray:
        """Surrogate 분포 계산"""
        ...

    def compute_net_flow_all(self, returns: pd.DataFrame) -> Dict[str, NetFlowResult]:
        """모든 페어의 Net Flow 계산"""
        ...

    def get_top_significant(
        self,
        all_te: Dict[str, TEResult],
        n: Optional[int] = None
    ) -> List[Tuple[str, TEResult]]:
        """상위 N개 유의미한 TE 페어"""
        ...
```

### 2.7 utils/logging.py

```python
"""로깅 설정 모듈"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """로깅 설정"""

    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )

def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 획득"""
    return logging.getLogger(f'market_monitor.{name}')
```

---

## 3. 인터페이스 설계

### 3.1 CLI 인터페이스

```python
# cli.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Market Network Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run command
    run_parser = subparsers.add_parser('run', help='Run analysis')
    run_parser.add_argument('-d', '--data', required=True, type=Path,
                          help='Path to MARKET_WATCH.xlsx')
    run_parser.add_argument('-o', '--output', default='./output', type=Path,
                          help='Output directory')
    run_parser.add_argument('-c', '--config', type=Path,
                          help='Path to config.yaml')
    run_parser.add_argument('--mode', choices=['basic', 'clustered'],
                          default='clustered', help='Analysis mode')
    run_parser.add_argument('--report-only', action='store_true',
                          help='Generate report only (no dashboard)')
    run_parser.add_argument('-v', '--verbose', action='store_true',
                          help='Verbose output')
    run_parser.add_argument('-q', '--quiet', action='store_true',
                          help='Quiet mode')

    # validate-config command
    validate_parser = subparsers.add_parser('validate-config',
                                           help='Validate config file')
    validate_parser.add_argument('config', type=Path, help='Config file path')

    args = parser.parse_args()

    if args.command == 'run':
        run_analysis(args)
    elif args.command == 'validate-config':
        validate_config(args)
    else:
        parser.print_help()
```

### 3.2 사용 예시

```bash
# 기본 실행
python -m market_monitor run -d MARKET_WATCH.xlsx

# 상세 모드
python -m market_monitor run -d MARKET_WATCH.xlsx -v

# 리포트만 생성
python -m market_monitor run -d MARKET_WATCH.xlsx --report-only

# 설정 파일 검증
python -m market_monitor validate-config config/clustered.yaml

# 기본 모드 (v1.0 호환)
python -m market_monitor run -d MARKET_WATCH.xlsx --mode basic
```

---

## 4. 에러 처리 전략

### 4.1 예외 계층

```
MarketMonitorError (Base)
├── ConfigError
│   ├── ConfigNotFoundError
│   ├── ConfigValidationError
│   └── MissingRequiredKeyError
├── DataLoadError
│   ├── FileNotFoundError
│   ├── InvalidFormatError
│   └── MissingColumnError
└── AnalysisError
    ├── InsufficientDataError
    ├── NetworkConstructionError
    └── ComputationError
```

### 4.2 에러 처리 원칙

1. **구체적 예외 사용**: `except Exception` 대신 구체적 타입
2. **체인 예외**: `raise NewError() from e`로 원인 추적
3. **사용자 친화적 메시지**: 기술적 상세 + 해결 방안
4. **로깅**: 모든 예외는 로그에 기록

```python
# 예시
try:
    df = pd.read_excel(filepath)
except FileNotFoundError as e:
    logger.error(f"Data file not found: {filepath}")
    raise DataLoadError(f"Cannot find data file: {filepath}") from e
except pd.errors.EmptyDataError as e:
    logger.error(f"Empty data file: {filepath}")
    raise DataLoadError(f"Data file is empty: {filepath}") from e
```

---

## 5. 테스트 전략

### 5.1 테스트 범위

| 모듈 | 단위 테스트 | 통합 테스트 |
|------|------------|------------|
| config | 로드, 검증, 기본값 | - |
| loader | 파싱, 필터링, 수익률 계산 | 전체 로드 |
| network | MST, centrality, sync | 스냅샷 |
| transfer_entropy | 이산화, TE 계산, surrogate | 전체 흐름 |

### 5.2 Fixtures

```python
# conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_returns():
    """테스트용 샘플 수익률 데이터"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    assets = ['USDKRW', 'USDJPY', 'SPX', 'VIX', 'KTB_10Y']
    data = np.random.randn(100, 5) * 0.01
    return pd.DataFrame(data, index=dates, columns=assets)

@pytest.fixture
def sample_config():
    """테스트용 설정"""
    from market_monitor.core.config import Config, AnalysisConfig
    return Config(
        assets={'USDKRW': 'test'},
        rate_assets=['KTB_10Y'],
        categories={'FX': ['USDKRW']},
        category_colors={'FX': '#ff0000'},
        analysis=AnalysisConfig(),
        ...
    )
```

---

## 6. 마이그레이션 계획

### 6.1 하위 호환성

기존 CLI 유지:
```bash
# 기존 방식 (계속 동작)
python run_monitor.py -d MARKET_WATCH.xlsx

# 새 방식 (추가)
python -m market_monitor run -d MARKET_WATCH.xlsx
```

### 6.2 단계별 마이그레이션

1. **Phase 1**: 새 구조 생성 (기존 파일 유지)
2. **Phase 2**: 기존 파일을 새 구조로 이동
3. **Phase 3**: 기존 CLI를 wrapper로 변환
4. **Phase 4**: 문서화 및 사용자 가이드

---

## 7. 성능 최적화 포인트

### 7.1 Transfer Entropy

현재:
- All-pairs O(n²) 계산
- Surrogate 30회 반복

최적화:
```python
# Numba JIT 적용
from numba import jit

@jit(nopython=True)
def compute_te_fast(discretized, bins, lag):
    ...

# 병렬화 (선택적)
from concurrent.futures import ProcessPoolExecutor
```

### 7.2 메모리 최적화

- 대용량 데이터셋 청크 처리
- 불필요한 중간 결과 즉시 해제
- `np.float32` 사용 (정밀도 허용 시)

---

## 8. 구현 순서

### Step 1: 기반 구조 (1일)
- [ ] 디렉토리 구조 생성
- [ ] `__init__.py` 파일들 생성
- [ ] `core/exceptions.py` 구현
- [ ] `core/constants.py` 구현

### Step 2: 설정 시스템 (1일)
- [ ] `core/config.py` 구현
- [ ] 설정 검증 로직
- [ ] 기본값 처리

### Step 3: 로깅 시스템 (0.5일)
- [ ] `utils/logging.py` 구현
- [ ] 각 모듈에 로깅 적용

### Step 4: 데이터 로더 (1일)
- [ ] `data/loader.py` - BaseDataLoader
- [ ] DataLoader (v1.0 호환)
- [ ] ClusteredDataLoader

### Step 5: 분석 모듈 (2일)
- [ ] `analysis/network.py`
- [ ] `analysis/transfer_entropy.py`
- [ ] `analysis/volatility.py`
- [ ] `analysis/tail_dependence.py`
- [ ] `analysis/impulse.py`
- [ ] `analysis/timeline.py`

### Step 6: 리포트/시각화 (1일)
- [ ] `report/generator.py`
- [ ] `report/visualizer.py`

### Step 7: CLI (0.5일)
- [ ] `cli.py` 구현
- [ ] `run_monitor.py` wrapper

### Step 8: 테스트 (1일)
- [ ] conftest.py + fixtures
- [ ] 핵심 모듈 테스트

---

*Generated: 2026-02-04*
*Feature: code-review-enhancement*
*Version: 1.0*
