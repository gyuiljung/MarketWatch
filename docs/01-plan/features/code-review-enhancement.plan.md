# PDCA Plan: Market Watch 코드 재점검 및 완성도 향상

## 1. 개요

### 1.1 현황 분석
**프로젝트 위치**: `C:\Users\infomax\.claude\Market Watch\market-monitor`

**현재 파일 구조**:
```
market-monitor/
├── run_monitor.py           # v1.0 기본 버전 (764줄)
├── run_monitor_clustered.py # v2.3 확장 버전 (~1200줄 이상)
├── config.yaml              # 기본 설정 (144줄)
└── config_clustered.yaml    # 클러스터 설정 (164줄)
```

**주요 컴포넌트** (run_monitor_clustered.py 기준):
| 클래스 | 역할 | 라인수(추정) |
|--------|------|-------------|
| `ClusteredDataLoader` | MARKET_WATCH.xlsx 로드 및 전처리 | ~70 |
| `TransferEntropyCalculator` | All-pairs TE + Surrogate testing | ~200 |
| `ClusteredNetworkAnalyzer` | MST/Betweenness/Multi-scale Sync | ~150 |
| `KeyPairsAnalyzer` | 주요 페어 상관관계 추적 | ~60 |
| `VolatilityAnalyzer` | RV Percentile 계산 | ~30 |
| `TailDependenceCalculator` | Copula-free Tail Dependence | ~100 |
| `ImpulseResponseAnalyzer` | Lead/Lag + Conditional Response | ~100 |
| `TimelineTracker` | 주간 Hub/TE 변화 추적 | ~100 |
| `ReportGenerator` | 텍스트 리포트 생성 | ~300 |
| `Visualizer` | 대시보드 이미지 생성 | ~300+ |

### 1.2 현재 문제점

#### A. 코드 품질
1. **에러 핸들링 부족**: bare `except:` 사용, 구체적 예외 처리 없음
2. **로깅 없음**: `print()` 기반 출력만 사용
3. **타입 힌트 부분적**: 일부 함수에만 적용
4. **Docstring 불완전**: 일부 클래스/메서드 설명 누락
5. **매직 넘버**: 하드코딩된 상수 (0.35, 0.15, 0.20 등)

#### B. 구조적 문제
1. **단일 파일 구조**: 모든 클래스가 하나의 파일에 집중
2. **중복 코드**: v1.0과 v2.3 간 중복 로직 존재
3. **설정 검증 없음**: config 파일 유효성 검증 부재
4. **테스트 부재**: 유닛 테스트, 통합 테스트 없음

#### C. 기능 검토 필요
1. **인코딩 문제**: 한글 docstring이 깨져서 표시됨
2. **출력 포맷**: 리포트 구조 최적화 필요
3. **CLI 옵션**: 추가 옵션 필요 여부 검토

---

## 2. 개선 계획

### Phase 1: 코드 품질 개선 (Priority: High)

#### 1.1 에러 핸들링 강화
- [ ] bare `except:` → 구체적 예외 타입으로 변경
- [ ] 사용자 친화적 에러 메시지 추가
- [ ] 데이터 로딩 실패 시 graceful degradation

#### 1.2 로깅 시스템 도입
- [ ] `logging` 모듈 적용
- [ ] 로그 레벨 설정 (DEBUG/INFO/WARNING/ERROR)
- [ ] 파일 로깅 옵션 추가

#### 1.3 타입 힌트 완성
- [ ] 모든 함수/메서드에 타입 힌트 추가
- [ ] `typing` 모듈 활용 (Optional, Dict, List 등)
- [ ] 복잡한 반환 타입 명시

#### 1.4 Docstring 보완
- [ ] 모든 클래스/메서드에 Google-style docstring 추가
- [ ] 한글 인코딩 문제 해결 (UTF-8 BOM 확인)

### Phase 2: 구조 개선 (Priority: Medium)

#### 2.1 모듈 분리
```
market-monitor/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # DataLoader 클래스
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── network.py          # NetworkAnalyzer
│   │   ├── transfer_entropy.py # TransferEntropyCalculator
│   │   ├── volatility.py       # VolatilityAnalyzer
│   │   ├── tail_dependence.py  # TailDependenceCalculator
│   │   └── impulse.py          # ImpulseResponseAnalyzer
│   ├── report/
│   │   ├── __init__.py
│   │   ├── generator.py        # ReportGenerator
│   │   └── visualizer.py       # Visualizer
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Config loader/validator
│       └── constants.py        # 매직 넘버 상수화
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_network.py
│   └── test_te.py
├── config/
│   ├── default.yaml
│   └── clustered.yaml
├── run_monitor.py              # CLI 엔트리포인트
└── requirements.txt
```

#### 2.2 설정 검증
- [ ] YAML 스키마 정의
- [ ] 필수 키 검증
- [ ] 타입 검증
- [ ] 기본값 처리

#### 2.3 중복 코드 통합
- [ ] v1.0과 v2.3 공통 로직 추출
- [ ] Base 클래스 도입

### Phase 3: 테스트 추가 (Priority: Medium)

#### 3.1 유닛 테스트
- [ ] DataLoader 테스트
- [ ] NetworkAnalyzer 테스트
- [ ] TransferEntropy 테스트
- [ ] 샘플 데이터 fixture 준비

#### 3.2 통합 테스트
- [ ] 전체 파이프라인 테스트
- [ ] 출력 검증

### Phase 4: 문서화 (Priority: Low)

#### 4.1 README 작성
- [ ] 설치 방법
- [ ] 사용법
- [ ] 설정 옵션 설명
- [ ] 출력물 설명

#### 4.2 사용 가이드
- [ ] 각 지표 해석 방법
- [ ] Alert 대응 가이드

---

## 3. 상세 개선 항목

### 3.1 즉시 수정 필요 항목 (Quick Wins)

| # | 항목 | 위치 | 설명 |
|---|------|------|------|
| 1 | bare except 제거 | run_monitor_clustered.py:262, 458 | 구체적 예외로 변경 |
| 2 | 인코딩 수정 | 전체 파일 | UTF-8 + BOM 확인 |
| 3 | 매직 넘버 상수화 | TailDependenceCalculator | 0.35, 0.15, 0.20 등 |
| 4 | 함수 반환값 명시 | compute_* 함수들 | 타입 힌트 추가 |

### 3.2 중요 개선 항목

#### A. TransferEntropyCalculator 최적화
현재 문제:
- All-pairs 계산 시 O(n²) 복잡도
- Surrogate 30회 반복으로 시간 소요

개선안:
- 병렬 처리 (multiprocessing)
- Numba JIT 컴파일 적용
- 캐싱 전략

#### B. Visualizer 분리
현재 문제:
- 단일 클래스에 모든 차트 로직 집중
- 패널별 커스터마이징 어려움

개선안:
- 각 패널을 별도 클래스/함수로 분리
- 차트 템플릿 시스템 도입

#### C. CLI 개선
현재:
```bash
python run_monitor.py -d MARKET_WATCH.xlsx -o ./output
```

개선안:
```bash
# 서브커맨드 도입
python -m market_monitor run -d MARKET_WATCH.xlsx
python -m market_monitor validate-config
python -m market_monitor export --format json
```

---

## 4. 일정 계획

### Week 1: Phase 1 (코드 품질)
- Day 1-2: 에러 핸들링 + 로깅
- Day 3-4: 타입 힌트 완성
- Day 5: Docstring 보완

### Week 2: Phase 2 (구조 개선)
- Day 1-3: 모듈 분리
- Day 4-5: 설정 검증 + 중복 제거

### Week 3: Phase 3-4 (테스트 + 문서화)
- Day 1-3: 테스트 작성
- Day 4-5: 문서화

---

## 5. 성공 기준

| 항목 | 목표 | 측정 방법 |
|------|------|----------|
| 코드 품질 | bare except 0개 | grep 검색 |
| 타입 커버리지 | 90% 이상 | mypy |
| 테스트 커버리지 | 70% 이상 | pytest-cov |
| 문서화 | README + docstring 완료 | 수동 검토 |

---

## 6. 리스크

| 리스크 | 영향 | 완화 방안 |
|--------|------|----------|
| 대규모 리팩토링 시 버그 발생 | High | 단계별 진행, 테스트 선작성 |
| 기존 사용법 변경 | Medium | 하위 호환성 유지, 마이그레이션 가이드 |
| 성능 저하 가능 | Low | 벤치마크 테스트 수행 |

---

## 7. 다음 단계

1. **이 Plan 승인** 후 Design 단계 진행
2. Design에서 각 모듈의 상세 인터페이스 정의
3. Do 단계에서 구현 시작 (Phase 1부터)

---

*Generated: 2026-02-04*
*Feature: code-review-enhancement*
*Version: 1.0*
