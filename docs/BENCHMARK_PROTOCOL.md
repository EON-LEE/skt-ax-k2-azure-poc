# A.X K2 서빙 재측정 프로토콜 (A100)

- 상태: **구현 완료, GPU 실행 전.** 하네스·작업 템플릿·분석 스크립트를 구현했다. 확인은 합성 데이터 단위 테스트와 셸 구문 검사로만 했다. GPU와 Azure에서는 실행하지 않았다.
- 근거: [검토 결과](report/REVIEW_2026-11-09.md) 8절의 계획을 구현했다. 이 문서와 그 계획이 다르면 이 문서가 구현된 내용이다. 다른 점은 9절에 모았다.
- 단일 원천: 측정 조건은 모두 [`aml/src/bench/slo_protocol.json`](../aml/src/bench/slo_protocol.json)에 있다. 하네스는 이 파일만 읽는다. 이 문서의 수치는 이 파일의 값을 옮긴 것이다.
- 사전 등록: 1라운드 전에 `aml/src/bench/` 전체와 이 문서를 하나의 커밋으로 고정한다. 고정한 뒤 SLO, 도착률 격자, 시드, 판정 규칙을 바꾸면 새 커밋으로 다시 등록하고 그 사실을 공개한다. 무효 실행과 중단된 실행도 모두 공개한다.

## 1. 목적

FP8 A, C2와 NVFP4 D를 반복 측정한다. 고객에게 보여 줄 판단 근거는 다음 순서로 정한다.

1. 기준 SLO에서의 최대 도착률(λ*)
2. λ*에서 출력 100만 토큰당 비용
3. 단일 요청 TPOT
4. 단일 요청 TTFT

λ*는 정상 상태 구간에 시작한 요청 중 90% 이상이 TTFT 기준과 TPOT 기준을 모두 지키는 가장 높은 도착률이다. 실패한 요청은 기준을 지키지 못한 요청으로 센다.

## 2. 구성

| 이름 | 구성 | 노드/GPU | `PHASE_PLAN` 토큰 | 실행 작업 |
|---|---|---|---|---|
| A | FP8 TP8×PP2, Ray | 2/16 | `slo-A` | 라운드 |
| C2 | FP8 DP2×TP8+EP16, Ray DP 백엔드, `--all2all-backend allgather_reducescatter` | 2/16 | `slo-C2` | 라운드 |
| D | NVFP4 TP8, mp 실행기, node0만 사용 | 1/8 | `slo-D` | 라운드 |
| A′ | A를 라운드 끝에 다시 띄운다. 시드는 A와 같다 | 2/16 | `slo-A2` | 라운드 |
| C2-s | C2에서 엔진별 `--max-num-seqs` 64, `--max-num-batched-tokens` 4,096 | 2/16 | `slo-C2s` | 1라운드만 |
| D×2 | 노드마다 D 서버 1개, node0의 `lb_proxy.py`가 부하를 나눈다 | 2/16 | `dx2-D` (비교 기준 `dx1-D`) | D×2 작업 |

- 용량 단계 `cap-A`, `cap-C2`, `cap-D`는 `--max-model-len -1`로 띄워 자동으로 맞춘 최대 컨텍스트와 KV 토큰 수를 기록한다. 1라운드 끝에 실행한다. 성능 비교에는 쓰지 않는다.
- 진단 단계는 `diag-A`, `diag-A-maxlen`, `diag-C1`, `diag-C1rr`, `diag-B0`, `diag-B0dcp`다. 7절에서 설명한다.

## 3. 공통 서버 설정

모든 단계가 같은 플래그를 쓴다. 구성마다 다른 값은 병렬 방식 플래그, C2의 `VLLM_RAY_DP_PACK_STRATEGY=strict`, C2-s의 엔진별 한도, 용량 단계의 `--max-model-len`뿐이다.

| 설정 | 값 | 고정하는 이유 |
|---|---|---|
| `--max-model-len` | 성능 단계 131,072, 용량 단계 -1 | 자동 맞춤을 쓰면 구성마다 값이 달라진다. 인덱서 커널의 블록 수가 이 값에 비례한다(검토 결과 4.3절 A1) |
| `--max-num-seqs` | 128 | 이전 FP8 64, NVFP4 128의 차이를 없앤다 |
| `--gpu-memory-utilization` | 0.90 | 이전 NVFP4의 0.92를 FP8과 맞춘다 |
| `--max-num-batched-tokens` | 8,192 | 이전 측정과 같다 |
| `--api-server-count` | 1 | C2의 기본값 2를 다른 구성과 맞춘다 |
| `--async-scheduling` | 켬 | vLLM이 자동으로 고르던 값을 명시한다 |
| `--compilation-config` | `{"cudagraph_mode": "FULL_AND_PIECEWISE"}` | CUDA graph 방식을 명시한다 |
| `--scheduling-policy`와 파서 | `priority`, `deepseek_v3`, `hermes`, `--enable-auto-tool-choice` | 데모와 같다 |
| 환경 변수 | `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` | 실행기를 명시한다 |
| 환경 변수 | `VLLM_SERVER_DEV_MODE=1` | `POST /reset_prefix_cache`에 필요하다. 측정용 서버에만 준다. 데모 서버에는 주지 않는다 |

- 프리픽스 캐시는 운영 기본값대로 켠다. 측정 지점마다 캐시를 비우고, 게이트로 캐시 적중을 검사한다(5절).
- 비교 단계는 CUDA graph로 시작하지 못하면 실패로 끝난다. `--enforce-eager`로 다시 시도하지 않는다.
- FP8 2노드 단계는 InfiniBand에서만 실행한다. IB 탐침이 IB를 고르지 않았거나 NCCL 통신기 하나라도 `NET/Socket`을 쓰면 그 단계는 실패한다.

## 4. 워크로드

공통 조건은 다음과 같다. `/v1/completions`, temperature 0.6, top_p 0.95, `--ignore-eos`, `--random-range-ratio 0`, 백분위수 50·90·95·99를 쓴다. 모든 측정 지점에서 `--save-detailed`로 요청별 데이터를 저장한다. `--num-warmups`는 쓰지 않는다. 이 옵션은 0번 프롬프트를 다시 보내 캐시 적중을 만든다.

| 워크로드 | 입력/출력 토큰 | 부하 | 도착률 또는 동시성 | 정상 상태 구간 | 요청 수 |
|---|---|---|---|---|---|
| Part R | 1K·8K·32K / 1,024 | 폐쇄형 | 동시 1(4개), 동시 32(64개). 32K·동시 32(32개)는 1라운드만 | 전체 | 표의 값 |
| S1 | 1K / 512 | 폐쇄형 | 동시 32, 128 | 전체 | 동시성 × 4 |
| W1 채팅형 | 1K / 512 | Poisson | 0.25, 0.6, 1.0, 1.5 req/s, 도착 360초 | 60~360초 | max(60, ⌈R×360⌉) |
| W2 RAG형 | 8K / 512 | Poisson | 0.05, 0.12, 0.2, 0.3 req/s, 도착 420초 | 120~420초 | max(40, ⌈R×420⌉) |
| W1-real | 약 1K / 512 | Poisson | 0.6, 1.0 req/s (A와 C2만) | 60~360초 | max(60, ⌈R×360⌉) |
| 장시간 부하 | 1K / 512 | Poisson | 각 구성의 0.8 × λ*(W1, 기준 SLO), 2,700초 | 60~2,700초 | ⌈R×2,700⌉ |
| D×2 격자 | W1, W2와 같다 | Poisson | W1 0.5, 1.2, 2.0, 3.0 / W2 0.1, 0.24, 0.4, 0.6 | W1, W2와 같다 | 같은 규칙 |

- 정상 상태 지표는 구간 안에 시작한 요청으로 계산한다. 출력 tok/s는 구간 안에 생성된 토큰 수(시작 시각 + TTFT + 누적 ITL)를 구간 길이로 나눈다.
- 측정 전에 입력 1K와 8K마다 동시 32 묶음을 1번 보내 커널을 데운다. 이 결과는 버린다.
- W1-real은 `parity_corpus.json`의 실제 한국어·영어 문서에서 1,024토큰 창을 고른다. 창의 시작 토큰이 모두 달라서 프롬프트끼리 캐시 접두사를 공유하지 않는다. 고객 데이터가 있으면 `REALTEXT_JSONL`에 JSONL(`{"prompt": ...}`) 경로를 준다.
- 도착률 격자는 90% goodput 경계의 양쪽을 포함해야 한다. 포함하지 못하면 그 측정은 무효다. 격자를 옮겨 다시 잰다. 격자 밖으로 외삽하지 않는다. 1라운드 뒤 `bench_stats.py`가 제안하는 격자(λ*의 0.6~1.4배, 4단계)로 다시 맞춘다.

## 5. SLO (가정)

**이 값은 가정이다. 1라운드 전에 고객과 합의한다. 고객 기준이 있으면 그것을 쓴다.**

| 워크로드 | 기준 SLO (P90) | 엄격 SLO (P90) |
|---|---|---|
| W1 | TTFT ≤ 3초, TPOT ≤ 100 ms | TTFT ≤ 1초, TPOT ≤ 50 ms |
| W2 | TTFT ≤ 6초, TPOT ≤ 100 ms | TTFT ≤ 3초, TPOT ≤ 50 ms |
| W3 (전체안에서만) | TTFT ≤ 30초, TPOT ≤ 100 ms | 없음 |

`vllm bench serve --goodput`에는 기준 SLO를 준다. 엄격 SLO는 요청별 데이터로 나중에 계산한다.

## 6. 라운드, 순서, 시드

### 6.1 라운드

라운드 하나는 별도의 AML 작업 하나다. 3라운드로 시작한다. 3라운드는 1·2라운드와 다른 노드 쌍에서 실행한다. 결과 꾸러미에 노드 호스트 이름의 해시가 들어 있어 다른 쌍인지 확인할 수 있다.

| 라운드 | 단계 순서 |
|---|---|
| 1 | A → C2 → D → C2-s → A′ → cap-A → cap-C2 → cap-D |
| 2 | A → D → C2 → A′ |
| 3 | A → C2 → D → A′ (장시간 부하 포함) |
| 4 (연장할 때) | A → D → C2 → A′ |
| 5 (연장할 때) | A → C2 → D → A′ |

`PHASE_PLAN=slo-round`를 주면 `ROUND` 값에 따라 위 순서로 펼쳐진다.

### 6.2 시드

- 측정 지점의 시드는 `1000000×ROUND + 10000×WORKLOAD + 100×POINT + REP`다. REP는 0이다.
- WORKLOAD 번호: Part R 1, S1 2, W1 3, W2 4, W1-real 5, 장시간 부하 6, W3 7.
- POINT는 워크로드 전체 목록(도착률 격자 또는 Part R 목록)에서의 순번이다. 일부 지점만 실행하는 A′와 C2-s도 같은 시드를 쓴다.
- 같은 라운드의 모든 구성이 같은 프롬프트를 받는다. A′는 A와 같은 시드를 쓴다.
- 워밍업 시드는 `9000000 + 10000×ROUND + 100×POINT`다.
- 도착률 지점의 실행 순서는 `1000000×ROUND + 10000×WORKLOAD + 99`로 섞는다. 같은 라운드에서는 모든 구성이 같은 순서다.
- 게이트에 걸린 지점은 같은 시드로 1번 다시 실행한다. 두 시도를 모두 보고한다.
- 진단 작업은 `ROUND=0`이라 라운드 시드와 겹치지 않는다.

## 7. 단계 하나의 실행 순서

`entry.sh`가 `aml/src/bench/protocol.sh`를 읽어 다음 순서로 실행한다. `PHASE_PLAN`에 측정 단계가 없으면 이 파일은 아무것도 하지 않는다.

1. 작업 시작 시 모든 노드에서 계측을 켠다. nvidia-smi가 1초마다 클럭, 전력, 온도, 사용률, 스로틀 원인을 기록한다. InfiniBand 포트 카운터는 5초마다 기록한다. NCCL은 프로세스마다 로그 파일을 남긴다(`NCCL_DEBUG_FILE`).
2. 단계가 쓰는 노드의 상태를 확인한다. GPU 계산 프로세스가 없어야 하고, GPU 메모리 사용량이 2 GiB 미만이어야 한다. 남은 Ray placement group이 없어야 하고, 8000번 포트가 비어 있어야 한다. 180초 안에 정리되지 않으면 단계는 실패한다.
3. FP8 2노드 단계에서 IB 탐침이 InfiniBand를 고르지 않았으면 단계는 실패한다.
4. `vllm serve`를 별도 세션(`setsid`)으로 띄운다. 정상 상태가 되지 않으면 단계는 실패한다. eager로 다시 시도하지 않는다.
5. 서버 로그에서 실제 설정을 추출한다. 엔진 설정 줄 전체, 비동기 스케줄링, API 서버 수, 실행기, PP 층 분할, DP 배치, CUDA graph 방식, DCP, 전문가 배치, KV 용량, 가중치 적재 시간을 저장한다. `max_model_len`이 고정값과 다르거나, CUDA graph가 꺼졌거나, 비동기 스케줄링이 꺼졌으면 단계는 실패한다.
6. FP8 2노드 단계는 NCCL 로그를 검사한다. 단계 시작 뒤 생긴 모든 노드의 로그에서 `NET/Socket`이 0건이고 IB 전송이 1건 이상이어야 한다.
7. 단계에 정해진 클라이언트 스위트를 실행한다. 순서는 parity → warmup → partr → slo → realtext → soak → paritylong이다. A′는 parity, warmup, partr, W1 2개 지점, paritylong만 실행한다. C2-s는 warmup, Part R의 동시 32 지점, W1 2개 지점만 실행한다.
8. 각 측정 지점은 다음 순서로 실행한다.
   1. 모든 엔진의 실행 중·대기 요청이 0이 될 때까지 기다린다.
   2. `POST /reset_prefix_cache`로 캐시를 비운다.
   3. `/metrics`를 저장한다.
   4. `vllm bench serve`를 실행한다. 이 동안 `/metrics`를 1초마다 엔진별로 기록한다. 엔진 값을 더하지 않는다.
   5. `/metrics`를 다시 저장하고 엔진별 카운터 증가량을 계산한다.
   6. 게이트를 적용한다(8절). 무효이면 같은 시드로 1번 다시 실행한다.
9. 서버의 프로세스 그룹 전체에 SIGINT를 보내고, 90초 안에 끝나지 않으면 SIGKILL을 보낸다. 그 뒤 2번의 정리 상태를 다시 기록한다.

## 8. 유효성 게이트

다음 중 하나라도 해당하면 그 지점은 **무효**다.

- 시작 전에 서버가 유휴 상태가 아니었다.
- 캐시 초기화가 실패했다.
- `vllm bench serve`가 0이 아닌 코드로 끝났다.
- 폐쇄형 지점에서 실패한 요청이 1개 이상이다.
- 실패 요청이 없는데 서버가 센 프롬프트 토큰 수가 클라이언트 값과 0.1% 넘게 다르다.
- 조회한 프롬프트 토큰 중 1% 넘게 프리픽스 캐시에 적중했다.
- 지점 전후의 `/metrics`가 없다.
- 요청별 데이터(`--save-detailed` 결과)가 없다.
- 지점 실행 중 어느 노드에서든 GPU 하드웨어 감속, 열 감속, 전력 브레이크(스로틀 원인 마스크 0xE8)가 나타났다.

다음은 지점을 **표시**만 하고 무효로 하지 않는다.

- 선점이 1번 이상 있었다: capacity-bound.
- 어느 엔진이든 KV 사용률이 0.95 이상이었거나, 폐쇄형 지점의 최대 실행 요청 수가 동시성보다 작았다: kv-bound. 이 지점은 속도 판정에서 뺀다.
- 개방형 지점의 실패 요청: SLO 미달로 센다.
- 소프트웨어 전력 상한(0x4)에 걸린 표본 비율: 기록만 한다.

라운드는 A, 가운데 두 구성, A′의 모든 측정 스위트가 결과를 냈고, 중단되지 않았고, 계획한 모든 지점에 유효한 시도가 있을 때만 쓴다. 3라운드에는 장시간 부하도 필요하다. 하나라도 빠지면 그 라운드 전체를 쓰지 않는다. C2-s와 용량 단계는 라운드의 유효성을 정하지 않는다.

## 9. 통계와 판정 규칙

### 9.1 지표

- 반복의 단위는 서버 1회 기동이다.
- λ*: 기준을 지킨 가장 높은 격자 도착률과 그다음 도착률 사이를 선형 보간한다. 가장 높은 도착률도 기준을 지키면 "격자가 너무 낮다", 가장 낮은 도착률도 기준을 못 지키면 "격자가 너무 높다"로 기록하고 λ*를 내지 않는다. 전체 격자를 실행한 단계(A, C2, D와 D×2 작업의 두 단계)만 λ*를 계산한다. A′와 C2-s는 W1 지점 2개만 실행하므로 λ*가 없다.
- 비용: 노드시간당 가격 × 노드 수 ÷ (λ*에서의 출력 tok/s × 3,600) × 10⁶. 라운드마다 계산한다. 가동률 50%를 가정하면 비용은 2배다.
- 가격: 영국 남부 Azure 소매가로 종량제 40.96달러, Spot 12.89달러(노드시간당)다. 저우선순위 노드는 2026-03-31부터 Spot 요금으로 과금된다. 발표일에 가격을 다시 조회한다.
- 주요 지표: λ*(W1, W2, 기준 SLO), W1 λ*에서의 비용(Spot, 종량제), Part R 단일 요청 TPOT(1K·8K·32K), 단일 요청 TTFT(8K·32K). 보조 지표 목록은 `slo_protocol.json`의 `metrics`에 있다.
- P99는 정상 상태 구간의 요청이 1,000개 이상일 때만 보고한다. 기본 격자의 W1 1.5 req/s는 구간 안 요청이 약 450개라 P99를 보고하지 않는다.

### 9.2 신뢰구간

- 실행 안: 30초 분량의 연속 요청을 한 블록으로 하는 이동 블록 부트스트랩을 2,000번 반복한다. 지점의 요청 수가 충분했는지 확인하는 용도다.
- 라운드 사이: 라운드 k의 로그비 r_k = ln(X_k / 기준_k)를 쓴다. A′도 그 지표를 쟀으면 기준_k = √(A_k × A′_k)이고, 아니면 기준_k = A_k다. 비율 exp(평균 r)과 95% 신뢰구간 exp(평균 ± t × 표준편차 / √n)을 보고한다. t는 n=3에서 4.303, n=4에서 3.182, n=5에서 2.776이다. 라운드별 최솟값과 최댓값도 보고한다.
- 3라운드는 서로 다른 재표집이 10개뿐이라 라운드 사이에는 부트스트랩을 쓰지 않는다.

### 9.3 판정

- δ_min = max(ln 1.05, 2 × 최대 |ln(A′_k / A_k)|)다. A′가 재지 않은 지표(λ*, W2, 비용)는 A′가 잰 지표들의 가장 큰 δ_min을 쓴다.
- **나음:** 다음 세 조건을 모두 만족한다.
  1. 모든 라운드에서 X가 앞선다.
  2. 95% 신뢰구간이 1을 포함하지 않는다.
  3. |평균 r| ≥ δ_min이다.
- **나쁨:** 같은 세 조건을 반대 방향으로 만족한다.
- **동등:** 95% 신뢰구간 전체가 [0.95, 1.05] 안에 있다.
- **판정 불가:** 나머지. 신뢰구간을 함께 쓴다.
- kv-bound 또는 capacity-bound 지점의 지표는 판정하지 않는다.

### 9.4 라운드 연장

유효한 라운드가 3개 모이면 다음을 검사한다. 하나라도 해당하면 라운드를 추가한다. 최대 5라운드다.

- A′가 잰 주요 지표에서 라운드 간 ln(A′/A)의 표준편차가 2%를 넘는다.
- 주요 지표의 95% 신뢰구간 반폭이 5%를 넘는다.

## 10. 출력 동등성

| 검사 | 입력 | 요청 |
|---|---|---|
| P1 | 문서 16개(한국어 8개, 영어 8개)를 각각 2,048토큰으로 자른다 | `prompt_logprobs: 20`, `max_tokens: 1`, temperature 0 |
| P1 장문 | 문서 16개를 정한 순서와 역순으로 이어 붙여 32,768토큰으로 자른 문서 2개 | P1과 같다. 단계의 마지막에 실행한다 |
| P2 | 문서마다 0~511번 토큰과 1,024~1,535번 토큰, 모두 32개 | 그리디 256토큰, 한 번에 하나씩 |

- 비교 기준은 같은 라운드의 A다. 진단 작업에서는 `diag-A`다.
- FP8 구성(C2, C2-s)의 허용 기준은 다음 세 가지를 모두 만족하는 것이다.
  1. 평균 |Δ 선택 토큰 log-prob| ≤ 2 × (A′ 대 A) + 0.005 nats
  2. top-1 일치율 ≥ (A′ 대 A) − 0.01
  3. 16번째 토큰 전에 출력이 달라진 프롬프트가 2개 이하
- 3번을 만족하지 못하면 구성 설정이 잘못된 것으로 본다.
- NVFP4(D)는 차이만 보고한다. 품질 동등성은 별도 정확도 평가(KoBALT 200문항, CLIcK 300문항, 대응 McNemar 검정, 비열등 허용 폭 −3점)로 판정한다. 이 평가는 이 하네스에 포함하지 않았다.
- `parity_corpus.json`에는 저작권 보호를 받지 않는 글만 넣었다. 한국어는 법령 5개(저작권법 제7조로 보호 대상이 아니다)와 1962년 이전에 사망한 작가의 소설 3개다. 영어는 미국 건국 문서와, 1930년까지 사망한 작가가 1892년까지 출판한 공유 저작물이다. 문서마다 출처와 근거를 기록했다.

## 11. D×2 검증

1. `dx1-D`: node0에서 D를 혼자 띄우고 W1·W2 격자를 잰다.
2. `dx2-D`: node0과 node1에서 D를 하나씩 띄운다. node1의 서버는 Ray 분리 actor가 별도 세션으로 띄우고 멈춘다.
3. node0의 `lb_proxy.py`(127.0.0.1:8100)가 요청마다 처리 중 요청이 가장 적은 서버를 고른다. 같으면 차례대로 고른다. 스트리밍 응답은 받는 대로 전달한다.
4. 프록시는 `/health`, `/lb/stats`, `/metrics`(각 표본에 `replica` 라벨 추가), `/reset_prefix_cache`(두 서버에 모두 전달)를 직접 처리한다.
5. 통과 기준은 W1과 W2 모두 λ*(dx2) ≥ 1.9 × λ*(dx1)이다(기준 SLO).

NVFP4 품질 검증을 통과한 뒤 실행한다.

## 12. 진단 실험

`serving-diag-nd96.yml`이 라운드 전에 다음을 확인한다. 단계마다 1회 실행이므로 결과는 경향이며 통계적으로 확정된 차이가 아니다.

- **커널 시험:** DSA 포트의 커널 시험과 인덱서 점수 저장 회귀 시험(`aml/src/bench/test_dsa_port_regression.py`)을 GPU에서 실행한다.
- **A1:** `diag-A`(131,072)와 `diag-A-maxlen`(자동 맞춤, 약 262,144)의 동시 32 ITL, 출력 tok/s, 단일 요청 TPOT을 비교한다.
- **A8:** `diag-C1`(linear 전문가 배치)과 `diag-C1rr`(`--expert-placement-strategy round_robin`)을 비교한다.
- **DCP:** `diag-B0`(TP16)와 `diag-B0dcp`(`--decode-context-parallel-size 2`)를 비교한다. 서버가 뜨는지, KV 토큰 수가 뚜렷이 늘었는지, DCP 그룹 로그가 있는지, `diag-A` 대비 출력이 같은지, 짧은 부하가 끝나는지를 본다. 다섯 가지를 모두 만족해야 DCP를 지원한다고 본다. 서버가 뜨지 않아도 실패를 기록하고 작업은 계속된다.
- **32K prompt_logprobs:** `diag-A`의 P1 장문이 메모리 부족 없이 끝나는지 확인한다. 실패하면 1라운드 전에 장문 P1을 빼고 그 사실을 사전 등록에 적는다.

## 13. 작업 렌더링과 제출

> **경고:** 아래 제출 명령은 GPU 노드 비용을 발생시킨다. 진단 실험과 최소안 예산이 승인된 뒤에만 실행한다. 지금 승인된 범위는 GPU를 쓰지 않는 작업뿐이다.

전제 조건은 다음과 같다.

- `aml/src` 변경이 커밋되어 원격 저장소에 push되어 있어야 한다. `render_job.py`는 커밋되지 않았거나 원격에 없는 커밋을 거부한다.
- Azure CLI의 `ml` 확장이 설치되어 있고 로그인되어 있어야 한다.
- 아래의 `<workspace>`, `<resource-group>`, `<job-name>`, `<mlflow-tracking-uri>`는 자리 표시자다. 실제 값을 이 문서나 저장소에 적지 않는다.

순서는 다음과 같다. 명령은 bash 기준이다. 실행해서 확인한 명령이 아니다.

1. 템플릿을 렌더링한다. 결과는 `aml/jobs/.rendered/`에 생긴다. 새 템플릿은 `aml/src`와 `aml/src/bench`의 파일을 고정 커밋에서 내려받아 파일마다 해시를 확인한다.

   ```bash
   python aml/render_job.py aml/jobs/serving-diag-nd96.yml aml/jobs/slo-round-nd96.yml aml/jobs/slo-nvfp4-2replica-nd96.yml
   ```

2. 진단 작업을 제출한다.

   ```bash
   az ml job create --file aml/jobs/.rendered/serving-diag-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   ```

3. 진단 결과로 공통 `--max-model-len`, 대표 구성, 장문 P1 포함 여부를 확정한다. 바뀐 값이 있으면 `slo_protocol.json`을 고치고 사전 등록 커밋을 만든 뒤 1번부터 다시 렌더링한다.
4. 1라운드를 제출한다.

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=1
   ```

5. 작업이 끝나면 결과를 꺼낸다. 저장소가 비공개 정책이라 같은 작업 공간의 CPU 작업이 결과를 작업 기록으로 돌려준다. `sed`로 만든 템플릿 사본은 작업 이름이 들어 있으므로 커밋하지 않는다.

   ```bash
   sed 's/__RUN__/<job-name>/' aml/jobs/collect-slo-outputs.yml > aml/jobs/collect-slo-outputs-<job-name>.yml
   python aml/render_job.py aml/jobs/collect-slo-outputs-<job-name>.yml
   az ml job create --file aml/jobs/.rendered/collect-slo-outputs-<job-name>.yml --workspace-name <workspace> --resource-group <resource-group>
   ```

6. 수집 작업이 끝나면 꾸러미를 내려받아 분석한다. 분석 결과에는 다음 라운드의 격자 제안(`next_grid`)과 라운드 연장 판단(`extension`)이 들어 있다.

   ```bash
   AXK2_MLFLOW_URI=<mlflow-tracking-uri> python aml/fetch_results.py show <collect-job-name> > round1.json
   python aml/src/bench/bench_stats.py analyze round1.json --out slo-evidence.json --md slo-summary.md
   ```

7. 2라운드를 제출한다. 격자를 다시 맞췄으면 `SLO_RATES`(모든 구성 공통) 또는 `SLO_RATES_<구성>`(예: `SLO_RATES_C2`)을 함께 준다. A′는 A의 격자를 쓴다.

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=2 --set "environment_variables.SLO_RATES=W1=<r1>,<r2>,<r3>,<r4>;W2=<r1>,<r2>,<r3>,<r4>"
   ```

8. 3라운드는 1·2라운드와 다른 노드 쌍에서 실행한다. 장시간 부하를 위해 구성별 0.8 × λ*(W1, 기준 SLO)를 `SOAK_RATES`로 준다. 값이 없으면 장시간 부하가 실행되지 않아 3라운드가 무효가 된다.

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=3 --set "environment_variables.SOAK_RATES=A=<req/s>,C2=<req/s>,D=<req/s>"
   ```

9. 3라운드까지 5·6번으로 분석한다. 모든 라운드의 꾸러미를 함께 준다(`bench_stats.py analyze round1.json round2.json round3.json ...`). `extension.recommended`가 참이면 `ROUND=4`, 필요하면 `ROUND=5`를 제출한다.
10. NVFP4 품질 검증을 통과했으면 D×2 작업을 제출하고 5·6번으로 분석한다.

    ```bash
    az ml job create --file aml/jobs/.rendered/slo-nvfp4-2replica-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
    ```

## 14. 비용 (추정, 미실행)

[검토 결과](report/REVIEW_2026-11-09.md) 8.6절의 추정을 옮겼다. 가격은 Azure 소매가(영국 남부)로 종량제 40.96달러, Spot 12.89달러(노드시간당)다.

| 항목 | 2노드 할당 시간 | 노드시간 | Spot | 종량제 |
|---|---|---|---|---|
| 진단 실험 1회: A, A(`--max-model-len` 131,072), C1, C1(round_robin)의 기존 측정 재현, DCP 확인, 버그 GPU 재현 | 약 2시간 | 약 4 | 약 52달러 | 약 164달러 |
| 최소안 ① 3라운드: A, C2, D, A′ | 약 20.8시간 | 약 41.5 | 약 535달러 | 약 1,700달러 |
| 최소안 ② 정확도 부분 평가 1회 (NVFP4·FP8) | 약 3시간 | 약 6 | 약 77달러 | 약 246달러 |
| 최소안 ③ 실제 텍스트 W1 (A, C2, 3라운드) | 약 1.5시간 | 약 3 | 약 39달러 | 약 123달러 |
| 최소안 ④ C2-s 1회 | 약 0.6시간 | 약 1.2 | 약 15달러 | 약 49달러 |
| 최소안 ⑤ D×2 검증 1회 | 약 1.5시간 | 약 3 | 약 39달러 | 약 123달러 |
| **최소안 합계 (①~⑤)** | **약 27.4시간** | **약 54.7** | **약 705달러** | **약 2,240달러** |
| 추가 라운드 (필요할 때, 1회당) | 약 5.8시간 | 약 11.7 | 약 151달러 | 약 479달러 |
| 전체안 | 약 38시간 | 약 76 | 약 980달러 | 약 3,110달러 |

구현하면서 위 추정에 없는 단계가 생겼다. 모두 추정이며 측정하지 않았다.

- 1라운드 끝의 용량 단계 3개: 서버 기동 3번과 짧은 기능 점검이다. 기동 시간을 보면 0.5시간 이내로 추정한다.
- 라운드 작업마다 커널 시험: 이전 실행에서 약 2분이 걸렸다.
- 진단 작업의 `diag-B0`(DCP 비교 기준): 0.3시간 이내로 추정한다.
- 라운드 작업 제한 시간은 12시간, 진단 작업은 5시간, D×2 작업은 6시간이다.

## 15. 사전 등록 전에 정할 사항과 한계

### 15.1 사전 등록 전에 정할 사항

1. **SLO:** 5절의 값은 가정이다.
2. **공통 `--max-model-len`:** 131,072는 예시 값이다. 고객의 최대 입력과 출력 길이에 맞추고, C2의 상한 238,656 이하로 정한다.
3. **C2-s의 엔진별 시퀀스 한도:** 검토 결과 8.2절은 "엔진별 배치 한도 절반(4,096토큰, 32시퀀스)"으로 적었다. 32는 이전 값 64의 절반이다. 공통 값 128의 절반은 64이므로 64로 구현했다. 32로 하려면 `slo_protocol.json`의 `layouts.C2s.max_num_seqs`를 바꾼다.
4. **스로틀 게이트:** 측정 방법 검토는 "유휴 외의 스로틀 원인이 하나라도 있으면 무효"를 제안했다. 부하 중인 A100은 소프트웨어 전력 상한에 흔히 걸려 거의 모든 지점이 무효가 될 수 있다(추정). 그래서 하드웨어 감속·열 감속·전력 브레이크만 무효로 하고, 전력 상한은 비율만 기록하도록 구현했다. 바꾸려면 `gates.throttle`을 고친다.
5. **명시한 설정의 일치 여부:** `--async-scheduling`, `--compilation-config`, `--all2all-backend`, `--api-server-count 1`이 이전 실행의 자동 선택과 같은지는 진단 실험의 실제 설정 기록으로 확인한다.

### 15.2 한계

- GPU와 Azure에서 실행하지 않았다. 확인은 합성 데이터 단위 테스트, 가짜 백엔드를 쓴 부하 분산기 시험, 셸 구문 검사로만 했다.
- 32K 문서의 `prompt_logprobs`는 로짓 메모리가 커서 서버가 멈출 수 있다(추정). 그래서 단계 마지막에 실행하고, 진단 실험에서 먼저 확인한다.
- `dx2-D`에서 node1 서버의 실제 설정은 실행 중에 검사하지 않는다. 로그는 node1의 출력 폴더에 남고 수집 단계에서 꾸러미에 들어간다.
- NCCL 검사는 노드별 전송 방식 건수와 통신기 구성원(`Init COMPLETE`)을 센다. 채널 연결 줄을 통신기별로 정확히 나누지는 않는다. `NET/Socket` 0건 조건에는 영향이 없다.
- nvidia-smi 시각은 UTC로 해석한다.
- 스텝별 스케줄 토큰 수는 `vllm:iteration_tokens_total` 히스토그램의 엔진별 1초 합계·개수로만 남는다. 스텝별 CUDA graph 방식은 기록하지 않는다.
- 전문가별 라우팅 토큰 수(EPLB 균형 로그)는 구현하지 않았다.
- 이전 단계(`epc-*`, `docsweep` 등)는 이전 기록과의 비교를 위해 기존 동작을 유지한다. 다만 측정 지점마다 다른 시드를 쓰도록 고쳤다. 이 단계들은 시작에 실패하면 여전히 `--enforce-eager`로 다시 시도하고, `/metrics` 표본은 여전히 엔진 값을 더한다.

## 16. 파일

| 파일 | 내용 |
|---|---|
| `aml/src/bench/slo_protocol.json` | 구성, 공통 플래그, 워크로드, 격자, SLO, 시드, 게이트, 판정 규칙, 가격 |
| `aml/src/bench/protocol.sh` | `entry.sh`가 읽는 측정 단계 실행부: 계측, 정리 상태 확인, 실제 설정·NCCL 검사, 프로세스 그룹 종료, D×2 복제본 |
| `aml/src/bench/slo_suites.py` | 단계 설정 생성, 측정 스위트(parity, warmup, partr, slo, realtext, soak, paritylong), 엔진별 `/metrics` 시계열, 게이트 |
| `aml/src/bench/bench_stats.py` | 꾸러미 생성(작업 공간 안), 라운드 분석과 판정, Markdown 요약 |
| `aml/src/bench/lb_proxy.py` | D×2용 부하 분산기 |
| `aml/src/bench/parity_corpus.json` | 출력 동등성과 W1-real용 공유 저작물·법령 텍스트 |
| `aml/jobs/serving-diag-nd96.yml` | 진단 작업 |
| `aml/jobs/slo-round-nd96.yml` | 라운드 작업 |
| `aml/jobs/slo-nvfp4-2replica-nd96.yml` | D×2 작업 |
| `aml/jobs/collect-slo-outputs.yml` | 결과 수집 작업 |

`aml/src/bench/`는 데모 작업이 받는 스크립트 묶음에 들어가지 않는다. 고정 커밋에서 파일을 받는 측정 작업만 이 폴더를 내려받는다.
