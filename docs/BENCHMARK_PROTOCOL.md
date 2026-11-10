# A.X K2 서빙 재측정 프로토콜 (A100)

- 상태: **구현 완료, GPU 실행 전**(라운드, mp-multinode 실행, PD 재실행 기준). 하네스·작업 템플릿·분석 스크립트를 구현했다. 스크리닝 작업 3개는 main 5dc653b로 GPU에서 실행했다. 그 결과로 추가한 mp-multinode 실행(6.8절), PD 재실행(6.5절)과 그 밖의 변경은 합성 데이터 단위 테스트와 셸 구문 검사로만 확인했다. 라운드는 아직 실행하지 않았다.
- 근거: [검토 결과](report/REVIEW_2026-11-09.md) 8절의 계획을 구현했다. 이 문서와 그 계획이 다르면 이 문서가 구현된 내용이다. 다른 점은 17.1절에 모았다.
- 단일 원천: 측정 조건은 모두 [`aml/src/bench/slo_protocol.json`](../aml/src/bench/slo_protocol.json)에 있다. 하네스는 이 파일만 읽는다. 이 문서의 수치는 이 파일의 값을 옮긴 것이다.
- 사전 등록: 스크리닝(6절)으로 라운드 후보를 고른 뒤, 1라운드 전에 `aml/src/bench/` 전체, 이 문서, 고른 `ROUND_LAYOUTS` 값을 하나의 커밋으로 고정한다. 고정한 뒤 SLO, 도착률 격자, 시드, 판정 규칙을 바꾸면 새 커밋으로 다시 등록하고 그 사실을 공개한다. 무효 실행과 중단된 실행도 모두 공개한다.

## 1. 목적

FP8 A와 라운드 후보(기본값은 FP8 C2와 NVFP4 D)를 반복 측정한다. 라운드 전에는 스크리닝으로 A100에서 가능한 서빙 구성을 모두 한 번씩 측정하고, 사전 등록한 규칙으로 후보를 고른다(6절). 고객에게 보여 줄 판단 근거는 다음 순서로 정한다.

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
- 진단 단계는 `diag-A`, `diag-A-maxlen`, `diag-C1`, `diag-C1rr`, `diag-B0`, `diag-B0dcp`다. 13절에서 설명한다.
- 위 표의 C2와 D는 기본 후보다. 스크리닝의 선택 규칙이 다른 구성을 고르면 `ROUND_LAYOUTS`로 후보를 바꾼다(6.6절, 7.1절). 스크리닝 구성은 6.3절에 있다.

## 3. 공통 서버 설정

모든 단계가 같은 플래그를 쓴다. 구성마다 다른 값은 병렬 방식 플래그, Ray DP 구성의 `VLLM_RAY_DP_PACK_STRATEGY=strict`, DP×EP 구성의 엔진별 배치 토큰 한도(6.3절), 민감도 실행(C2-s 등)의 엔진별 한도, 용량 단계의 `--max-model-len`뿐이다.

| 설정 | 값 | 고정하는 이유 |
|---|---|---|
| `--max-model-len` | 성능 단계 131,072, 용량 단계 -1 | 자동 맞춤을 쓰면 구성마다 값이 달라진다. 인덱서 커널의 블록 수가 이 값에 비례한다(검토 결과 4.3절 A1) |
| `--max-num-seqs` | 128 | 이전 FP8 64, NVFP4 128의 차이를 없앤다 |
| `--gpu-memory-utilization` | 0.90 | 이전 NVFP4의 0.92를 FP8과 맞춘다 |
| `--max-num-batched-tokens` | 8,192. DP×EP 구성의 엔진별 값은 6.3절의 규칙으로 정한다 | 이전 측정과 같다. DP×EP 구성의 규칙은 MoE 작업 공간 때문이다 |
| `--api-server-count` | 1 | C2의 기본값 2를 다른 구성과 맞춘다 |
| `--async-scheduling` | 켬 | vLLM이 자동으로 고르던 값을 명시한다 |
| `--compilation-config` | `{"cudagraph_mode": "FULL_AND_PIECEWISE"}` | CUDA graph 방식을 명시한다 |
| `--scheduling-policy`와 파서 | `priority`, `deepseek_v3`, `hermes`, `--enable-auto-tool-choice` | 데모와 같다 |
| 환경 변수 | `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` | 실행기를 명시한다 |
| 환경 변수 | `VLLM_SERVER_DEV_MODE=1` | `POST /reset_prefix_cache`에 필요하다. 측정용 서버에만 준다. 데모 서버에는 주지 않는다 |

- 프리픽스 캐시는 운영 기본값대로 켠다. 측정 지점마다 캐시를 비우고, 게이트로 캐시 적중을 검사한다(9절).
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
| W1-real | 약 1K / 512 | Poisson | 0.6, 1.0 req/s (FP8 구성만) | 60~360초 | max(60, ⌈R×360⌉) |
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

## 6. 스크리닝

라운드 전에 A100에서 가능한 서빙 구성을 한 번씩 띄워 측정한다. 이 단계를 스크리닝이라고 한다. 스크리닝 결과와 사전 등록한 선택 규칙(6.6절)으로 라운드에서 A와 비교할 후보를 정한다. 구성마다 1회만 측정하므로 결과는 경향이다. 통계적으로 확정된 차이가 아니다.

### 6.1 작업

| 작업 템플릿 | 노드 | `PHASE_PLAN` | 예상 시간 (추정) | 제한 시간 |
|---|---|---|---|---|
| `serving-screen-fp8-nd96.yml` | 2 | `scr-A` `scr-A-auto` `scr-C1` `scr-C1rr` `scr-B0` `scr-C2` `scr-C4` `scr-C8` `scr-A4` `scr-C16` `scr-B0dcp` | 약 7시간 | 12시간 |
| `serving-screen-fp8-dp-nd96.yml` | 2 | `scr-A` `scr-C4m` `scr-C8m` `scr-C16m` | 약 3.2시간 | 6시간 |
| `serving-screen-nvfp4-nd96.yml` | 1 | `scr-D` `scr-E` `scr-Err` `scr-D24` `scr-D42` `scr-D8` | 약 4.5시간 | 8시간 |
| `serving-screen-nvfp4-2node-nd96.yml` | 2 | `scr-Dx1` `scr-Dx2` `scr-PD` | 약 4시간 | 7시간 |
| `serving-screen-nvfp4-pd-nd96.yml` | 2 | `scr-Dref` `scr-PD` | 약 2시간 | 3시간 |

- 예상 시간은 10-21 측정의 기동 시간(구성당 4~8분)과 Part R 시간(약 20분)으로 추정했다. 구성 하나는 35~45분, 도착률 격자를 실행하는 2노드 NVFP4 구성은 60~70분이 걸린다고 본다. `-1`로 다시 띄울 것으로 예상하는 C16과 D8은 45~50분, 기동 실패를 예상하는 B0dcp는 약 10분으로 잡았다. FP8 DP 작업은 스크리닝에서 잰 단계 시간(A 약 36분, 다른 구성 약 37분)에 C16m의 재기동 약 15분을 더했다. PD 재실행 작업은 Dref 약 15분, NIXL 사전 점검 약 5분, PD 60~70분으로 잡았다.
- 스크리닝 작업은 모두 `ROUND=0`이다. 시드가 라운드의 시드와 겹치지 않고, 같은 지점은 모든 구성에서 같은 프롬프트를 받는다.
- FP8 작업은 `RUN_KERNEL_TESTS=1`이다. DSA 포트 커널 시험과 회귀 시험(`aml/src/bench/test_dsa_port_regression.py`)을 먼저 실행한다. FP8 DP 작업은 이 시험을 다시 실행하지 않는다.
- 각 작업의 첫 단계가 출력 동등성의 기준이다. FP8 작업과 FP8 DP 작업은 각각 자기 `scr-A`, NVFP4 1노드 작업은 `scr-D`, NVFP4 2노드 작업은 `scr-Dx1`, PD 재실행 작업은 `scr-Dref`이다.
- 실패하거나 `-1`로 다시 띄울 것으로 예상하는 구성은 계획의 끝에 둔다. FP8 작업의 C16과 B0dcp, FP8 DP 작업의 C16m, NVFP4 1노드 작업의 D8, 2노드 작업과 PD 재실행 작업의 PD다. 이 구성이 노드를 정리되지 않은 상태로 남겨도 앞에서 측정한 구성에는 영향이 없다.
- 단계가 실패해도 작업은 다음 단계로 넘어간다. 실패 이유는 단계 결과와 결과 꾸러미(14절)에 남는다. 단계를 시작할 때 노드가 정리되지 않았으면 회수를 한 번 실행한다(8절 2번).

### 6.2 A100 서빙 패턴 지도

| 패턴 | 상태 | 근거 또는 이유 |
|---|---|---|
| FP8 TP8×PP2 (A) | 10-21 측정, 라운드 기준, `scr-A` | 현재 제안 구성 |
| FP8 TP8×PP2, `--max-model-len -1` | `scr-A-auto` | 인덱서 격자가 최대 길이에 비례하는 효과(A1) 확인 |
| FP8 TP8×PP2+EP8 (C1), round_robin 전문가 배치 (C1rr) | C1은 10-21 측정, 둘 다 스크리닝 | 전문가 배치 효과(A8) 확인 |
| FP8 TP16 (B0), TP16+DCP 2 (B0dcp) | B0는 10-21 측정, 둘 다 스크리닝 | 포팅한 sparse MLA가 DCP를 지원하는지 확인한다. B0dcp는 기동 실패를 예상한다(6.7절) |
| FP8 DP2×TP8+EP16 (C2) | 10-21 측정, 기본 라운드 후보, 스크리닝 | - |
| FP8 DP4×TP4+EP16 (C4), DP8×TP2+EP16 (C8), DP16×TP1+EP16 (C16) | Ray DP 실행은 기동 실패(측정). mp-multinode 실행 `scr-C4m`, `scr-C8m`, `scr-C16m`을 스크리닝 | 6.8절. 엔진별 배치 토큰은 6.3절의 규칙을 따른다. C16은 131,072토큰을 담지 못할 것으로 추정한다(6.7절) |
| FP8 TP4×PP4 (A4) | 스크리닝 | - |
| NVFP4 TP8 (D), TP8+EP8 (E), round_robin 배치 (Err) | D와 E는 10-21 측정, 셋 다 스크리닝 | - |
| NVFP4 DP8×TP1+EP8 (D8), DP2×TP4+EP8 (D24), TP4×PP2 (D42) | 스크리닝 | D8은 131,072토큰을 담지 못할 것으로 추정한다(6.7절) |
| NVFP4 D×2 (노드마다 서버 1개와 부하 분산기) | `scr-Dx2`, 별도 D×2 작업 | 12절의 통과 기준을 쓴다 |
| NVFP4 PD (프리필·디코드 분리, NixlConnector) | `scr-PD` (실험). 스크리닝에서 기동 실패(측정), PD 재실행 작업에서 다시 측정 | 선택 규칙의 후보가 아니다(6.5절) |
| DeepEP all2all, DeepGEMM, FlashMLA, DBO | A100에서 불가 | SM90 이상이 필요하다 |
| FP8 KV 캐시 | 불가 | 포팅한 TRITON_MLA_SPARSE 백엔드는 auto, fp16, bf16만 지원한다 |
| EAGLE3 추측 디코딩 | 불가 | 포팅한 Triton MLA 디코드는 요청당 쿼리 1개만 처리한다 |
| FP8 2복제본, FP8 PD | 현재 할당량으로 불가 | FP8 서버 하나가 노드 2개를 쓰므로 노드 4개가 필요하다. 저우선순위 할당량은 지역당 300 vCPU, 곧 ND96 3대다 |
| 종량제(전용) 노드 | 불가 | 전용 A100 VM 할당량이 0이다 |
| EPLB, MTP 등 다른 추측 디코딩, 다른 all2all 백엔드 | 범위 밖 | A100에서 되는지 확인하지 않았다 |

### 6.3 스크리닝 구성의 플래그

모든 구성은 3절의 공통 플래그를 쓴다. 예외는 DP×EP 구성의 엔진별 `--max-num-batched-tokens`다(아래 규칙). `--max-num-seqs`와 `--max-num-batched-tokens`는 엔진마다 적용된다. 그래서 DP 구성의 시스템 전체 시퀀스 한도는 C2처럼 DP 크기만큼 커진다.

| 구성 | 체크포인트 | 노드 | 엔진별 배치 토큰 | 병렬 플래그와 환경 변수 |
|---|---|---|---|---|
| A4 | FP8 | 2 | 8,192 | `--tensor-parallel-size 4 --pipeline-parallel-size 4 --distributed-executor-backend ray` |
| C4 | FP8 | 2 | 4,096 | `--data-parallel-size 4 --data-parallel-size-local 2 --tensor-parallel-size 4 --data-parallel-backend ray --enable-expert-parallel --all2all-backend allgather_reducescatter`, `VLLM_RAY_DP_PACK_STRATEGY=strict` |
| C8 | FP8 | 2 | 2,048 | C4와 같고 `--data-parallel-size 8 --data-parallel-size-local 4 --tensor-parallel-size 2` |
| C16 | FP8 | 2 | 1,024 | C4와 같고 `--data-parallel-size 16 --data-parallel-size-local 8 --tensor-parallel-size 1` |
| C4m | FP8 | 2 | 4,096 | node0: `--tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-backend mp --distributed-executor-backend mp --enable-expert-parallel --all2all-backend allgather_reducescatter --data-parallel-address <node0 주소> --data-parallel-rpc-port 13345`. node1: API 서버 플래그를 뺀 같은 플래그와 `--headless --data-parallel-start-rank 2`. `VLLM_HOST_IP`는 각 노드의 주소(6.8절) |
| C8m | FP8 | 2 | 2,048 | C4m과 같고 `--tensor-parallel-size 2 --data-parallel-size 8 --data-parallel-size-local 4`, node1의 시작 랭크 4 |
| C16m | FP8 | 2 | 1,024 | C4m과 같고 `--tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 8`, node1의 시작 랭크 8 |
| E | NVFP4 | 1 | 8,192 | `--tensor-parallel-size 8 --distributed-executor-backend mp --enable-expert-parallel` |
| Err | NVFP4 | 1 | 8,192 | E와 같고 `--expert-placement-strategy round_robin` |
| D8 | NVFP4 | 1 | 1,024 | `--data-parallel-size 8 --data-parallel-size-local 8 --tensor-parallel-size 1 --data-parallel-backend mp --distributed-executor-backend mp --enable-expert-parallel --all2all-backend allgather_reducescatter` |
| D24 | NVFP4 | 1 | 8,192 | D8과 같고 `--data-parallel-size 2 --data-parallel-size-local 2 --tensor-parallel-size 4` |
| D42 | NVFP4 | 1 | 8,192 | `--tensor-parallel-size 4 --pipeline-parallel-size 2 --distributed-executor-backend mp` |
| PD | NVFP4 | 2 | 서버마다 8,192 | 서버마다 D의 플래그와 `--kv-transfer-config`(프리필 `kv_producer`, 디코드 `kv_consumer`, `kv_load_failure_policy` `fail`, `kv_connector` `NixlConnector`). `VLLM_NIXL_SIDE_CHANNEL_PORT=5600`, `VLLM_NIXL_SIDE_CHANNEL_HOST`는 그 서버가 도는 노드의 주소. UCX 환경은 NIXL 사전 점검이 고른 값(6.5절) |

**DP×EP 구성의 배치 토큰 규칙**

- 2노드 FP8 DP×EP 구성은 C2의 전체 배치 예산 16,384토큰을 나눠 쓴다. 엔진별 값은 16,384 ÷ DP다. C2는 8,192(변경 없음), C4는 4,096, C8은 2,048, C16은 1,024다.
- NVFP4는 D8이 1,024, D24가 8,192다.
- 민감도 실행(`C2s`, `C4s`, `C8s`, `C16s`)은 원래 정의를 따른다. A의 전체 값 8,192토큰을 DP로 나눈다.
- 이유: `--all2all-backend allgather_reducescatter`에서 vLLM 0.23은 MoE 순전파마다 모든 DP 랭크의 토큰을 나누지 않고 모은다. 그래서 프로파일 실행이 M = DP × `max_num_batched_tokens` 토큰에 맞춰 Marlin MoE 작업 공간을 잡는다. 이 작업 공간은 BF16에서 토큰당 약 229,376바이트다. 엔진당 8,192토큰이면 C8은 엔진당 KV 토큰이 약 1.9만 개만 남고, C16과 D8은 메모리가 부족하다(추정).
- 근거(vLLM v0.23.0 소스): `vllm/forward_context.py:98-114`, `vllm/distributed/device_communicators/all2all.py:84-122`, `vllm/model_executor/layers/fused_moe/modular_kernel.py:1224-1236`, `vllm/model_executor/layers/fused_moe/marlin_moe.py:726-730`, `vllm/v1/worker/gpu_model_runner.py:6252-6255`.
- 규칙과 근거는 `slo_protocol.json`의 `server.dp_ep_batch_rule`에 있다.

**그 밖의 정의**

- 1라운드 민감도 실행용으로 `C4s`, `C8s`, `C16s`와 mp-multinode용 `C4ms`, `C8ms`, `C16ms`를 정의했다. 엔진별 한도를 DP 크기로 나눠 A와 같은 시스템 전체 한도(128시퀀스, 8,192토큰)를 쓴다. C2-s와 같은 취지다.
- `scr-A-auto` 외의 스크리닝 단계는 공통 `--max-model-len` 131,072로 시작한다.

### 6.4 스크리닝 단계 하나의 순서

1. 서버를 띄운다. 정리 상태 확인과 회수, 실제 설정 검사, FP8 2노드 단계의 NCCL 검사는 8절과 같다.
2. 서버가 vLLM의 "larger than the available KV cache memory" 오류로 끝나면, `--max-model-len -1`로 한 번만 다시 띄운다. 요청 값과 vLLM이 추정한 길이를 시작 기록(`start-<단계>.json`)에 남긴다. 이 구성은 라운드 후보가 될 수 없다.
3. 시작 사실을 기록한다. 정상 상태까지 걸린 시간, `/v1/models`의 `max_model_len`, 엔진별 KV 토큰 수(`vllm:cache_config_info`와 로그의 "GPU KV cache size")다.
   - `max_model_len`은 서버가 실제로 제공한 값(시작 기록, `-1`이면 `/v1/models`)을 쓴다. 엔진 설정 줄의 값은 자동 맞춤 전의 모델 기본값일 수 있다. 예를 들어 `-1`로 다시 띄운 D8은 31,616을 제공했지만 설정 줄에는 262,144가 남았다(측정).
   - node1에서 따로 도는 엔진(D×2의 두 번째 서버, PD의 디코드 서버, mp-multinode의 headless 엔진)은 node0의 로그에 나오지 않는다. 그 엔진의 KV 토큰은 node1의 로그에서 Ray로 읽어 합계에 더한다. 읽지 못하면 끌어온 로그 끝부분에서 찾고, 그래도 없으면 "node0 only"로 표시한다.
4. 짧은 출력 동등성을 잰다. 문서 4개(`ko-01-constitution`, `ko-06-unsu-joeun-nal`, `en-01-us-constitution`, `en-04-pride-and-prejudice`)를 2,048토큰으로 잘라 P1을 실행한다. 같은 문서의 앞 512토큰으로 그리디 128토큰(P2)을 실행한다.
5. 워밍업(입력 1K·8K 묶음)을 실행하고 결과를 버린다.
6. Part R 전체를 측정한다. 동시 1(4개)과 동시 32(64개)를 1K·8K로, 동시 1을 32K로 측정한다. 32K·동시 32(32개)는 출력을 64토큰으로 줄여 실행 중 요청, KV 사용률, 선점만 기록한다.
7. Poisson W1 지점 1개를 측정한다. 0.6 req/s, 도착 360초, 정상 상태 60~360초다. `scr-Dx1`은 W1 격자(0.25~1.5 req/s), `scr-Dx2`와 `scr-PD`는 dx2 격자(0.5~3.0 req/s)를 대신 실행한다.

- 3~7번의 각 단계는 실패해도 기록하고 다음 단계로 넘어간다. 서버가 `/health`에 응답하지 않으면 측정을 멈추고 단계를 실패로 기록한다.
- 측정 지점은 라운드와 같은 게이트를 쓴다(9절).

### 6.5 PD (프리필·디코드 분리)

PD는 실험 단계다. 실패하면 "experimental phase:"로 시작하는 기록을 남기고 작업은 계속된다.

1. node0에 프리필 서버(`kv_producer`)를 띄운다. node1에는 Ray 분리 actor가 디코드 서버(`kv_consumer`)를 띄운다. 두 서버의 플래그는 `--kv-transfer-config`의 역할만 다르다.
2. node0의 `lb_proxy.py --prefill http://127.0.0.1:8000 --decode http://<node1>:8000`이 127.0.0.1:8100에서 요청을 받는다.
3. 완성 요청이 오면 프록시는 먼저 프리필 서버에 요청을 보낸다. 이때 `max_tokens`를 1로, `stream`을 false로 바꾸고, `kv_transfer_params`에 `do_remote_decode: true`를 넣는다. `stream_options`, `min_tokens`, `min_completion_tokens`는 뺀다.
4. 프록시는 프리필 응답의 `kv_transfer_params`(원격 블록, 엔진 ID, 요청 ID, 주소, 포트)를 원래 요청에 붙여 디코드 서버에 보낸다.
5. 디코드 서버는 NIXL(UCX)로 프리필 서버의 KV 캐시를 읽고 응답을 스트리밍한다. 프록시는 받은 스트리밍 응답을 그대로 전달한다.

- `prompt_logprobs` 요청(P1)과 다른 경로(`/tokenize` 등)는 프리필 서버에만 보낸다. 그래서 PD의 P1은 프리필 서버만 검사하고, P2가 전송 경로 전체를 검사한다.
- 프롬프트 토큰 게이트는 프리필 서버의 카운터만 클라이언트 값과 비교한다. 디코드 서버도 같은 프롬프트를 받아 세기 때문이다.
- KV 전송은 NCCL이 아니라 UCX를 쓰므로 NCCL 게이트를 적용하지 않는다. 지점마다 node0의 InfiniBand 송수신량을 기록하므로 KV가 InfiniBand로 갔는지 확인할 수 있다.
- 프록시가 정상 상태가 되면 프록시를 거쳐 확인 요청 1개를 보낸다. 짧은 프롬프트, `max_tokens` 16, 제한 180초다. 그다음 `/lb/stats`에서 `missing_kv_transfer_params`가 0이고 프록시 오류가 없어야 한다. 하나라도 어긋나면 "experimental phase: PD smoke failed: <이유>"를 기록하고 측정 스위트를 실행하지 않는다. 실패 보고에는 프록시 로그의 끝부분(4 KB)도 붙인다.
- 실험 단계의 클라이언트는 `timeout` 5,400초(`BENCH_EXPERIMENTAL_TIMEOUT`) 안에서 실행한다. 제한에 걸리면 그 사실을 단계 결과에 기록한다.
- 서버 2개를 쓰는 단계(`dx2-*`, `scr-Dx2`, `scr-PD`)가 실패하면, node1의 `vllm-<단계>-replica1.log` 끝부분(최대 200줄, 16 KB)을 Ray로 가져와 실패 보고(`node0.<단계>.failure` 또는 `node0.<단계>.client_failure`)에 붙인다. 결과 꾸러미에도 단계마다 이 로그의 끝부분(`replica_tails`)이 들어간다.
**스크리닝의 기동 실패와 재실행**

- 측정: 스크리닝(main 5dc653b)의 `scr-PD`는 기동 중 `register_kv_caches`에서 실패했다. NIXL 오류는 `NIXL_ERR_BACKEND`였고, UCX가 "VRAM memory is detected as host by UCX"를 냈다.
- 확인: 이미지의 nixl-cu12 1.2.0에는 CUDA를 지원하도록 빌드한 UCX 1.21.0이 들어 있다. 다시 빌드할 필요는 없다.
- 측정(PD 재실행, main 2d104e4): NIXL 사전 점검이 두 노드 모두에서 작업 환경 그대로일 때와 `UCX_MEMTYPE_CACHE=n`일 때 실패했다. 길이 0 대조군을 포함한 모든 등록이 `NIXL_ERR_BACKEND`였다. 로더 캐시에는 `libcuda`, `libnvidia-ml`, `libcudart`가 있었고 `libuct_cuda`도 적재되어 있었다.
- 측정: 사전 점검의 환경 기록에 따르면 작업 환경에 `UCX_TLS=tcp`와 `UCX_NET_DEVICES=eth0`이 있었다.
- 추정: 이미지와 이 하네스는 두 값을 설정하지 않는다. 따라서 Azure ML 작업 런타임이 넣은 값으로 본다.
- 원인: `UCX_TLS=tcp`에는 `cuda_copy`가 빠져 있다. 그래서 UCX가 VRAM을 호스트 메모리로 분류했다. 첫 진단의 가설 중 "UCX가 CUDA 지원을 쓰지 못한다"에 해당한다. `UCX_MEMTYPE_CACHE=n`으로도 실패했으므로 메모리 종류 캐시는 원인이 아니다. 길이 0 서술자 가설은 이 결과로 배제되지 않는다. 대책을 적용한 뒤에도 실패하면 디버그 재기동(아래 4번)이 확인한다.
- 대책: 두 서버에 `UCX_TLS=all`과 `UCX_NET_DEVICES=all`을 준다. vLLM의 `docs/features/nixl_connector_usage.md`가 권장하는 값이다. 전송 종류를 제한하는 `UCX_TLS` 목록은 쓰지 않는다. 제한 목록은 InfiniBand나 CUDA 전송을 뺄 수 있다.

PD 재실행 작업(`serving-screen-nvfp4-pd-nd96.yml`)의 PD 단계는 다음 순서로 실행한다.

1. 서버를 띄우기 전에 `nixl_preflight.py`를 node0과 node1에서 실행한다. node1에서는 그 노드에 고정한 Ray 작업이 실행한다. 다음 환경을 차례로 시험하고, 실행마다 120초로 제한한다.
   1. 작업 환경 그대로
   2. `UCX_TLS=all`, `UCX_NET_DEVICES=all`
   3. 2번과 `UCX_MEMTYPE_CACHE=n`
2. 사전 점검은 GPU 1개에서 다음을 확인한다. 로더 캐시의 `libcuda`, `libnvidia-ml`, `libcudart`, NIXL 에이전트를 만든 뒤 UCX CUDA 전송(`libuct_cuda`)의 적재, 첫 등록 전과 후에 할당한 GPU 메모리의 VRAM 등록, 길이 0 서술자(대조군, 실패가 정상)다. 결과는 JSON 한 줄이고, 환경 변수는 `UCX_*`와 `NIXL_*`만 기록한다.
3. 두 노드에서 처음으로 통과한 환경을 두 서버에 똑같이 준다. 서버와 사전 점검이 같은 UCX 값을 쓴다. 어느 환경도 통과하지 못하면 서버를 띄우지 않고 "experimental phase: PD not possible here: NIXL preflight failed (<이유>)"를 기록한다. `node0.<단계>.preflight`와 결과 꾸러미에는 노드마다 작업 환경의 `UCX_*` 값과 고른 환경이 남는다.
4. 사전 점검을 통과했는데도 기동이 "VRAM memory is detected as host"로 실패하면, 두 서버를 `VLLM_LOGGING_LEVEL=DEBUG`로 한 번 다시 띄운다. 두 로그의 "Registering descs" 줄과 길이 0 서술자 수를 `node0.<단계>.failure`에 넣고 단계를 끝낸다. 측정 스위트는 실행하지 않는다.
5. 기동에 성공하면 확인 요청, Part R, dx2 격자를 위와 같이 실행한다.

- `scr-Dref`는 node0에서 D를 띄워 시작 사실과 짧은 출력 동등성만 기록한다. PD 재실행 작업의 출력 동등성 기준이다.
- 재실행 작업에는 Dx1과 Dx2가 없다. 그래서 PD의 비교 쌍(PD 대 D, PD 대 D×2)은 이전 2노드 작업의 Dx1과 Dx2를 쓰고, `select` 출력에 "cross-job"으로 표시한다. 두 작업의 PD 실행은 모두 보고한다(`scr-PD`, `scr-PD@<작업>`).
- 근거: vLLM v0.23.0의 `docs/features/nixl_connector_usage.md`와 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`다. v0.23에서 NixlConnector의 `kv_both` 역할은 폐기 예정이다. 이미지에는 nixl 1.2.0이 들어 있다(10-21 실행의 환경 기록).

### 6.6 선택 규칙 (사전 등록)

- A는 항상 라운드에 들어간다.
- FP8 선택은 A가 아닌 구성 중에서 고른다. 후보는 `scr-C2`, `scr-C4`, `scr-C4m`, `scr-C8`, `scr-C8m`, `scr-C16`, `scr-C16m`, `scr-C1`, `scr-C1rr`, `scr-B0`, `scr-B0dcp`, `scr-A4` 중 자격을 갖춘 구성이다. 이 중 R/isl8192-c32의 출력 tok/s가 가장 높은 구성 1개를 고른다. 기준은 `scr-A`다.
- NVFP4 후보는 `scr-D`, `scr-E`, `scr-Err`, `scr-D8`, `scr-D24`, `scr-D42` 중 자격을 갖춘 구성이다. 같은 지표로 1개를 고른다. D 자신도 후보다. 기준은 `scr-D`다.
- 자격은 다음 다섯 가지를 모두 만족하는 것이다.
  1. 단계가 통과했다. 정상적으로 시작했고, 실제 설정과 NCCL 게이트를 통과했고, 스크리닝을 마쳤다.
  2. 공통 `--max-model-len`을 담았다. 길이를 -1로 바꿔 다시 띄우지 않았다.
  3. 짧은 출력 동등성이 기준과 맞는다. 평균 |Δ log-prob| ≤ 0.05 nats, top-1 일치율 ≥ 0.95, 16번째 토큰 전에 달라진 그리디 프롬프트 ≤ 1개다.
  4. 순위 지점(R/isl8192-c32)이 유효하다. 이 지점의 kv-bound 또는 capacity-bound 표시는 제외 사유가 아니다.
  5. 단일 요청 TPOT(입력 1K와 8K 각각)이 기준의 1.5배(FP8) 또는 1.2배(NVFP4) 이하다.
- 값이 같으면 목록의 앞 구성을 고른다.
- 후보의 출력 동등성과 TPOT 기준은 같은 작업의 기준 단계다. 예를 들어 `scr-C4m`은 FP8 DP 작업의 `scr-A`와 비교한다.
- 같은 단계가 여러 작업의 꾸러미에 있으면(두 FP8 작업의 `scr-A` 등) 작업마다 따로 보고한다. 두 번째부터는 `<단계>@<작업>`으로 표시한다. 실패한 Ray 실행 `scr-C4`와 mp-multinode 실행 `scr-C4m`도 서로 다른 실행으로 함께 보고한다.
- 순위 지점의 kv-bound·capacity-bound 표시는 점수 옆에 보고한다. 그 지점의 최대 KV 사용률과 최대 실행 요청 수도 함께 보고한다.
- 표시로 제외하지 않는 이유: 이 지점은 32 × 9,216 = 294,912 토큰을 담아야 한다. D, E, Err(KV 토큰 약 23.2만~25.4만)와 B0(약 31만, 10-21 측정의 최대 KV 사용률 0.953)는 이 지점에서 항상 kv-bound다. 표시로 제외하면 D, E, Err는 선택될 수 없다.
- D×2: `scr-Dx2`의 W1 λ*(기준 SLO)가 `scr-Dx1`의 1.9배 이상이면 D×2 작업을 실행한다. 격자 밖의 λ*는 한계값으로 쓴다. 한계값으로 판정할 수 없으면 D×2 작업이 전체 격자로 판정한다.
- `bench_stats.py select`는 `ROUND_LAYOUTS=<FP8 선택>,<NVFP4 선택>`, 민감도 실행 이름, D×2 판정, 구성별 제외 이유, 순위 지점의 표시와 최댓값을 낸다. 라운드가 찾을 도착률 키(`SLO_RATES_<구성>`과 `SOAK_RATES`의 키, 7.1절)도 낸다. 이 값을 사전 등록 커밋에 적고 모든 라운드에 같은 값을 준다.
- 3번의 기준은 망가진 구성을 거르기 위해 느슨하게 정했다. 라운드에서는 11절의 A′ 기준을 적용한다.

### 6.7 메모리와 KV 용량 (추정)

아래 값은 추정이다. 스크리닝이 실제 가중치, KV 토큰 수, 맞춘 길이를 기록한다.

- GPU당 가중치 ≈ (라우팅 전문가 ÷ EP + 공유 전문가 + TP로 나누는 비전문가 가중치 ÷ TP + 복제되는 비전문가 가중치) × (가장 무거운 PP 단계의 층 비율). 시퀀스 병렬 MoE(DP>1, TP>1, EP)이면 공유 전문가를 랭크마다 전부 둔다. TP1이면 원래 전부 둔다.
- KV는 토큰·층·GPU당 1,284바이트다(bf16 MLA 1,152 + 인덱서 132). TP 랭크마다 복제된다.
- 가중치 외 메모리(활성값, CUDA graph, 버퍼)는 이전 측정 기준으로 TP·PP 구성(A) 약 3 GiB, NVFP4 TP8(D) 약 6.6 GiB다. DP×EP 구성은 여기에 M = DP × `max_num_batched_tokens` 토큰의 Marlin MoE 작업 공간이 더해진다(6.3절).
- C4, C8, C16, D8, D24, B0dcp의 값은 vLLM v0.23.0 소스로 기동 조건을 점검해 얻은 예측이다. 6.3절의 엔진별 배치 토큰을 전제로 한다. 스크리닝이 이 예측을 확인하거나 반박한다.
- 근거로 쓴 GPU당 가중치: TP16 43.27 GiB, C2 46.39 GiB, TP8×PP2 node0 42.28 GiB, NVFP4 TP8 48.5 GiB.

| 구성 | GPU당 가중치 (GiB, 추정) | 엔진당 KV 토큰 (추정) | 131,072 수용 (추정) |
|---|---|---|---|
| A4 | 35~47 | 약 110만 | 예 |
| C4 | 48 | 약 21.6만 | 예 |
| C8 | 51.3 | 약 17.2만 | 예 |
| C16 | 57.8 | 약 8.2만 (엔진 16개, 합계 약 130만) | 아니오. `-1`로 다시 띄울 것으로 예상한다 |
| B0dcp | B0와 같다 (43.27) | - | 기동 실패를 예상한다 |
| E | 48.5 | 약 23만 (0.92에서 253,120 측정) | 예 |
| D8 | 62.1 | 약 4.6만 (엔진 8개, 합계 약 37만) | 아니오. `-1`로 다시 띄울 것으로 예상한다 |
| D24 | 52.3 | 약 15.5만 | 예 |
| D42 | 47~50 | 약 44만 | 예 |
| PD | 서버마다 48.5 | 서버마다 약 23만 | 예 |

- C16과 D8은 라운드 후보가 될 수 없을 것으로 예상한다. 그래도 엔진을 모두 합친 KV 용량이 크므로 두 구성의 수치를 보고한다.
- C4m, C8m, C16m의 추정값은 C4, C8, C16과 같다. 실행 방식은 메모리 사용을 바꾸지 않는다. C16m은 두 노드 모두 `-1`로 다시 띄울 것으로 예상한다. D8은 스크리닝에서 엔진당 KV 32,896토큰을 기록했다(측정, 추정 약 4.6만). 따라서 C16m의 실제 값은 추정보다 작을 수 있다.
- B0dcp의 기동 실패를 예상하는 근거: vLLM의 `check_attention_cp_compatibility`는 어텐션 백엔드가 디코드 softmax LSE를 돌려주는지 검사한다(`vllm/v1/worker/cp_utils.py:30-37`, 호출 위치 `vllm/v1/worker/gpu_model_runner.py:6804`). 포팅한 `TritonMLASparseImpl`은 `can_return_lse_for_decode`를 설정하지 않으므로 기본값 False를 쓴다(`vllm/v1/attention/backend.py:707, 761-763`).

### 6.8 Ray DP 기동 실패와 mp-multinode 실행

**측정.** 스크리닝(main 5dc653b)에서 C4, C8, C16은 기동 약 70초 뒤 실패했다. 호출 경로는 `RayWorkerProc.initialize_worker` → `WorkerProc.__init__` → `PyNcclCommunicator` → `ncclCommInitRank`였고, NCCL 오류는 "invalid usage"였다. 세 구성은 Ray DP 백엔드로 노드마다 엔진을 2개 이상 둔다. 같은 스크리닝에서 노드마다 엔진이 1개인 C2(Ray DP)와, 한 노드에 엔진을 여러 개 둔 D8·D24(mp DP)는 통과했다.

**원인(추론).**

- 소거법으로 확인: NCCL 2.28.9에서 `ncclCommInitRank`가 "invalid usage"를 돌려주는 경로 가운데 이 구성에 남는 것은 중복 GPU 검사("Duplicate GPU detected", `src/init.cc:903-907`)뿐이다.
- 추정: 같은 노드의 두 엔진을 처음 묶는 통신기는 DP 그룹이다. 두 엔진의 작업자가 같은 GPU에 묶였다. 원인 위치는 RayExecutorV2가 이미 실행 중인 작업자의 `CUDA_VISIBLE_DEVICES`를 다시 쓰는 부분(`vllm/v1/executor/ray_executor_v2.py:368-397`)으로 추정한다.
- 미확인: 충돌을 일으킨 정확한 문장은 찾지 못했다. NCCL 로그의 "Duplicate GPU detected" 줄과 랭크별 `nvmlDev` 값으로 확인할 수 있다.

**대응.** Ray 구성(C4, C8, C16)과 실패 기록은 그대로 둔다. 같은 위상을 vLLM 문서의 다중 노드 multiprocessing DP 실행(`docs/serving/data_parallel_deployment.md`)으로 다시 측정한다. 이 문서는 이 실행 방식을 mp-multinode라고 부른다. 단계는 `scr-C4m`, `scr-C8m`, `scr-C16m`이고, 보고서의 구성 이름은 C4, C8, C16이다.

| 단계 | TP | DP | 노드당 엔진 (L) | node1의 시작 랭크 | 엔진별 배치 토큰 |
|---|---|---|---|---|---|
| `scr-C4m` | 4 | 4 | 2 | 2 | 4,096 |
| `scr-C8m` | 2 | 8 | 4 | 4 | 2,048 |
| `scr-C16m` | 1 | 16 | 8 | 8 | 1,024 |

한 단계의 실행 순서는 다음과 같다.

1. node0에서 `vllm serve`를 띄운다. 공통 플래그에 `--data-parallel-backend mp --distributed-executor-backend mp --data-parallel-size DP --data-parallel-size-local L --data-parallel-address <node0 주소> --data-parallel-rpc-port 13345`를 더한다. API 서버 플래그(`--served-model-name`, `--host`, `--port`, `--api-server-count 1`, `--tool-call-parser`, `--enable-auto-tool-choice`, `--no-enable-log-requests`)는 node0에만 준다. `VLLM_HOST_IP`는 node0의 주소다.
2. 몇 초 안에 node1에서 Ray 분리 actor가 `vllm serve <모델> <같은 엔진 플래그> --headless --data-parallel-start-rank L`을 띄운다. `VLLM_HOST_IP`는 node1의 주소다.
3. node0의 `/health`만 기다린다. node0의 API 서버는 두 노드의 DP 엔진이 모두 연결된 뒤에 응답한다. 기다리는 동안 node1의 프로세스를 1분마다 확인한다. node0이나 node1의 서버가 끝나면 바로 실패로 기록한다. 제한 시간은 다른 단계와 같다.
4. 단계가 끝나거나 실패하면 node1의 프로세스 그룹을 멈추고 node0을 멈춘다. 정리 상태 확인은 두 노드를 모두 본다.

- 엔진 플래그는 두 노드에서 같다. node0은 DP 엔진의 병렬 설정 해시를 비교한다. `--reasoning-parser`는 vLLM 0.23에서 엔진 인자이므로(`vllm/engine/arg_utils.py:610, 2171-2177, 2441-2445`) 두 노드에 모두 준다.
- node0에는 `--data-parallel-start-rank`를 주지 않는다. 이 옵션이 있으면 vLLM이 hybrid 부하 분산으로 바뀐다. node1에는 `--api-server-count`를 주지 않는다. `--headless`와 함께 주면 vLLM이 오류를 낸다.
- 길이 대체: KV 용량 오류는 어느 노드의 로그에든 나올 수 있다. node0의 로그와 node1 로그의 끝부분을 모두 확인한다. 오류가 있으면 두 노드를 모두 멈추고 두 노드를 `--max-model-len -1`로 다시 띄운다. C16m은 이 경로를 탈 것으로 예상한다(6.7절).
- 게이트: 실제 설정 검사와 NCCL 검사는 다른 FP8 2노드 단계와 같다. node0의 로그에는 node0의 엔진만 나온다. node1 엔진의 KV 토큰과 DP 랭크 줄은 node1의 로그에서 Ray로 읽어 결과 꾸러미(`peer`)에 넣는다. 이 기록이 없어도 게이트는 실패하지 않는다.
- 작업: `serving-screen-fp8-dp-nd96.yml`(2노드)이 `scr-A`를 먼저 실행한다. 이 A가 같은 작업의 출력 동등성과 TPOT 기준이다.
- 선택: `scr-C4m`, `scr-C8m`, `scr-C16m`은 같은 규칙의 FP8 후보다(6.6절). 선택되면 `ROUND_LAYOUTS`에 `C4m` 같은 이름이 들어가고, 라운드도 같은 실행 방식을 쓴다. 1라운드 민감도 실행은 `C4ms`, `C8ms`, `C16ms`다.
- 위험(미확인): vLLM 이슈 #43547은 MoE와 DP를 함께 쓸 때 약 1시간 뒤 멈추는 현상을 보고한다. 이 현상은 장시간 부하에서 C2와 mp-multinode 구성 모두에 영향을 줄 수 있다.

## 7. 라운드, 순서, 시드

### 7.1 라운드

라운드 하나는 별도의 AML 작업 하나다. 3라운드로 시작한다. 3라운드는 1·2라운드와 다른 노드 쌍에서 실행한다. 결과 꾸러미에 노드 호스트 이름의 해시가 들어 있어 다른 쌍인지 확인할 수 있다.

`ROUND_LAYOUTS`가 A와 비교할 후보다. 기본값은 `C2,D`다. 홀수 라운드는 적힌 순서로, 짝수 라운드는 역순으로 실행한다. 1라운드에는 후보의 민감도 실행(`C2s` 등)과 용량 단계(`cap-A`, `cap-<후보>`)를 붙인다. 기본값의 순서는 다음과 같다.

| 라운드 | 단계 순서 |
|---|---|
| 1 | A → C2 → D → C2-s → A′ → cap-A → cap-C2 → cap-D |
| 2 | A → D → C2 → A′ |
| 3 | A → C2 → D → A′ (장시간 부하 포함) |
| 4 (연장할 때) | A → D → C2 → A′ |
| 5 (연장할 때) | A → C2 → D → A′ |

`PHASE_PLAN=slo-round`를 주면 `ROUND`와 `ROUND_LAYOUTS` 값에 따라 위 규칙대로 펼쳐진다. 예를 들어 `ROUND_LAYOUTS=C4,E`이면 1라운드는 A → C4 → E → C4s → A′ → cap-A → cap-C4 → cap-E이고, 2라운드는 A → E → C4 → A′다. `protocol.sh`는 펼친 계획과 후보를 `protocol-plan.json`에 남기고, `bench_stats.py analyze`는 그 기록으로 비교 쌍을 정한다.

라운드는 구성의 도착률 격자를 `SLO_RATES_<이름>`에서 찾고, 장시간 부하의 도착률을 `SOAK_RATES`의 `<이름>=<req/s>`에서 찾는다. 이름은 후보 자신의 이름이다. 예를 들어 C1rr, Err, B0dcp는 C1, E, B0의 값을 쓰지 않는다. A′와 민감도 실행(`C2s` 등)만 기준 구성(A, C2 등)의 이름을 쓴다. `bench_stats.py select`가 라운드가 찾을 키를 출력한다.

### 7.2 시드

- 측정 지점의 시드는 `1000000×ROUND + 10000×WORKLOAD + 100×POINT + REP`다. REP는 0이다.
- WORKLOAD 번호: Part R 1, S1 2, W1 3, W2 4, W1-real 5, 장시간 부하 6, W3 7.
- POINT는 워크로드 전체 목록(도착률 격자 또는 Part R 목록)에서의 순번이다. 일부 지점만 실행하는 A′와 C2-s도 같은 시드를 쓴다.
- 같은 라운드의 모든 구성이 같은 프롬프트를 받는다. A′는 A와 같은 시드를 쓴다.
- 워밍업 시드는 `9000000 + 10000×ROUND + 100×POINT`다.
- 도착률 지점의 실행 순서는 `1000000×ROUND + 10000×WORKLOAD + 99`로 섞는다. 같은 라운드에서는 모든 구성이 같은 순서다.
- 게이트에 걸린 지점은 같은 시드로 1번 다시 실행한다. 두 시도를 모두 보고한다.
- 진단 작업과 스크리닝 작업은 `ROUND=0`이라 라운드 시드와 겹치지 않는다.

## 8. 단계 하나의 실행 순서

`entry.sh`가 `aml/src/bench/protocol.sh`를 읽어 다음 순서로 실행한다. `PHASE_PLAN`에 측정 단계가 없으면 이 파일은 아무것도 하지 않는다.

1. 작업 시작 시 모든 노드에서 계측을 켠다. nvidia-smi가 1초마다 클럭, 전력, 온도, 사용률, 스로틀 원인을 기록한다. InfiniBand 포트 카운터는 5초마다 기록한다. NCCL은 프로세스마다 로그 파일을 남긴다(`NCCL_DEBUG_FILE`).
2. 단계가 쓰는 노드의 상태를 확인한다. GPU 계산 프로세스가 없어야 하고, GPU 메모리 사용량이 2 GiB 미만이어야 한다. 남은 Ray placement group이 없어야 하고, 8000번 포트가 비어 있어야 한다. 180초 안에 정리되지 않으면 회수를 한 번 실행하고 상태를 다시 확인한다. 그래도 정리되지 않으면 단계는 실패한다.
   - 회수는 단계가 쓰는 모든 노드에서 실행한다. node0이 아닌 노드에서는 그 노드에 고정한 Ray 작업(`num_cpus=0`)이 실행한다.
   - 회수 대상 후보는 컨테이너 안에서 보이는 프로세스 중 GPU 장치(`/dev/nvidia<N>`, `/dev/nvidia-uvm`)를 열었거나 8000번 포트에서 연결을 기다리는 프로세스다.
   - 후보 중 다음 프로세스만 SIGKILL로 끝낸다. 명령줄에 vllm이 들어 있는 프로세스(대소문자 무시), 명령줄이 `ray::`로 시작하는 Ray 작업자·actor, 조상 중에 vllm 프로세스가 있는 multiprocessing 자식 프로세스다.
   - nvidia-smi, Ray 시스템 프로세스(raylet, gcs_server, dashboard 등), 회수 프로세스 자신과 그 조상은 끝내지 않는다. 그 밖의 후보도 끝내지 않고 이유(예: "unknown GPU process")와 함께 기록한다. 그 프로세스가 GPU를 쓰고 있으면 다시 확인할 때 단계가 실패한다.
   - `nvidia-smi --query-compute-apps=pid`의 PID는 진단용으로 기록만 한다. 컨테이너 안에서는 이 값이 호스트의 PID일 수 있어, 같은 번호의 다른 컨테이너 프로세스를 가리킬 수 있다.
   - 상태가 REMOVED가 아닌 placement group은 지운다.
   - 회수는 300초로 제한하고 단계마다 한 번만 실행한다. 결과는 단계 결과의 비고와 결과 꾸러미(`reclaim`)에 남는다.
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
   6. 게이트를 적용한다(9절). 무효이면 같은 시드로 1번 다시 실행한다.
9. 서버의 프로세스 그룹 전체에 SIGINT를 보내고, 90초 안에 끝나지 않으면 SIGKILL을 보낸다. 그 뒤 2번의 정리 상태를 다시 기록한다.
10. node0이 지금까지의 결과 꾸러미를 만들어 작업 기록으로 보낸다(14절).

## 9. 유효성 게이트

다음 중 하나라도 해당하면 그 지점은 **무효**다.

- 시작 전에 서버가 유휴 상태가 아니었다.
- 캐시 초기화가 실패했다.
- `vllm bench serve`가 0이 아닌 코드로 끝났다.
- 폐쇄형 지점에서 실패한 요청이 1개 이상이다.
- 실패 요청이 없는데 서버가 센 프롬프트 토큰 수가 클라이언트 값과 0.1% 넘게 다르다. PD에서는 프리필 서버의 값만 비교한다(6.5절).
- 조회한 프롬프트 토큰 중 1% 넘게 프리픽스 캐시에 적중했다.
- 지점 전후의 `/metrics`가 없다.
- 요청별 데이터(`--save-detailed` 결과)가 없다.
- 지점 실행 중 어느 노드에서든 GPU 하드웨어 감속, 열 감속, 전력 브레이크(스로틀 원인 마스크 0xE8)가 나타났다.

다음은 지점을 **표시**만 하고 무효로 하지 않는다.

- 선점이 1번 이상 있었다: capacity-bound.
- 어느 엔진이든 KV 사용률이 0.95 이상이었거나, 폐쇄형 지점의 최대 실행 요청 수가 동시성보다 작았다: kv-bound. 이 지점은 라운드의 속도 판정에서 뺀다. 스크리닝의 선택 규칙에서는 표시만 보고하고 구성을 제외하지 않는다(6.6절).
- 개방형 지점의 실패 요청: SLO 미달로 센다.
- 소프트웨어 전력 상한(0x4)에 걸린 표본 비율: 기록만 한다.

라운드는 A, 후보 구성들, A′의 모든 측정 스위트가 결과를 냈고, 중단되지 않았고, 계획한 모든 지점에 유효한 시도가 있을 때만 쓴다. 3라운드에는 장시간 부하도 필요하다. 결과 꾸러미에서 A, 후보 구성, A′의 부분이 하나라도 빠졌거나 위 조건 중 하나라도 빠지면 그 라운드 전체를 쓰지 않는다. 민감도 실행이나 용량 단계의 부분만 빠졌으면 그 단계만 쓰지 않는다. 민감도 실행(C2-s 등)과 용량 단계는 라운드의 유효성을 정하지 않는다.

## 10. 통계와 판정 규칙

### 10.1 지표

- 반복의 단위는 서버 1회 기동이다.
- λ*: 기준을 지킨 가장 높은 격자 도착률과 그다음 도착률 사이를 선형 보간한다. 가장 높은 도착률도 기준을 지키면 "격자가 너무 낮다", 가장 낮은 도착률도 기준을 못 지키면 "격자가 너무 높다"로 기록하고 λ*를 내지 않는다. 이때 격자의 끝 값을 한계값(λ*가 그보다 크다 또는 작다)으로 함께 기록한다. 전체 격자를 실행한 단계(A, 후보 구성, D×2 작업의 두 단계, 스크리닝의 격자 단계)만 λ*를 계산한다. A′와 민감도 실행은 W1 지점 2개만 실행하므로 λ*가 없다.
- 비용: 노드시간당 가격 × 노드 수 ÷ (λ*에서의 출력 tok/s × 3,600) × 10⁶. 라운드마다 계산한다. 가동률 50%를 가정하면 비용은 2배다.
- 가격: 영국 남부 Azure 소매가로 종량제 40.96달러, Spot 12.89달러(노드시간당)다. 저우선순위 노드는 2026-03-31부터 Spot 요금으로 과금된다. 발표일에 가격을 다시 조회한다.
- 주요 지표: λ*(W1, W2, 기준 SLO), W1 λ*에서의 비용(Spot, 종량제), Part R 단일 요청 TPOT(1K·8K·32K), 단일 요청 TTFT(8K·32K). 보조 지표 목록은 `slo_protocol.json`의 `metrics`에 있다.
- P99는 정상 상태 구간의 요청이 1,000개 이상일 때만 보고한다. 기본 격자의 W1 1.5 req/s는 구간 안 요청이 약 450개라 P99를 보고하지 않는다.

### 10.2 신뢰구간

- 실행 안: 30초 분량의 연속 요청을 한 블록으로 하는 이동 블록 부트스트랩을 2,000번 반복한다. 지점의 요청 수가 충분했는지 확인하는 용도다.
- 라운드 사이: 라운드 k의 로그비 r_k = ln(X_k / 기준_k)를 쓴다. A′도 그 지표를 쟀으면 기준_k = √(A_k × A′_k)이고, 아니면 기준_k = A_k다. 비율 exp(평균 r)과 95% 신뢰구간 exp(평균 ± t × 표준편차 / √n)을 보고한다. t는 n=3에서 4.303, n=4에서 3.182, n=5에서 2.776이다. 라운드별 최솟값과 최댓값도 보고한다.
- 3라운드는 서로 다른 재표집이 10개뿐이라 라운드 사이에는 부트스트랩을 쓰지 않는다.

### 10.3 판정

- δ_min = max(ln 1.05, 2 × 최대 |ln(A′_k / A_k)|)다. A′가 재지 않은 지표(λ*, W2, 비용)는 A′가 잰 지표들의 가장 큰 δ_min을 쓴다.
- **나음:** 다음 세 조건을 모두 만족한다.
  1. 모든 라운드에서 X가 앞선다.
  2. 95% 신뢰구간이 1을 포함하지 않는다.
  3. |평균 r| ≥ δ_min이다.
- **나쁨:** 같은 세 조건을 반대 방향으로 만족한다.
- **동등:** 95% 신뢰구간 전체가 [0.95, 1.05] 안에 있다.
- **판정 불가:** 나머지. 신뢰구간을 함께 쓴다.
- kv-bound 또는 capacity-bound 지점의 지표는 판정하지 않는다.

### 10.4 라운드 연장

유효한 라운드가 3개 모이면 다음을 검사한다. 하나라도 해당하면 라운드를 추가한다. 최대 5라운드다.

- A′가 잰 주요 지표에서 라운드 간 ln(A′/A)의 표준편차가 2%를 넘는다.
- 주요 지표의 95% 신뢰구간 반폭이 5%를 넘는다.

## 11. 출력 동등성

| 검사 | 입력 | 요청 |
|---|---|---|
| P1 | 문서 16개(한국어 8개, 영어 8개)를 각각 2,048토큰으로 자른다 | `prompt_logprobs: 20`, `max_tokens: 1`, temperature 0 |
| P1 장문 | 문서 16개를 정한 순서와 역순으로 이어 붙여 32,768토큰으로 자른 문서 2개 | P1과 같다. 단계의 마지막에 실행한다 |
| P2 | 문서마다 0~511번 토큰과 1,024~1,535번 토큰, 모두 32개 | 그리디 256토큰, 한 번에 하나씩 |

- 비교 기준은 같은 라운드의 A다. 진단 작업에서는 `diag-A`, 스크리닝에서는 작업의 첫 단계다(6.1절). 스크리닝은 문서 4개의 P1과 그리디 128토큰의 짧은 P2만 실행한다(6.4절).
- FP8 후보(C2, C2-s 등)의 허용 기준은 다음 세 가지를 모두 만족하는 것이다.
  1. 평균 |Δ 선택 토큰 log-prob| ≤ 2 × (A′ 대 A) + 0.005 nats
  2. top-1 일치율 ≥ (A′ 대 A) − 0.01
  3. 16번째 토큰 전에 출력이 달라진 프롬프트가 2개 이하
- 3번을 만족하지 못하면 구성 설정이 잘못된 것으로 본다.
- NVFP4(D)는 차이만 보고한다. 품질 동등성은 별도 정확도 평가(KoBALT 200문항, CLIcK 300문항, 대응 McNemar 검정, 비열등 허용 폭 −3점)로 판정한다. 이 평가는 이 하네스에 포함하지 않았다.
- `parity_corpus.json`에는 저작권 보호를 받지 않는 글만 넣었다. 한국어는 법령 5개(저작권법 제7조로 보호 대상이 아니다)와 1962년 이전에 사망한 작가의 소설 3개다. 영어는 미국 건국 문서와, 1930년까지 사망한 작가가 1892년까지 출판한 공유 저작물이다. 문서마다 출처와 근거를 기록했다.

## 12. D×2 검증

1. `dx1-D`: node0에서 D를 혼자 띄우고 W1·W2 격자를 잰다.
2. `dx2-D`: node0과 node1에서 D를 하나씩 띄운다. node1의 서버는 Ray 분리 actor가 별도 세션으로 띄우고 멈춘다.
3. node0의 `lb_proxy.py`(127.0.0.1:8100)가 요청마다 처리 중 요청이 가장 적은 서버를 고른다. 같으면 차례대로 고른다. 스트리밍 응답은 받는 대로 전달한다.
4. 프록시는 `/health`, `/lb/stats`, `/metrics`(각 표본에 `replica` 라벨 추가), `/reset_prefix_cache`(두 서버에 모두 전달)를 직접 처리한다.
5. 통과 기준은 W1과 W2 모두 λ*(dx2) ≥ 1.9 × λ*(dx1)이다(기준 SLO).

NVFP4 품질 검증을 통과한 뒤 실행한다. 스크리닝의 `scr-Dx2`가 이 기준을 확실히 못 넘으면(6.6절) 실행하지 않는다. NVFP4 후보가 D가 아니면 `PHASE_PLAN="dx1-<구성> dx2-<구성>"`으로 같은 검증을 할 수 있다.

## 13. 진단 실험

`serving-diag-nd96.yml`이 라운드 전에 다음을 확인한다. 단계마다 1회 실행이므로 결과는 경향이며 통계적으로 확정된 차이가 아니다. FP8 스크리닝 작업이 같은 항목(`scr-A-auto`, `scr-C1rr`, `scr-B0dcp`, 커널 시험)을 포함하므로 이 작업은 선택 사항이다. 다만 32K `prompt_logprobs` 확인은 이 작업에만 있다.

- **커널 시험:** DSA 포트의 커널 시험과 인덱서 점수 저장 회귀 시험(`aml/src/bench/test_dsa_port_regression.py`)을 GPU에서 실행한다.
- **A1:** `diag-A`(131,072)와 `diag-A-maxlen`(자동 맞춤, 약 262,144)의 동시 32 ITL, 출력 tok/s, 단일 요청 TPOT을 비교한다.
- **A8:** `diag-C1`(linear 전문가 배치)과 `diag-C1rr`(`--expert-placement-strategy round_robin`)을 비교한다.
- **DCP:** `diag-B0`(TP16)와 `diag-B0dcp`(`--decode-context-parallel-size 2`)를 비교한다. 서버가 뜨는지, KV 토큰 수가 뚜렷이 늘었는지, DCP 그룹 로그가 있는지, `diag-A` 대비 출력이 같은지, 짧은 부하가 끝나는지를 본다. 다섯 가지를 모두 만족해야 DCP를 지원한다고 본다. 기동 실패를 예상한다(6.7절). 서버가 뜨지 않아도 실패를 기록하고 작업은 계속된다.
- **32K prompt_logprobs:** `diag-A`의 P1 장문이 메모리 부족 없이 끝나는지 확인한다. 실패하면 1라운드 전에 장문 P1을 빼고 그 사실을 사전 등록에 적는다.

## 14. 결과 전달

측정 작업은 자기 결과를 작업 기록으로 직접 보낸다. 이것이 기본 경로다.

- 이유: 작업 공간 저장소는 공용 네트워크 접근이 꺼져 있어, 작업 공간 밖에서는 작업 산출물을 읽을 수 없다. 또 작업 공간에는 CPU 클러스터가 없고, 제출하는 사용자에게 Blob 데이터 읽기 역할이 없을 수 있다. 그래서 CPU 수집 작업(`collect-slo-outputs.yml`)은 대체 경로로만 남긴다.
- 보내는 방법:
  1. 단계가 끝날 때마다 node0의 `protocol.sh`가 `bench_stats.py package --report`로 지금까지의 결과 꾸러미를 만든다. 2노드 작업은 먼저 node1의 GPU 계측 파일, InfiniBand 카운터, 두 번째 서버 로그의 끝부분(최대 200줄, 16 KB)을 Ray로 가져온다(`node_tools.py pull`).
  2. `report.py`가 꾸러미를 MLflow 태그로 보낸다. 태그 하나는 4,000자 조각이다. `slo.package`(색인)에는 측정 지점을 뺀 전부와 부분 목록이 들어간다. `slo.package.<단계>`(부분)에는 그 단계의 측정 지점이 들어간다. 부분을 먼저 보내고 색인을 마지막에 보낸다. 그래서 색인은 아직 보내지 않은 부분을 가리키지 않는다.
  3. 키 하나는 150조각(base64 약 600 KB)을 넘지 않는다. 넘으면 그 단계의 지점을 반으로 나눠 `<단계>-p1`, `<단계>-p2`로 보낸다. 색인이 150조각을 넘으면 색인에서 실제 설정의 로그 줄을 뺀다.
  4. `report.py`는 항상 0으로 끝난다. 그래서 `bench_stats.py`가 `report.py`의 출력을 읽어 전송 여부를 판단한다. "mlflow unavailable" 경고, 네 번째 `set_tags` 시도의 실패, 그 키에 대한 예외 경고 중 하나라도 있거나 `report.py`가 300초 안에 끝나지 않으면 그 키는 보내지 않은 것으로 기록한다.
  5. 단계가 끝난 뒤에는 바뀐 키와 아직 보내지 못한 키만 보낸다. 작업이 끝날 때(EXIT 트랩) 모든 키를 다시 보낸다.
  6. 이 보고는 최선 노력이다. Ray 수집은 180초, 꾸러미 생성과 전송은 420초로 제한한다. 실패하거나 시간이 지나도 단계와 작업의 종료 상태를 바꾸지 않는다.
- 읽는 방법: `aml/fetch_results.py show <job-name>`의 출력을 `bench_stats.py analyze` 또는 `select`에 그대로 준다. 분석 스크립트가 부분을 합친다. 빠진 부분은 `missing_parts`에, 그 부분에 든 단계는 `missing_launches`에 남는다. 빠진 부분은 그 단계만 무효로 만든다. 그 단계가 라운드에 필요한 단계이면 라운드가 무효가 되고(9절), 스크리닝 단계이면 선택 자격을 잃는다.
- 꾸러미 생성 시간(측정): FP8 스크리닝 작업이 끝날 때 크기의 합성 출력 폴더를 만들어 쟀다. 단계 11개, GPU 16개의 1초 간격 계측 7.5시간, 요청별 파일 전부를 넣었다(파일 252개, 112 MB). `bench_stats.py package`는 5.7초, 부분 나누기와 인코딩은 0.1초 미만이 걸렸다(WSL, 1회). 그래서 단계마다 꾸러미를 처음부터 다시 만들고 캐시는 쓰지 않는다. node1에서 가져오는 파일은 약 24 MB다. Ray 수집 시간은 측정하지 않았다.
- 꾸러미 크기(추정): 실제 요청 수와 같은 합성 꾸러미로 계산했다. 다시 실행한 지점 15%를 포함했다.

| 작업 | 측정 지점 | 요청 | 꾸러미 전체 (base64) | 보고 키 | 키당 최대 조각 |
|---|---|---|---|---|---|
| FP8 스크리닝 | 87 | 4,756 | 약 180 KB (46조각) | 12 | 8 |
| NVFP4 1노드 스크리닝 | 51 | 2,784 | 약 100 KB (26조각) | 7 | 5 |
| NVFP4 2노드 스크리닝 | 35 | 7,992 | 약 220 KB (56조각) | 4 | 25 |

## 15. 작업 렌더링과 제출

> **경고:** 아래 제출 명령은 GPU 노드 비용을 발생시킨다. 16절의 예산 안에서만 실행한다.

전제 조건은 다음과 같다.

- `aml/src` 변경이 커밋되어 원격 저장소에 push되어 있어야 한다. `render_job.py`는 커밋되지 않았거나 원격에 없는 커밋을 거부한다.
- Azure CLI의 `ml` 확장이 설치되어 있고 로그인되어 있어야 한다.
- 작업 공간에 저우선순위 A100 클러스터 `a100-nd96-lp`(ND96amsr A100 v4)가 있어야 한다. 모든 측정 템플릿이 이 클러스터를 쓴다.
- 아래의 `<workspace>`, `<resource-group>`, `<job-name>`, `<mlflow-tracking-uri>`는 자리 표시자다. 실제 값을 이 문서나 저장소에 적지 않는다.

순서는 다음과 같다. 명령은 bash 기준이다. 실행해서 확인한 명령이 아니다.

1. 템플릿을 렌더링한다. 결과는 `aml/jobs/.rendered/`에 생긴다. 측정 템플릿은 `aml/src`와 `aml/src/bench`의 파일을 고정 커밋에서 내려받아 파일마다 해시를 확인한다.

   ```bash
   python aml/render_job.py aml/jobs/serving-screen-fp8-nd96.yml aml/jobs/serving-screen-fp8-dp-nd96.yml \
     aml/jobs/serving-screen-nvfp4-nd96.yml aml/jobs/serving-screen-nvfp4-2node-nd96.yml \
     aml/jobs/serving-screen-nvfp4-pd-nd96.yml aml/jobs/slo-round-nd96.yml aml/jobs/slo-nvfp4-2replica-nd96.yml
   ```

2. 스크리닝 작업 5개를 제출한다. FP8 DP 작업과 PD 재실행 작업은 첫 스크리닝 결과로 추가한 작업이다. 저우선순위 할당량은 ND96 3대다. 2노드 작업 하나와 1노드 작업은 함께 실행할 수 있지만, 2노드 작업 둘은 차례로 실행해야 한다.

   ```bash
   az ml job create --file aml/jobs/.rendered/serving-screen-fp8-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   az ml job create --file aml/jobs/.rendered/serving-screen-fp8-dp-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   az ml job create --file aml/jobs/.rendered/serving-screen-nvfp4-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   az ml job create --file aml/jobs/.rendered/serving-screen-nvfp4-2node-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   az ml job create --file aml/jobs/.rendered/serving-screen-nvfp4-pd-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
   ```

3. 결과를 내려받는다. 결과는 단계마다 갱신되므로 작업이 끝나기 전에도 받을 수 있다. 나머지 작업도 같은 방법으로 `screen-fp8-dp.json`, `screen-nvfp4.json`, `screen-nvfp4-2node.json`, `screen-nvfp4-pd.json`에 받는다.

   ```bash
   AXK2_MLFLOW_URI=<mlflow-tracking-uri> python aml/fetch_results.py show <job-name> > screen-fp8.json
   ```

4. 선택 규칙을 적용한다.

   ```bash
   python aml/src/bench/bench_stats.py select screen-fp8.json screen-fp8-dp.json screen-nvfp4.json screen-nvfp4-2node.json \
     screen-nvfp4-pd.json --out screen-selection.json --md screen-selection.md
   ```

5. 출력의 `round_layouts`, 공통 `--max-model-len`, 장문 P1 포함 여부를 확정한다. `screen-selection.json`의 `rates`와 명령의 화면 출력에 라운드가 찾을 `SLO_RATES_<구성>`과 `SOAK_RATES`의 키가 있다. 바뀐 값이 있으면 `slo_protocol.json`을 고친다. 그다음 사전 등록 커밋을 만들고 1번부터 다시 렌더링한다.
6. 1라운드를 제출한다. `ROUND_LAYOUTS`에는 4번의 `round_layouts` 값을 준다.

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=1 --set environment_variables.ROUND_LAYOUTS=<FP8 choice>,<NVFP4 choice>
   ```

7. 작업이 끝나면 결과를 내려받아 분석한다. 분석 결과에는 다음 라운드의 격자 제안(`next_grid`)과 라운드 연장 판단(`extension`)이 들어 있다.

   ```bash
   AXK2_MLFLOW_URI=<mlflow-tracking-uri> python aml/fetch_results.py show <job-name> > round1.json
   python aml/src/bench/bench_stats.py analyze round1.json --out slo-evidence.json --md slo-summary.md
   ```

8. 2라운드를 제출한다. `ROUND_LAYOUTS`는 1라운드와 같다. 격자를 다시 맞췄으면 `SLO_RATES`(모든 구성 공통) 또는 `SLO_RATES_<구성>`(예: `SLO_RATES_C2`, `SLO_RATES_C1rr`)을 함께 준다. 구성 이름은 후보 자신의 이름이다. A′는 A의 격자를, 민감도 실행은 기준 구성의 격자를 쓴다(7.1절).

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=2 --set environment_variables.ROUND_LAYOUTS=<FP8 choice>,<NVFP4 choice> \
     --set "environment_variables.SLO_RATES=W1=<r1>,<r2>,<r3>,<r4>;W2=<r1>,<r2>,<r3>,<r4>"
   ```

9. 3라운드는 1·2라운드와 다른 노드 쌍에서 실행한다. 장시간 부하를 위해 구성별 0.8 × λ*(W1, 기준 SLO)를 `SOAK_RATES`로 준다. 키는 4번 출력의 `rates.soak_rates`와 같다. 값이 없으면 장시간 부하가 실행되지 않아 3라운드가 무효가 된다.

   ```bash
   az ml job create --file aml/jobs/.rendered/slo-round-nd96.yml --workspace-name <workspace> --resource-group <resource-group> \
     --set environment_variables.ROUND=3 --set environment_variables.ROUND_LAYOUTS=<FP8 choice>,<NVFP4 choice> \
     --set "environment_variables.SOAK_RATES=A=<req/s>,<FP8 choice>=<req/s>,<NVFP4 choice>=<req/s>"
   ```

10. 3라운드까지 7번으로 분석한다. 모든 라운드의 결과를 함께 준다(`bench_stats.py analyze round1.json round2.json round3.json ...`). `extension.recommended`가 참이면 `ROUND=4`, 필요하면 `ROUND=5`를 제출한다.
11. NVFP4 품질 검증을 통과했고 4번 출력의 `dx2.run_dx2_job`이 참이면 D×2 작업을 제출하고 7번으로 분석한다.

    ```bash
    az ml job create --file aml/jobs/.rendered/slo-nvfp4-2replica-nd96.yml --workspace-name <workspace> --resource-group <resource-group>
    ```

작업 기록에 결과가 없으면(14절의 보고가 실패하면) 대체 경로를 쓴다. 이 경로에는 CPU 클러스터와 작업 공간 산출물 저장소의 읽기 권한이 필요하다. `collect-slo-outputs.yml`에서 `__RUN__`을 작업 이름으로, `cpu-lp`를 있는 CPU 클러스터 이름으로 바꾼 사본을 렌더링해 제출한다. 사본에는 작업 이름이 들어 있으므로 커밋하지 않는다.

## 16. 비용 (추정, 미실행)

가격은 Azure 소매가(영국 남부)로 종량제 40.96달러, Spot 12.89달러(노드시간당)다. 저우선순위 노드는 2026-03-31부터 Spot 요금으로 과금된다. 종량제(전용) 노드는 쓸 수 없다. 전용 A100 VM 할당량이 0이다. 발표일에 가격을 다시 조회한다.

| 항목 | 노드시간 | Spot |
|---|---|---|
| 스크리닝 (계획 추정) | 약 15 | 약 195달러 |
| 라운드 (계획 추정) | 약 55 | 약 705달러 |
| **합계 (계획 추정)** | **약 70** | **약 900달러** |

- 템플릿 설명의 예상 시간으로 다시 계산하면 스크리닝은 약 37노드시간(Spot 약 475달러)이다. FP8 작업이 약 14, FP8 DP 작업이 약 6.4, NVFP4 1노드 작업이 약 4.5, NVFP4 2노드 작업이 약 8, PD 재실행 작업이 약 4노드시간이다. FP8 DP 작업(Spot 약 83달러)은 Ray DP 실행이 실패해 추가했고, PD 재실행 작업(Spot 약 52달러)은 PD 기동이 실패해 추가했다. 계획 추정보다 큰 주된 이유는 구성마다 Part R 전체(출력 1,024토큰, 약 20분)를 실행하는 데 있다. 이 값도 추정이며, 실행 뒤 실제 시간으로 갱신한다.
- 라운드의 계획 추정은 아래 표의 최소안 합계(약 54.7노드시간)다. 아래 표는 [검토 결과](report/REVIEW_2026-11-09.md) 8.6절의 추정을 옮긴 것이다.

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

구현하면서 위 추정에 없는 단계가 생겼다. 따로 적지 않은 값은 추정이며 측정하지 않았다.

- 1라운드 끝의 용량 단계: 서버 기동과 짧은 기능 점검이다. 기동 시간을 보면 후보 2개 기준 0.5시간 이내로 추정한다.
- 라운드 작업마다 커널 시험: 이전 실행에서 약 2분이 걸렸다.
- 진단 작업의 `diag-B0`(DCP 비교 기준): 0.3시간 이내로 추정한다.
- 단계마다 결과 꾸러미 보고: 단계당 1분 이내로 추정한다. 꾸러미 생성 시간은 합성 데이터에서 5.7초로 쟀다(14절).
- 정리되지 않은 노드의 회수: 단계당 최대 300초다. 노드가 깨끗하면 실행하지 않는다.
- 제한 시간은 라운드 작업 12시간, 진단 작업 5시간, D×2 작업 6시간, 스크리닝 작업 12·6·8·7·3시간이다.

## 17. 사전 등록 전에 정할 사항과 한계

### 17.1 사전 등록 전에 정할 사항

1. **SLO:** 5절의 값은 가정이다.
2. **공통 `--max-model-len`:** 131,072는 예시 값이다. 고객의 최대 입력과 출력 길이에 맞추고, 라운드 후보들이 담을 수 있는 길이(C2는 238,656) 이하로 정한다.
3. **C2-s의 엔진별 시퀀스 한도:** 검토 결과 8.2절은 "엔진별 배치 한도 절반(4,096토큰, 32시퀀스)"으로 적었다. 32는 이전 값 64의 절반이다. 공통 값 128의 절반은 64이므로 64로 구현했다. 32로 하려면 `slo_protocol.json`의 `layouts.C2s.max_num_seqs`를 바꾼다.
4. **스로틀 게이트:** 측정 방법 검토는 "유휴 외의 스로틀 원인이 하나라도 있으면 무효"를 제안했다. 부하 중인 A100은 소프트웨어 전력 상한에 흔히 걸려 거의 모든 지점이 무효가 될 수 있다(추정). 그래서 하드웨어 감속·열 감속·전력 브레이크만 무효로 하고, 전력 상한은 비율만 기록하도록 구현했다. 바꾸려면 `gates.throttle`을 고친다.
5. **명시한 설정의 일치 여부:** `--async-scheduling`, `--compilation-config`, `--all2all-backend`, `--api-server-count 1`이 이전 실행의 자동 선택과 같은지는 스크리닝과 진단 실험의 실제 설정 기록으로 확인한다.
6. **스크리닝 선택 기준:** 짧은 출력 동등성의 한계(0.05 nats, 0.95, 1개)와 TPOT 배수(FP8 1.5배, NVFP4 1.2배)는 이 문서가 정한 값이다. 스크리닝 전에 확정한다.
7. **라운드 후보:** `bench_stats.py select`의 결과로 `ROUND_LAYOUTS`를 정하고 사전 등록 커밋에 적는다.

### 17.2 한계

- 라운드와 mp-multinode 실행은 GPU와 Azure에서 실행하지 않았다. 이 부분의 확인은 합성 데이터 단위 테스트, 가짜 백엔드를 쓴 프록시 시험, 셸 구문 검사로만 했다.
- 스크리닝 구성은 main 5dc653b로 한 번씩 실행했다. 측정값은 작업 기록의 결과 꾸러미에 있고, 이 문서의 메모리·KV 값(6.7절)은 아직 추정값이다. C4m, C8m, C16m은 아직 띄워 보지 않았다.
- mp-multinode의 node1 엔진 기록은 node1 로그를 Ray로 읽어서 얻는다. Ray가 멈췄거나 로그가 없으면 KV 합계는 "node0 only"로 남는다.
- DP×EP 구성의 엔진별 배치 토큰을 줄이면 긴 프롬프트의 프리필이 더 많은 스텝으로 나뉘어 TTFT가 늘 수 있다(추정). 스크리닝 결과는 6.3절의 설정에서 얻은 값이다.
- 회수는 규칙에 맞는 프로세스, placement group, 포트만 정리한다. 규칙에 맞지 않는 GPU 프로세스, 프로세스 없이 남은 GPU 메모리, GPU 오류는 회수하지 못한다. 그러면 단계는 실패로 기록된다.
- NIXL 사전 점검은 노드마다 GPU 0 하나만 검사한다. PD 서버는 노드마다 GPU 8개를 쓴다.
- `UCX_MEMTYPE_CACHE=n`은 UCX가 메모리 종류를 매번 확인하게 하므로 전송이 느려질 수 있다(추정). 이 값으로 띄운 PD는 단계 기록에 그 사실이 남는다.
- 작업 런타임이 `UCX_TLS=tcp`, `UCX_NET_DEVICES=eth0`을 넣는 이유는 확인하지 않았다. PD 서버만 이 값을 덮어쓰고, 다른 단계는 작업 환경을 그대로 쓴다.
- 길이 0 서술자 가설은 디버그 재기동의 "Registering descs" 줄로만 확인한다. 이 하네스는 포트를 고치지 않는다.
- PD에서 NIXL의 UCX가 A100 노드 사이에서 GPU 메모리를 직접 읽는지는 확인하지 않았다. UCX가 TCP나 호스트 메모리 경유를 고르면 느리다. 지점별 InfiniBand 송수신량과 서버 로그의 NIXL·UCX 줄로 확인한다.
- `report.py`는 MLflow를 쓸 수 없어도 0으로 끝난다. `bench_stats.py`는 출력의 경고 문구로 전송 실패를 판단한다. `report.py`의 경고 문구가 바뀌면 실패를 놓칠 수 있다. 작업 끝의 보고가 모든 키를 다시 보낸다.
- node1의 계측 파일은 Ray가 살아 있을 때만 가져온다. 작업이 끝날 때 Ray가 이미 멈췄으면 마지막으로 가져온 사본을 쓴다.
- 꾸러미 크기(14절)는 합성 데이터로 계산한 추정이다.
- 32K 문서의 `prompt_logprobs`는 로짓 메모리가 커서 서버가 멈출 수 있다(추정). 그래서 단계 마지막에 실행하고, 진단 실험에서 먼저 확인한다.
- `dx2-D`, `scr-Dx2`, `scr-PD`에서 node1 서버의 실제 설정은 실행 중에 검사하지 않는다. 로그는 node1의 출력 폴더에 남는다. 결과 꾸러미에는 node0 서버의 설정 기록과 node1 서버 로그의 끝부분(최대 16 KB)만 들어간다.
- NCCL 검사는 노드별 전송 방식 건수와 통신기 구성원(`Init COMPLETE`)을 센다. 채널 연결 줄을 통신기별로 정확히 나누지는 않는다. `NET/Socket` 0건 조건에는 영향이 없다.
- nvidia-smi 시각은 UTC로 해석한다.
- 스텝별 스케줄 토큰 수는 `vllm:iteration_tokens_total` 히스토그램의 엔진별 1초 합계·개수로만 남는다. 스텝별 CUDA graph 방식은 기록하지 않는다.
- 전문가별 라우팅 토큰 수(EPLB 균형 로그)는 구현하지 않았다.
- 이전 단계(`epc-*`, `docsweep` 등)는 이전 기록과의 비교를 위해 기존 동작을 유지한다. 다만 측정 지점마다 다른 시드를 쓰도록 고쳤다. 이 단계들은 시작에 실패하면 여전히 `--enforce-eager`로 다시 시도하고, `/metrics` 표본은 여전히 엔진 값을 더한다.

## 18. 파일

| 파일 | 내용 |
|---|---|
| `aml/src/bench/slo_protocol.json` | 구성, 공통 플래그, 워크로드, 격자, SLO, 시드, 게이트, 판정 규칙, 스크리닝과 선택 규칙, 추정, 가격 |
| `aml/src/bench/protocol.sh` | `entry.sh`가 읽는 측정 단계 실행부: 계측, 정리 상태 확인과 회수, 실제 설정·NCCL 검사, 길이 대체, 프로세스 그룹 종료, D×2 복제본과 PD 디코드 서버, mp-multinode의 headless DP 엔진, NIXL 사전 점검과 디버그 재기동, PD 확인 요청, 결과 꾸러미 보고 |
| `aml/src/bench/slo_suites.py` | 단계 설정 생성, 라운드 계획, 측정 스위트(parity, warmup, partr, slo, realtext, soak, paritylong, screen), 엔진별 `/metrics` 시계열, 게이트 |
| `aml/src/bench/bench_stats.py` | 꾸러미 생성과 보고, 라운드 분석과 판정, 스크리닝 선택, Markdown 요약 |
| `aml/src/bench/lb_proxy.py` | D×2용 부하 분산기, PD용 프리필·디코드 프록시, 확인 요청(`--smoke`) |
| `aml/src/bench/node_tools.py` | Ray 작업으로 node1 파일과 로그 끝부분·기록을 가져오고, 정리되지 않은 노드를 회수하고, 두 노드에서 NIXL 사전 점검을 실행하고, PD 서버의 서술자 줄을 모은다 |
| `aml/src/bench/nixl_preflight.py` | PD 서버를 띄우기 전 NIXL·UCX의 VRAM 등록 사전 점검(GPU 1개, JSON 한 줄) |
| `aml/src/bench/parity_corpus.json` | 출력 동등성과 W1-real용 공유 저작물·법령 텍스트 |
| `aml/jobs/serving-screen-fp8-nd96.yml` | FP8 스크리닝 작업 |
| `aml/jobs/serving-screen-fp8-dp-nd96.yml` | FP8 DP×EP 스크리닝 작업(mp-multinode 실행) |
| `aml/jobs/serving-screen-nvfp4-nd96.yml` | NVFP4 1노드 스크리닝 작업 |
| `aml/jobs/serving-screen-nvfp4-2node-nd96.yml` | NVFP4 2노드 스크리닝 작업(D×2, PD) |
| `aml/jobs/serving-screen-nvfp4-pd-nd96.yml` | PD 재실행 작업(Dref, NIXL 사전 점검 뒤 PD) |
| `aml/jobs/serving-diag-nd96.yml` | 진단 작업 |
| `aml/jobs/slo-round-nd96.yml` | 라운드 작업 |
| `aml/jobs/slo-nvfp4-2replica-nd96.yml` | D×2 작업 |
| `aml/jobs/collect-slo-outputs.yml` | 결과 수집 작업(대체 경로) |

`aml/src/bench/`는 데모 작업이 받는 스크립트 묶음에 들어가지 않는다. 고정 커밋에서 파일을 받는 측정 작업만 이 폴더를 내려받는다.
