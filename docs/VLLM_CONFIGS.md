# A100에서 A.X-K2를 위한 vLLM 구성

스냅샷: 2026-10-22. 이 카탈로그는 이 저장소가 A100에서 A.X-K2를 서빙한 모든
vLLM 구성을 나열한다. 대상은 두 데모 프로필, expert-parallel 비교의 레이아웃,
그리고 [SERVING_DESIGN.md](SERVING_DESIGN.md)(아래에서 "설계 문서")에 기록된
그 밖의 구성이다. 각 항목은 설계 문서와 `evidence/`의 사실을 옮겨 적고 출처 절을
밝힌다.

## 운영 상태 (2026-10-22)

2026-10-22 현재 데모 GPU는 꺼져 있다. 두 데모 프로필(FP8, NVFP4)은 사용자의
요청에 따라 꺼졌다. 마지막 job은 취소되었고 GPU 클러스터에는 노드가 없다. 따라서
과금되는 GPU는 없다. 다시 서빙하려면 supervisor를 켠다. FP8은 `demo/ctl.sh on`을,
NVFP4 프로필은 관리자 전원 스위치(`/models/nvfp4/api/admin/power`)를 쓴다. 또는
두 프로필 모두 `demo/target.py --directory <target-directory> on --keep-on`을 쓴다
(`demo/README.md`). 노드가 비어 있으면 FP8 프로필은 전원을 켠 뒤 약 20-30분 후에
준비된다. `demo/README.md`에는 25분으로, 설계 문서 9의 두 차례 복구에서는 22분과
32분으로 기록되어 있다. 이 시간의 대부분은 694 GB 체크포인트 다운로드(2026-10-21
기준 노드당 약 10-12분)와 약 4분의 서버 시작이다. 단일 노드 NVFP4 레이아웃은
2026-10-21에 다운로드와 시작에 약 15분이 걸렸다(459 s와 442 s). 이 클러스터는
예약 용량이 없는 low-priority(Spot) 클러스터다. 따라서 노드가 늦게 돌아오거나 아예
돌아오지 않을 수 있다. 2026-10-21에 AML은 비워진 노드 세 개를 반납했고 새 노드를
할당하는 데 약 10분이 걸렸다. 이전 탐색에서는 12개 리전 중 두 곳에서만 빈 노드 두
개를 찾았다(설계 문서 9). 이와 별도로, 구독의 거버넌스 자동화가 매일 데모 웹 앱을
Always On이 꺼진 F1 티어로 옮겨 왔다. Always On에는 Basic 티어 이상이 필요하다.
Always On이 없으면 App Service는 유휴 앱을 언로드한다. 이 경우 웹 앱 프로세스에서
실행되는 supervisor와 역방향 GPU 링크도 멈춘다(설계 문서 8.5, 8.6). 또한 데모
job은 웹 앱에서 스크립트를 다운로드한다. 데모 전에 플랜 티어와 Always On을
확인한다.

## 상태 레이블과 측정 규칙

| 레이블 | 의미 |
| --- | --- |
| 현재 데모 | 데모가 서빙하는 레이아웃이다. |
| 테스트함, 권장하지 않음 | 시작했고 기능 검사와 tool-calling 검사를 통과했다. 그러나 같은 할당에서 종합적으로 데모 레이아웃보다 나쁘게 측정되었다(해당 행 참조). |
| 처리량 전용 | 동시 요청이 많을 때만 데모 레이아웃보다 빠르고, 단일 요청에서는 더 느리다. |
| 테스트한 대안 | 동작한다. 데모 레이아웃 대비 트레이드오프가 문서화되어 있다. |
| 다른 조건의 기록 | 다른 설정으로 측정했다. expert-parallel 비교와 비교할 수 없다. |
| 검증 전용 | 서빙이 아니라 정확성이나 기능 확인에 사용했다. |
| A100에서 시작 불가 | vLLM 0.23이 이 GPU에서 해당 구성을 거부한다. |

- 속도 수치는 무작위 프롬프트와 `--ignore-eos`를 사용한 `vllm bench serve`
  결과다. 각 지점은 한 번만 실행했다(단일 샘플). 따라서 몇 퍼센트의 차이는
  노이즈일 수 있다.
- 2026-10-03과 2026-10-04의 측정은 모든 지점에 같은 seed를 썼다. 그래서 뒤 지점이 앞 지점의
  prompt를 prefix cache에서 재사용했다. 그래서 일부 처리량은 높게, TTFT와 동시 요청 TPOT는 짧게 나왔다(설계 문서 8.1 정정). 2026-10-21
  expert-parallel 비교는 지점마다 다른 seed를 써서 이 문제가 없다.
- 비율은 같은 할당에서 측정한 레이아웃끼리 비교한다. 설계 문서의 서로 다른 절에
  있는 수치는 서로 다른 조건(체크포인트 레이블, 컨텍스트, 시퀀스 수, batched
  token 수, 스케줄링, 인터커넥트)에서 측정했다. 따라서 서로 비교하지 않는다.

## 요약

| ID | 상태 | 체크포인트 | GPU | 레이아웃 | Executor | DSA | Job 템플릿, phase | 설계 문서 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `demo-fp8` | 현재 데모(꺼짐) | FP8 `4237c6a6` | 16 | TP8 x PP2 | Ray | native | `demo-fp8-nd96.yml`, `demo` | 8.1, 8.5, 8.7 (A) |
| `demo-nvfp4` | 현재 데모(꺼짐) | NVFP4 `9e2e804e` | 8 | TP8 | mp | native | `demo-nvfp4-nd96.yml`, `demo` | 8.4, 8.5, 8.7 (D) |
| `epc-fp8-tp16ep16` | 테스트함, 권장하지 않음 | FP8 `4237c6a6` | 16 | TP16 + EP (EP16) | Ray | native | `ep-compare-fp8-nd96.yml`, 같은 이름 | 8.7 (B) |
| `epc-fp8-tp8pp2ep8` | 테스트함, 권장하지 않음 | FP8 `4237c6a6` | 16 | TP8 x PP2 + EP (stage마다 EP8) | Ray | native | `ep-compare-fp8-nd96.yml`, 같은 이름 | 8.7 (C1) |
| `epc-fp8-tp16` | 테스트함, 권장하지 않음 | FP8 `4237c6a6` | 16 | TP16 | Ray | native | `ep-compare-fp8-nd96.yml`, 같은 이름 | 8.7 (B0) |
| `epc-fp8-dp2tp8ep16` | 처리량 전용 | FP8 `4237c6a6` | 16 | DP2 x TP8 + EP (EP16) | Ray | native | `ep-compare-fp8-nd96.yml`, 같은 이름 | 8.7 (C2) |
| `epc-nvfp4-tp8ep8` | 테스트함, 권장하지 않음 | NVFP4 `9e2e804e` | 8 | TP8 + EP (EP8) | mp | native | `ep-compare-nvfp4-nd96.yml`, 같은 이름 | 8.7 (E) |
| `dense-pp2` | 테스트한 대안(약 60K 토큰까지의 컨텍스트) | FP8, `2287ca45`에서 측정 | 16 | TP8 x PP2 | Ray | dense | `verify-remaining-nd96-hub.yml`, `dense-pp2` | 5, 8.1-8.3 |
| `dense-tp16` | 다른 조건의 기록 | FP8, `2287ca45`에서 측정 | 16 | TP16 | Ray | dense | `verify-remaining-nd96-hub.yml`, `dense-tp16` | 8.1 |
| `dense-tp16-eagle3` | 검증 전용(사용 불가) | FP8 `2287ca45` + EAGLE3 `24958e91` | 16 | TP16 + EAGLE3 drafter | Ray | dense | `verify-remaining-nd96-hub.yml`, `dense-tp16-eagle3` | 8.1 |
| `native-pp2-bi`, `dense-pp2-bi` | A100에서 시작 불가 | FP8, `2287ca45`에서 시도 | 16 | TP8 x PP2, batch-invariant 모드 | Ray | native, dense | `verify-remaining-nd96-hub.yml`, 같은 이름 | 8.1 |
| `nvfp4-tp8-b2048` | 테스트한 대안 | NVFP4 `9e2e804e` | 8 | TP8, batched token 2,048개 | mp | native | `nvfp4-a100-nd96.yml`, `nvfp4-tp8-b2048` | 8.4 |
| 2-layer cut | 검증 전용 | FP8 2-layer cut | 8, 이후 16 | TP8, 이후 TP8 x PP2 | vLLM 기본값, 이후 Ray | native, dense | `native-dsa-bench-nd96-hub.yml`, smoke phase | 7, 8.2, 8.3 |

## 공통 엔진과 명령

- **엔진.** 모든 항목은 `docker.io/vllm/vllm-openai:v0.23.0-cu129`
  (vLLM 0.23.0)에서 실행한다. job 시작 시 `aml/src/apply_overlay.py`가 SKT fork
  `axk2-v0.23.0` @ `023bc760`을 추가한다. 이 fork에는 upstream PR #38476의
  포팅(native DSA가 사용하는 `TRITON_MLA_SPARSE` 백엔드와 Triton indexer)과 두 줄짜리
  Ampere MLA 수정이 들어 있다(설계 문서 4, 5b). 2026-11-09 변경으로 이 포팅에 indexer 디코드 kernel의
  버그 수정이 들어갔다(설계 문서 5b). 이 카탈로그의 측정은 모두 수정 전 kernel로 실행했다. 버그가 그 결과에 준
  영향은 확인하지 않았다(처리량, 기능·tool 점검에서는 드러나지 않았다). NVFP4 항목은
  `aml/src/nvfp4_sm80_port.json`(`NVFP4_SM80_PORT=1`, 설계 문서 8.4)도 적용한다.
  A100에서 FP8 linear 레이어와 MoE 레이어는 Marlin W8A16으로, NVFP4 expert는
  Marlin W4A16으로 실행되고, KV cache는 BF16이다.
- **하드웨어.** 최대 세 노드로 이루어진 Azure ML low-priority(Spot) 클러스터의
  Standard_ND96amsr_A100_v4 노드(NVLink로 연결된 8 x A100 80GB, 노드 간
  InfiniBand)를 쓴다. FP8 데모는 노드 두 개를, NVFP4 데모는 노드 한 개를 쓴다.
  따라서 둘 다 켜면 클러스터가 가득 찬다(설계 문서 8.5).
- **체크포인트.** FP8: `skt/A.X-K2` @ `4237c6a60cec6ae449854870d01a42665a1166aa`.
  Hub는 2026-10-20부터 이전 pin `2287ca45`에 대해 404를 반환한다.
  `4237c6a6`은 `2287ca45`에 대해 기록된 파일 수, 전체 바이트 수, 아홉 개 shard
  해시가 모두 일치한다. PR #35는 이 확인 결과를 기록하고 데모 job의 pin을 바꾼다.
  PR #35가 병합되기 전까지 main과 이 브랜치의 `aml/jobs/demo-fp8-nd96.yml`은 여전히
  `2287ca45`를 지정한다. 설계 문서 8.1-8.3의 실행은 `2287ca45`를 측정했고,
  expert-parallel 비교는 `4237c6a6`을 사용했다. NVFP4:
  `skt/A.X-K2-NVFP4` @ `9e2e804e80f8d1b3afba5d7938173cec1ed46b49`. EAGLE3
  drafter: `skt/A.X-K2-EAGLE3` @ `24958e91737d760908f73a8af4b6e06080fc5c1d`.
- **DSA 모드.** native 모드는 체크포인트를 변경 없이 서빙한다. indexer는 query마다
  상위 2,048개 key를 선택한다. dense 모드는 config에 `index_topk`, `index_n_heads`,
  `index_head_dim`이 없는 체크포인트 디렉터리 사본을 서빙한다
  (`aml/src/make_dense_dir.py`). dense 모드는 컨텍스트가 2,048 토큰에 들어가는
  동안에만 DSA와 같다(설계 문서 5).
- **명령.** `aml/src/entry.sh`는 모든 서버를 다음과 같이 시작한다.

  ```bash
  vllm serve <model dir> --served-model-name axk2 --host 0.0.0.0 --port 8000 \
    --max-model-len <len> --gpu-memory-utilization <util> --max-num-seqs <seqs> \
    --reasoning-parser deepseek_v3 --no-enable-log-requests <phase flags>
  ```

  `PHASE_PLAN` phase(`plan_phase`)는 `--tool-call-parser hermes
  --enable-auto-tool-choice`, `--max-num-batched-tokens 8192`(`nvfp4-tp8-b2048`에서는
  2,048), 그리고 각자의 레이아웃 플래그를 추가한다. `epc-*` phase는
  `--scheduling-policy priority`도 추가한다. 데모 phase는 같은 parser 플래그,
  8,192 batched token, priority 스케줄링을 추가한다. 이전의 `MODES` phase
  (`smoke-*`, `full-*`)는 자신의 레이아웃 플래그만 추가한다. 서버가 CUDA graph로
  healthy 상태가 되지 않으면 `entry.sh`는 `--enforce-eager`로 한 번 재시도하고 이를
  `eager_fallback`으로 보고한다. 참고로 FP8 데모의 전체 명령은 다음과 같다.

  ```bash
  vllm serve <checkpoint dir> --served-model-name axk2 --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray \
    --max-model-len 262144 --max-num-seqs 64 --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 --scheduling-policy priority \
    --reasoning-parser deepseek_v3 --tool-call-parser hermes --enable-auto-tool-choice \
    --no-enable-log-requests
  ```

- **멀티 노드 job.** 노드 0이 Ray head를 시작하고 다른 노드가 여기에 참여한다.
  `IB_PROBE=1`이면 두 노드가 먼저 TCP와 InfiniBand에서 NCCL을 측정한다. 서빙은
  NCCL이 InfiniBand를 선택한 경우에만 InfiniBand를 쓴다(설계 문서 8.1). 설계 문서
  8.2와 8.3의 실행은 이 probe 이전에 이루어졌고 TCP를 사용했다.

## 현재 데모

### `demo-fp8`: FP8, 두 노드에 걸친 TP8 x PP2

| 항목 | 값 |
| --- | --- |
| 상태 | 현재 데모. 2026-10-22부터 꺼져 있다 |
| 체크포인트 | 배포 기준 `skt/A.X-K2` @ `4237c6a6`(FP8, block-scaled). 이 브랜치의 템플릿은 여전히 `2287ca45`를 지정한다(체크포인트 항목 참조) |
| GPU | 노드 2개, 16 x A100 80GB |
| 병렬 플래그 | `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray`. 레이어 0-30은 노드 0에, 레이어 31-60과 LM head는 노드 1에 있다 |
| 기타 플래그 | `--max-model-len 262144 --max-num-seqs 64 --gpu-memory-utilization 0.90 --max-num-batched-tokens 8192 --scheduling-policy priority`, parser 플래그는 위와 같다 |
| Executor, DSA | Ray, native |
| Job, env | `aml/jobs/demo-fp8-nd96.yml`, `PHASE_PLAN=demo`, `IB_PROBE=1`, `TP=8`, `PP=2`, `FULL_MAX_MODEL_LEN=262144`, `FULL_MAX_NUM_SEQS=64`, `FULL_GPU_MEM_UTIL=0.90`, `DEMO_RESTARTS=2`, `RUN_KERNEL_TESTS=0`. supervisor는 제출할 때마다 `AXK2_LINK_URL`, `AXK2_LINK_TOKEN`, `AXK2_SRC_URL`, `AXK2_SRC_SHA256`을 채운다. |
| 컨텍스트, KV cache | 262,144 토큰. KV cache 673,792 토큰(GPU당 weight 42.3 GiB, KV 25.0 GiB) |
| 시작 시간 | 2026-10-21 baseline A로 healthy까지 261 s. 설계 문서 8.1에서는 262 s |
| 측정값(설계 문서 8.7, baseline A, 단일 샘플) | 동시성 1 / 32에서 output tok/s: 1K 입력 48 / 404, 8K 입력 42 / 208, 32K 입력 32 / 61. 단일 요청의 median TPOT 20.7-22.7 ms. 32K 입력, 동시성 32에서 동시에 실행되는 요청은 20개다. 기능 4/4, tool 9/9. |
| 이전 기록(다른 조건) | 설계 문서 8.1(`native-pp2`, `2287ca45`, priority 스케줄링 없음): tool 9/9, SKT needle test 252,728 토큰까지 9/9, tech report Fig. 7 조건에서 1K-8K 입력 기준 B200 노드 하나의 0.43-0.51x(2K-8K는 같은 seed로 인한 prefix cache 재사용으로 부풀었다. 2026-10-21 값으로 환산하면 8K 0.39x). 설계 문서 8.2(컨텍스트 65,536, 시퀀스 128, vLLM의 A100 기본값인 2,048 batched token, TCP): 302 s 후 healthy, KV cache 724,544 토큰. |
| 주의 사항 | supervisor가 이 job을 렌더링하고 제출한다. 직접 제출하지 않는다. 출력은 비트 단위로 결정적이지 않고, batch-invariant 모드는 시작할 수 없다(설계 문서 8.1). |

### `demo-nvfp4`: NVFP4, 단일 노드 TP8

| 항목 | 값 |
| --- | --- |
| 상태 | 현재 데모. 2026-10-22부터 꺼져 있다 |
| 체크포인트 | `skt/A.X-K2-NVFP4` @ `9e2e804e`(ModelOpt 혼합 정밀도: NVFP4 routed expert, FP8 attention과 shared expert) |
| GPU | 노드 1개, 8 x A100 80GB |
| 병렬 플래그 | `--tensor-parallel-size 8 --pipeline-parallel-size 1 --distributed-executor-backend mp` |
| 기타 플래그 | `--max-model-len -1 --max-num-seqs 128 --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --scheduling-policy priority`, parser 플래그는 위와 같다 |
| Executor, DSA | mp(multiprocessing, Ray 없음), native |
| Job, env | `aml/jobs/demo-nvfp4-nd96.yml`, `PHASE_PLAN=demo`, `DEMO_EXECUTOR=mp`, `FULL_REPO=skt/A.X-K2-NVFP4`, `NVFP4_SM80_PORT=1`, `TP=8`, `PP=1`, `FULL_MAX_MODEL_LEN=-1`, `FULL_MAX_NUM_SEQS=128`, `FULL_GPU_MEM_UTIL=0.92`, `DEMO_RESTARTS=2`. NVFP4 supervisor가 link 값과 source 값을 채운다. |
| 컨텍스트, KV cache | 254,016 토큰으로 자동 맞춤. KV cache 254,016 토큰(GPU당 weight 48.5 GiB, KV 18.5 GiB) |
| 시작 시간 | 2026-10-21 baseline D로 healthy까지 442 s. 설계 문서 8.4에서는 433 s |
| 측정값(설계 문서 8.7, baseline D, 단일 샘플) | 동시성 1 / 32에서 output tok/s: 1K 입력 48 / 445, 8K 입력 42 / 188, 32K 입력 27 / 51. 단일 요청의 median TPOT 20.4-22.4 ms. 32K 입력, 동시성 32에서 동시에 실행되는 요청은 7개다. 기능 4/4, tool 9/9. |
| 이전 기록(다른 조건) | 설계 문서 8.4(`nvfp4-tp8`, priority 스케줄링 없음): needle 244,898 토큰까지 9/9. Fig. 7 조건에서 1K-4K 입력 기준 B200 노드 하나의 0.47-0.49x(2026-10-04 작업은 모든 측정점에 같은 seed를 써서 prefix cache 재사용이 섞였다. 재측정 전까지 사용하지 않는다. 같은 날 D 기준 1K는 0.47x(889.8/1,900)). |
| 주의 사항 | expert는 W4A16으로 실행된다(A100에는 FP4 tensor core가 없다). 엔진에는 vLLM 0.24 변경 두 건의 backport가 필요하다. 품질은 기능 검사, tool call, needle로만 확인했다(설계 문서 8.4). |

## Expert-parallel 비교 (2026-10-21, 설계 문서 8.7)

**공통 설정.** native DSA. FP8 `4237c6a6`은 시퀀스 64개, GPU 메모리 0.90으로,
NVFP4 `9e2e804e`는 시퀀스 128개, 0.92로 실행했다(데모 값). 모든 레이아웃은
`--max-model-len -1 --max-num-batched-tokens 8192
--scheduling-policy priority`와 parser 플래그를 추가한다. job은 다음과 같다.

- `aml/jobs/ep-compare-fp8-nd96.yml`:
  - `PHASE_PLAN="epc-fp8-tp8pp2 epc-fp8-tp16ep16 epc-fp8-tp8pp2ep8 epc-fp8-tp16 epc-fp8-dp2tp8ep16"`;
  - `IB_PROBE=1`, `TP=8`, `PP=2`, `FULL_MAX_MODEL_LEN=-1`, `FULL_MAX_NUM_SEQS=64`,
    `FULL_GPU_MEM_UTIL=0.90`;
  - 5시간 `limits.timeout`.
- `aml/jobs/ep-compare-nvfp4-nd96.yml`:
  - `PHASE_PLAN="epc-nvfp4-tp8 epc-nvfp4-tp8ep8"`;
  - `FULL_REPO=skt/A.X-K2-NVFP4`, `NVFP4_SM80_PORT=1`, `TP=8`, `PP=1`,
    `FULL_MAX_MODEL_LEN=-1`, `FULL_MAX_NUM_SEQS=128`, `FULL_GPU_MEM_UTIL=0.92`;
  - 3시간 `limits.timeout`.

`epc-fp8-tp8pp2`(A)와 `epc-nvfp4-tp8`(D)는 데모 레이아웃이다. 같은 할당에서
baseline으로 다시 측정했다.

`--enable-expert-parallel`은 크기가 아니라 스위치다. vLLM 0.23은 각 MoE 레이어를
해당 pipeline stage의 DP x TP rank에 걸쳐 expert parallel로 만들고 TP 1로 실행한다.
attention은 자신의 레이아웃을 유지한다. shared expert는 DP = 1인 레이아웃에서 TP로 나뉜 채 유지되고,
DP2 x TP8 + EP(C2)에서는 GPU마다 통째로 복제된다(설계 문서 8.7). 코드 변경이
필요한 레이아웃은 없었다. 모든 레이아웃이 CUDA graph로 시작했고 기능 검사 4/4와
tool call 9/9를 통과했다.

| Phase(8.7의 레이블) | 상태 | 병렬 플래그 | EP 크기, GPU당 expert 수 | 컨텍스트 / KV cache 토큰 | GPU당 weight / KV | healthy까지 |
| --- | --- | --- | --- | --- | --- | ---: |
| `epc-fp8-tp8pp2` (A) | baseline, = `demo-fp8` | `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray` | 없음 | 262,144 / 673,792 | 42.3 / 25.0 GiB | 261 s |
| `epc-fp8-tp16ep16` (B) | 테스트함, 권장하지 않음 | `--tensor-parallel-size 16 --pipeline-parallel-size 1 --distributed-executor-backend ray --enable-expert-parallel` | EP16, 256개 중 16개 | 262,144 / 309,376 | 43.3 / 22.6 GiB | 231 s |
| `epc-fp8-tp8pp2ep8` (C1) | 테스트함, 권장하지 않음 | `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray --enable-expert-parallel` | 각 stage에서 EP8, 32 | 262,144 / 673,216 | 42.3 / 25.0 GiB | 151 s |
| `epc-fp8-tp16` (B0) | 테스트함, 권장하지 않음(B의 대조군) | `--tensor-parallel-size 16 --pipeline-parallel-size 1 --distributed-executor-backend ray` | 없음 | 262,144 / 310,080 | 43.3 / 22.6 GiB | 251 s |
| `epc-fp8-dp2tp8ep16` (C2) | 처리량 전용 | `--tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-backend ray --enable-expert-parallel` | EP16, 16 | 238,656 / 238,720 + 238,656(엔진 두 개) | 46.4 / 17.4 GiB | 221 s |
| `epc-nvfp4-tp8` (D) | baseline, = `demo-nvfp4` | `--tensor-parallel-size 8 --pipeline-parallel-size 1 --distributed-executor-backend mp` | 없음 | 254,016 / 254,016 | 48.5 / 18.5 GiB | 442 s |
| `epc-nvfp4-tp8ep8` (E) | 테스트함, 권장하지 않음 | `--tensor-parallel-size 8 --pipeline-parallel-size 1 --distributed-executor-backend mp --enable-expert-parallel` | EP8, 32 | 253,120 / 253,120 | 48.5 / 18.5 GiB | 312 s |

healthy까지 걸린 시간은 phase마다 1회 측정했고 실행 순서와 cache 상태의 영향을 받는다. FP8 job은 A, B, C1,
B0, C2 순서로, NVFP4 job은 D 다음 E 순서로 실행했다. 따라서 레이아웃끼리 시작 시간을 비교하지 않는다.

**측정값(지점당 단일 샘플).** baseline은 절댓값이고, 나머지는 A(FP8) 또는
D(NVFP4) 대비 비율이다. 입력은 1K / 8K / 32K 토큰, 출력은 1,024 토큰이다.

| Phase | Output tok/s, 동시성 1 | Output tok/s, 동시성 32 | Median TPOT, 단일 요청 | 실행 중 요청 수, 32K 입력, 동시성 32 |
| --- | --- | --- | --- | ---: |
| A `epc-fp8-tp8pp2` | 48 / 42 / 32 | 404 / 208 / 61 | 20.7-22.7 ms | 20 |
| B `epc-fp8-tp16ep16` | 0.70 / 0.72 / 0.70 | 0.98 / 0.98 / 0.83 | 1.40-1.44x | 9 |
| C1 `epc-fp8-tp8pp2ep8` | 0.88 / 0.89 / 0.91 | 0.92 / 0.92 / 0.92 | 1.12-1.13x | 20 |
| B0 `epc-fp8-tp16` | 0.79 / 0.82 / 0.77 | 1.04 / 1.04 / 0.89 | 1.24-1.26x | 9 |
| C2 `epc-fp8-dp2tp8ep16` | 0.50 / 0.54 / 0.56 | 1.12 / 1.36 / 1.18 | 1.88-1.99x | 14(두 엔진 합계) |
| D `epc-nvfp4-tp8` | 48 / 42 / 27 | 445 / 188 / 51 | 20.4-22.4 ms | 7 |
| E `epc-nvfp4-tp8ep8` | 0.93 / 0.93 / 0.93 | 0.95 / 0.93 / 0.94 | 1.08x | 7 |

**레이아웃별 주의 사항.**

- B와 B0: TP16이 두 노드에 걸치므로 vLLM은 custom all-reduce를 끈다. 그래서 모든
  레이어의 all-reduce가 InfiniBand 위의 NCCL을 사용한다. MLA cache는 각각 61개
  레이어를 모두 가진 16개 rank에 복제된다. 따라서 KV cache는 A 토큰 수의 0.46x만
  담는다.
- C1과 E: DP = 1이면 EP는 아무것도 dispatch하지 않는다. 각 rank는 자신의 expert를
  계산하고, 출력은 EP가 없을 때처럼 all-reduce된다. output tok/s는 baseline보다
  8-12%(C1), 5-7%(E) 낮았다. 원인은 그룹 라우팅과 linear expert 배치가 겹친 부하 불균형으로
  추정한다(측정하지 않음, 설계 문서 8.7).
- C2:
  - 로그에 "Using ray distributed inference because data_parallel_backend
    is ray"가 출력된다. DP rank 0은 노드 0에서, DP rank 1은 노드 1에서 실행된다.
  - 두 API server 프로세스(`ApiServer_0`, `ApiServer_1`)가 하나의 endpoint를
    서빙했다.
  - MoE dispatch는 InfiniBand 위의 allgather/reduce-scatter를 사용한다
    (`AgRsAll2AllManager`).
  - 8K 입력, 동시성 32에서 median TTFT는 A의 1.56x이고, 컨텍스트는 238,656 토큰으로
    줄어든다. 줄어든 주된 이유는 GPU마다 통째로 복제한 shared expert다(GPU당 약 2.3 GiB, 추정).
  - A와의 비교에는 다른 조건이 섞였다. API server가 2개였고, batched token과 시퀀스 한도가 엔진마다
    적용되어 전체 한도가 A의 2배였다.
- DeepEP는 이미지에 들어 있지만 Hopper와 Blackwell용으로만 빌드되어 A100에서
  실행되지 않는다. EPLB, `--enable-ep-weight-filter`, round-robin expert 배치는
  테스트하지 않았다.

## 설계 문서의 기타 구성

이 구성들은 2026-10-02부터 2026-10-04까지 FP8 체크포인트 `2287ca45` 또는 NVFP4
체크포인트 `9e2e804e`로 측정했다. priority 스케줄링은 쓰지 않았고, 항목마다 나열한
설정을 사용했다. 이들은 각자의 조건에서 얻은 기록이다. 위의 expert-parallel
비교와 비교하지 않는다. 서로 다른 절의 기록끼리도 비교하지 않는다.

### `dense-pp2`: FP8 dense 모드, TP8 x PP2 (설계 문서 5, 8.1-8.3)

| 항목 | 값 |
| --- | --- |
| 상태 | 테스트한 대안. 약 60K 토큰까지의 컨텍스트에만 해당한다 |
| 레이아웃 | 데모의 레이아웃(`--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray`)으로 체크포인트의 dense 사본(`/tmp/axk2-full-dense`, `make_dense_dir.py`로 생성)을 서빙한다. decode는 `TRITON_MLA`를, prefill은 FlashAttention-2 MLA를 사용한다. |
| Job | `verify-remaining-nd96-hub.yml`, phase `dense-pp2`(`IB_PROBE=1`, 컨텍스트 262,144, 시퀀스 64, 0.90, batched token 8,192). `native-dsa-bench-nd96-hub.yml`과 `full-native-dense-nd96-hub.yml`, `MODES="native dense"`에서 나온 phase `full-dense-tp8-pp2`(컨텍스트 65,536, 시퀀스 128, vLLM의 A100 기본값인 batched token 2,048). |
| 시작 시간, KV cache | 8.1: 181 s, 766,032 토큰. 8.2: 171 s, 821,312 토큰(GPU당 weight 41.7 GiB, KV 26.8 GiB), TCP 사용. 8.3: 181 s, 843,920 토큰, TCP 사용. |
| 측정값(지점당 단일 샘플) | 8.1, native TP8 x PP2와 같은 할당, tech report Fig. 7 조건(동시성 32, 출력 1,024 토큰): 1K부터 120K 입력까지 native total tok/s의 1.2-2.4x. 8.2: native output tok/s의 1.1-1.6x. 두 측정 모두 같은 seed로 인한 prefix cache 재사용이 섞여 이 비율은 재측정 전까지 사용하지 않는다(설계 문서 8.1 정정). 기능 6/6(8.2), tool 9/9(8.1). |
| 주의 사항 | 컨텍스트 2,048 토큰까지만 정확하다(설계 문서 5). needle은 60K 토큰까지(8.2에서 12/12)와 32K에서(8.1에서 3/3) 찾았지만, 126K나 253K 토큰에서는 하나도 찾지 못했다(8.1에서 0/6). 더 긴 컨텍스트에는 native DSA를 사용한다. |

### `dense-tp16`: FP8 dense 모드, 두 노드에 걸친 TP16 (설계 문서 8.1)

| 항목 | 값 |
| --- | --- |
| 상태 | 다른 조건의 기록. dense 모드와 32,768 토큰 컨텍스트로 실행했으며 expert-parallel 비교의 TP16 레이아웃이 아니다. native TP16은 `epc-fp8-tp16`(B0)과 `epc-fp8-tp16ep16`(B)을 참조한다. |
| 레이아웃 | `--tensor-parallel-size 16 --pipeline-parallel-size 1 --distributed-executor-backend ray`, dense 체크포인트 |
| Job | `verify-remaining-nd96-hub.yml`, phase `dense-tp16`(컨텍스트 32,768, 시퀀스 64, 0.90, batched token 8,192, `IB_PROBE=1`) |
| 시작 시간, KV cache | 241 s, 362,848 토큰 |
| 측정값(지점당 단일 샘플) | 같은 할당의 `dense-pp2` 대비, 동시성 32, 출력 1,024 토큰: total tok/s가 1K 입력에서 1.41x, 4K에서 1.20x, 8K에서 1.12x(1,386 / 2,638 / 3,449 대 980 / 2,202 / 3,070). 4K·8K 값은 같은 seed로 인한 prefix cache 재사용이 섞였다. 기능 4/4. |
| 주의 사항 | KV cache가 `dense-pp2` 토큰 수의 0.47x만 담는다. 따라서 이 레이아웃은 짧은 컨텍스트에만 적합하다. dense 모드의 모든 주의 사항이 적용된다. |

### `dense-tp16-eagle3`: EAGLE3 drafter를 사용한 TP16 (설계 문서 8.1)

| 항목 | 값 |
| --- | --- |
| 상태 | 검증 전용. A100에서 사용할 수 없다 |
| 레이아웃 | `dense-tp16` 플래그에 `--speculative-config '{"method": "eagle3", "model": <drafter dir>, "num_speculative_tokens": 3}'`를 더한다. drafter는 `skt/A.X-K2-EAGLE3` @ `24958e91`(`EAGLE3_SOURCE`)이다 |
| Job | `verify-remaining-nd96-hub.yml`, phase `dense-tp16-eagle3` |
| 시작 시간, KV cache | CUDA graph 시작이 `assert m.max_query_len <= self.reorder_batch_threshold`에서 실패했다. eager 재시도는 131 s 후 healthy가 되었고 KV 토큰은 367,904개였다. |
| 측정값(지점당 단일 샘플) | eager: 동시성 32에서 1K / 4K / 8K 입력 기준 total tok/s 324 / 595 / 643. 이는 CUDA graph를 사용한 `dense-tp16`의 0.19-0.23x다. step당 1.05 토큰을 수락했고, 기능 검사 4/4를 통과했다. |
| 주의 사항 | SM80의 유일한 MLA decode kernel인 `TRITON_MLA`는 요청당 토큰 하나만 검증한다. 따라서 draft는 prefill 경로로 되돌아간다. EAGLE3는 PP와 함께 실행할 수 없다(drafter가 레이어 2, 30, 58을 읽는다). native DSA와의 조합은 시도하지 않았다. |

### `native-pp2-bi`와 `dense-pp2-bi`: batch-invariant 모드 (설계 문서 8.1)

| 항목 | 값 |
| --- | --- |
| 상태 | A100에서 시작 불가 |
| 레이아웃 | `VLLM_BATCH_INVARIANT=1`을 사용한 TP8 x PP2, 컨텍스트 32,768, 시퀀스 32, native 모드와 dense 모드 |
| Job | `verify-remaining-nd96-hub.yml`, phase `native-pp2-bi`와 `dense-pp2-bi` |
| 결과 | 시작이 "No FP8 MoE backend supports the deployment configuration"으로 실패한다. batch-invariant를 지원하는 유일한 MoE kernel은 Triton MoE kernel이고, 그 FP8 경로에는 compute capability 8.9가 필요하다. 이 모드가 없으면 동일한 greedy 요청이 68-264자 이후에 서로 달라진다. |

### `nvfp4-tp8-b2048`: batched token 2,048개를 사용한 NVFP4 TP8 (설계 문서 8.4)

| 항목 | 값 |
| --- | --- |
| 상태 | 테스트한 대안 |
| 레이아웃 | `demo-nvfp4` 레이아웃(`--tensor-parallel-size 8 --distributed-executor-backend mp`, `--max-model-len -1`, 시퀀스 128, 0.92)에 `--max-num-batched-tokens 2048`을 쓰고 priority 스케줄링을 쓰지 않는다 |
| Job | `nvfp4-a100-nd96.yml`, phase `nvfp4-tp8-b2048`. 같은 할당에서 phase `nvfp4-tp8`(batched token 8,192) 다음에 실행한다(`FULL_REPO=skt/A.X-K2-NVFP4`, `NVFP4_SM80_PORT=1`) |
| 시작 시간, KV cache | 352 s, 272,960 토큰(GPU당 19.91 GiB). 따라서 262,144 토큰 전체 컨텍스트가 들어간다. batched token 8,192개에서는 433 s, 254,016 토큰이다. |
| 측정값(지점당 단일 샘플) | 같은 할당의 `nvfp4-tp8` 대비, 설계 문서 8.4의 출력 256 토큰 지연 시간 계획 기준: output tok/s 0.97-1.07x(두 phase 모두 모든 지점에 같은 seed를 써서 prefix cache 재사용이 섞였다. 재측정 전까지 사용하지 않는다). 1K 요청 하나의 median TPOT는 19.0 ms 대 20.4 ms. 1K 입력, 동시성 128에서 output tok/s 655 대 673. |
| 주의 사항 | vLLM은 A100에서 기본으로 batched token 2,048개를 사용한다. 데모와 expert-parallel 비교는 tech report의 B200 설정인 8,192를 사용한다. 설계 문서 8.2의 이전 FP8 표는 2,048을 사용했다. |

### 2-layer cut: 정확성 리허설 (설계 문서 7, 8.2, 8.3)

| 항목 | 값 |
| --- | --- |
| 상태 | 검증 전용 |
| 레이아웃 | FP8 체크포인트를 바이트 단위로 정확히 잘라 낸 2-layer cut(`hf-smoke:<revision>`, `build_smoke_checkpoint.py`). 먼저 vLLM 기본 executor로 단일 노드 TP8을 native 모드와 dense 모드로 실행하고, 이어서 Ray 위의 TP8 x PP2를 native 모드로 실행한다. 컨텍스트 16,384, 시퀀스 16, 0.85, `--no-enable-prefix-caching`. |
| Job | `native-dsa-bench-nd96-hub.yml`(`RUN_KERNEL_TESTS=1`), phase `smoke-native-tp8`, `smoke-dense-tp8`, `smoke-native-tp8-pp2` |
| 측정값 | 공식 Transformers FP32 sparse reference 대비, 선택된 토큰의 log-probability 평균 절대 차이(`index_topk` 이내 / 초과): native TP8 0.0116 / 0.0111, native TP8 x PP2 0.0120 / 0.0115, dense TP8 0.0116 / 0.0123. 14,618개 위치 기준이다(8.2). PR #38476의 kernel test 94개를 통과했다. |
| 주의 사항 | 레이어 두 개의 결과는 효과 크기의 한계(bound)를 제시한다. 전체 모델은 FP32 CPU reference에서 실행할 수 없다. |

### 이 카탈로그에 없는 것

- `experiments/axk2_dist_*`의 수작업 block runtime은 vLLM 구성이 아니다. 수치
  oracle로만 보관한다(설계 문서 1).
- model-only API(설계 문서 8.6)는 데모 서버로 가는 웹 앱 경로이며, 별도의 vLLM
  구성이 아니다.
- 테스트하지 않은 것: 두 노드에 걸친 NVFP4(설계 문서 8.4), native DSA와 EAGLE3의
  조합(8.1), TP 없는 data-parallel attention(DP16 x TP1 + EP), EPLB와 DeepEP
  (8.7, 9).

## 구성 재현

**job이 스크립트를 받는 방법.** 템플릿은 다음 세 가지 방법 중 하나로
`aml/src`(`entry.sh`와 helper)를 전달한다.

| Placeholder | 템플릿 | 실행 방법 |
| --- | --- | --- |
| `__AXK2_SRC_URL__` | `demo-fp8-nd96.yml`, `demo-nvfp4-nd96.yml` | 웹 앱의 supervisor가 제출마다 새 link token으로 이 템플릿을 렌더링하고 제출한다. job은 웹 앱에서 스크립트를 다운로드한다(`/api/link/src`, `AXK2_SRC_SHA256`으로 검사). 직접 시작하지 말고 supervisor로 시작한다(운영 상태 참조). 데모 job은 웹 앱이 실행 중일 때만 시작할 수 있다. |
| `__AXK2_SRC_GIT__` | `ep-compare-fp8-nd96.yml`, `ep-compare-nvfp4-nd96.yml` | 먼저 `aml/src`를 commit하고 push한다. `render_job.py`는 push된 HEAD를 pin한다. job은 해당 commit 기준으로 GitHub에서 각 파일을 다운로드하고 SHA-256을 검사한다. `aml/src`가 commit되지 않았거나 push되지 않았으면 렌더링을 거부한다. |
| `__AXK2_SRC_B64__` | `verify-remaining-nd96-hub.yml`, `nvfp4-a100-nd96.yml`, `native-dsa-bench-nd96-hub.yml`, `full-native-dense-nd96-hub.yml`, `serve-tp8-pp2-nd96.yml` 등 | `aml/src`가 base64로 인코딩되어 환경 변수에 담겨 전달된다. 현재 96,292자로, `ArgumentTooLong`으로 실패했던 크기(약 91K, 설계 문서 6)보다 크다. 이 템플릿 중 하나를 다시 실행하기 전에 `ep-compare` 템플릿의 `__AXK2_SRC_GIT__` 방식으로 바꾼다(시도하지 않음). |

**체크포인트 pin.** 여전히 `hf:2287ca45...`를 지정하는 템플릿은 다운로드에
실패한다. Hub가 해당 revision에 대해 404를 반환하기 때문이다. 이를
`hf:4237c6a60cec6ae449854870d01a42665a1166aa`로 바꾼다. PR #35는
`demo-fp8-nd96.yml`에 대해 이 변경을 한다.

**job당 레이아웃 하나.** 이 카탈로그의 모든 `epc-*`, `dense-*`, `native-*`,
`nvfp4-*` phase는 `aml/src/entry.sh`의 `plan_phase`에 있는 case다. 이 중 하나를
실행하려면 해당 체크포인트와 노드 수에 맞는 템플릿을 복사하고, `PHASE_PLAN`을 phase
이름으로 설정하고, `FULL_MAX_MODEL_LEN`이나 `EAGLE3_SOURCE`처럼 그 phase에 나열된
변수를 설정한다. 두 노드 FP8 phase에는 `ep-compare-fp8-nd96.yml`을 사용한다.
단일 노드 NVFP4 phase에는 `ep-compare-nvfp4-nd96.yml`(`FULL_REPO`와
`NVFP4_SM80_PORT=1` 포함)을 사용한다. phase는 서빙하고, suite를 실행하고, MLflow에
보고하고, 서버를 멈춘다. job은 마지막 phase 뒤에 끝난다. `PHASE_PLAN=demo`
(`demo_phase`)만 job이 취소될 때까지 계속 서빙한다. `smoke-*` phase는 job의 첫 번째
인자가 smoke 모델을 지정할 때 실행된다. `full-*` phase는 `PHASE_PLAN`이 비어 있을 때
`MODES`에서 나온다.

**명령.** WSL이나 Linux에서 Azure ML CLI v2로 실행한다. 모든 호출에서 구독을
지정한다.

```bash
python3 aml/render_job.py aml/jobs/ep-compare-fp8-nd96.yml   # writes aml/jobs/.rendered/, which git ignores
az ml job create --subscription <subscription-id> -g <resource-group> -w <workspace> \
  -f aml/jobs/.rendered/ep-compare-fp8-nd96.yml --query name -o tsv
az ml job show --subscription <subscription-id> -g <resource-group> -w <workspace> -n <job-name> \
  --query status -o tsv
az ml job cancel --subscription <subscription-id> -g <resource-group> -w <workspace> -n <job-name>   # stops billing

# Results are MLflow tags, because the workspace storage is private (design doc 6).
export AXK2_MLFLOW_URI=$(az ml workspace show --subscription <subscription-id> -g <resource-group> \
  -n <workspace> --query mlflow_tracking_uri -o tsv | sed 's#^azureml://#https://#')
export MLFLOW_TRACKING_TOKEN=$(az account get-access-token --subscription <subscription-id> \
  --query accessToken -o tsv)
python3 aml/fetch_results.py show <job-name>   # needs mlflow and azureml-mlflow
```

두 `export` 줄은 CLI의 기본 계정이 다른 구독에 있을 때 중요하다. `azureml://` URI를
쓰면 MLflow 플러그인이 기본 계정으로 인증하며, 2026-10-21에는 403을 반환했다. job이
오류나 사용자 로그 없이 실패하면 run의 RunHistory 세부 정보를 읽는다(설계 문서 6).

**용량과 비용.**

- 클러스터는 최대 세 개의 Spot 노드를 가지며, 유휴 노드를 120초 후에 해제한다.
  두 데모 프로필을 모두 켜면 클러스터가 가득 차고, 제출한 실험은 queue에서
  대기한다.
- 2026-10-21에 AML은 데모가 비운 노드를 대기 중인 job에 넘기지 않았다. 그 노드를
  해제하고 새 노드를 할당했으며, 이 과정에 약 10분이 걸렸다.
- `aml/first_capacity_wins.sh`는 같은 job을 여러 리전의 queue에 넣고, 가장 먼저
  노드를 얻은 job을 유지한다(설계 문서 7).
- 비용 상한으로 `limits.timeout`을 설정한다. `ep-compare-fp8-nd96.yml`은 5시간,
  `ep-compare-nvfp4-nd96.yml`은 3시간이다.
- uksouth Spot 단가는 노드-시간당 USD 12.89다(설계 문서 8.1). 따라서 두 노드
  레이아웃은 시간당 약 USD 26이 든다. expert-parallel 비교는 job 시간 기준 7.5
  노드-시간을 사용했다(설계 문서 8.7). 실제 청구액은 확인하지 않았다.
