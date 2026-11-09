# H100 미만 Azure GPU에서 A.X-K2 실행

이 저장소는 표준 분산 추론 스택으로 Azure A100 하드웨어에서 **skt/A.X-K2**를
서빙하기 위한 연구 자료와 재현 가능한 배포 자산을 담는다.

**현재 설계:** [docs/SERVING_DESIGN.md](docs/SERVING_DESIGN.md). vLLM 엔진(SKT
fork) 하나가 8xA100-80GB 노드 두 대로 구성한 Ray 클러스터에서 실행된다. 각 노드
안에서는 NVLink로 tensor parallel 8을, 노드 사이에서는 pipeline parallel 2를
사용하며, 앞단에는 OpenAI 호환 엔드포인트 하나가 있다. 모델 고유의 DeepSeek Sparse
Attention은 대체 Triton 커널로 A100에서 실행된다. 이 커널은 upstream vLLM PR #38476
(`TRITON_MLA_SPARSE`)을 해시 검증하여 이식한 것이다. 약 60K 토큰까지의 컨텍스트에서는
더 빠른 dense 모드도 사용할 수 있다. 참조 단일 노드 배포와의 모든 차이는 설계 문서
5c절에 나열되어 있다. 8.1절은 속도를 기술 보고서의 B200 측정값과 비교한다.
SKT의 4-bit NVFP4 checkpoint도 upstream vLLM 변경을 backport한 뒤 이런 노드 **한 대**에서
실행된다(8.4절). 용량은 별도의 Azure ML low-priority 할당량(리전당 300 vCPU)에서
확보한다. 이 할당량의 노드는 2026-03-31부터 Spot으로 할당·과금된다. 배포 자산은 `aml/`에 있다.

**서빙한 구성:** [docs/VLLM_CONFIGS.md](docs/VLLM_CONFIGS.md)는 지금까지 서빙한
모든 vLLM 구성을 나열한다. 데모 프로필 두 개, expert-parallel 레이아웃, 이전 변형이
여기에 포함된다. 각 구성마다 정확한 flag, job 템플릿, 컨텍스트, KV 용량, 시작 시간,
측정 비율을 제시하며, 문서 첫머리에서 데모의 운영 상태를 밝힌다.

## 실험 결과 한눈에 보기

모든 실험의 결과, 그림, 표는 아래 문서에 모아 두었다.

- [docs/deck/README.md](docs/deck/README.md): 발표 자료 묶음이다. 슬라이드별 구성안(20장 + 부록 3장),
  1920×1080 그림 26장, CSV 표 11개가 있다. 그림과 표는 `docs/deck/scripts/make_figures.py`가
  `evidence/`에서 다시 만든다.
- [docs/deck/02_실험결과_요약.md](docs/deck/02_실험결과_요약.md): 실험 9건의 날짜·리전·연결,
  병렬 구성 7종의 속도·용량 결과, 측정으로 확인한 사실, 해석(측정 아님), 한계를 정리했다.
- [docs/report/REPORT.md](docs/report/REPORT.md): 고객용 기술 보고서이다.
- [docs/report/REVIEW_2026-11-09.md](docs/report/REVIEW_2026-11-09.md): 2026-11-09 서빙 실험 검토 결과(판정, 문서 정정 내역, 재측정 계획)이다.
- [docs/SERVING_DESIGN.md](docs/SERVING_DESIGN.md): 설계와 실험별 상세 결과이다.

| 실험 | 날짜 | 핵심 결과 | 자세히 |
| --- | --- | --- | --- |
| FP8 TP8×PP2, native DSA, 노드 2대 | 2026-10-02~03 | 서빙 성공. 기능·도구·needle 통과 | 설계 문서 8장 |
| dense 모드 대 native | 2026-10-03 | dense가 10개 조건 모두 더 빨랐다(이탈리아 북부·TCP. 배수는 같은 seed로 prefix cache 재사용이 섞여 재측정 필요). 128K·256K에서 needle 0/3(영국 남부·InfiniBand) | 설계 문서 8.1-8.2, 발표 자료 18장 |
| 2층 log-prob 대 공식 구현 | 2026-10-02~03 | 2,048 위치 밖에서 dense만 오차가 커진다 | 설계 문서 8.2, 발표 자료 10장 |
| 검증 실행(InfiniBand, 도구, needle, Fig. 7, TP16, EAGLE3) | 2026-10-03 | IB 20.6 GB/s, TCP 0.35 GB/s. EAGLE3는 CUDA graph 시작 실패로 eager만 측정 | 설계 문서 8.1, 발표 자료 19장 |
| NVFP4 TP8, 노드 1대 | 2026-10-04(동작 확인), 2026-10-21(속도 비교) | GPU 8장으로 FP8 16장의 0.83~1.10배(같은 날 측정, c1·c32, 입력 1K/8K/32K, 할당은 다름) | 설계 문서 8.4, 8.7, 발표 자료 16장 |
| 정확도 부분 평가(AIME26, KoBALT, CLIcK, IFBench) | 2026-10-06 | 일부 문항만 평가(계획한 생성의 55%). KoBALT·IFBench는 공개값과 통계적으로 구별되지 않는다(공개값이 95% 구간 안). AIME26은 10문항뿐이라 판단할 수 없다. CLIcK은 85.5로 공개값 91.6보다 낮다 | 발표 자료 10장 |
| Foundry BYOM 모델 전용 API, 데모 웹앱 | 2026-10 | APIM → 웹앱 → 역방향 연결 → vLLM 경로 검증 | 설계 문서 8.6, `demo/` |
| Expert parallelism 7개 구성 | 2026-10-21 | EP만 켜면 5~12% 느리다(3쌍, 각 1회). 동시 32에서 세 입력 모두 빠른 구성은 DP2×TP8+EP16뿐이다(TP16은 1K·8K에서 ×1.04) | 설계 문서 8.7, 발표 자료 9·12~17장 |
| Hosted A.X 에이전트(오프라인 계약, MCP 위임) | 2026-10 | 실험 코드와 테스트 | `experiments/hosted_ax/` |

값은 조건마다 1회 측정이다. 실험마다 리전과 노드 간 연결(InfiniBand 또는 TCP)이 다르다.
2026-10-03·10-04 작업은 모든 측정점에 같은 seed를 써서 일부 처리량에 prefix cache 재사용이 섞였다.
그런 값은 재측정 전까지 사용하지 않는다.

## 현재 상태

| 근거 | 결과 | 증명하지 않는 것 |
| --- | --- | --- |
| **ND96amsr(8 x A100) 한 대에서 SKT의 NVFP4 checkpoint, TP8, native DSA(설계 문서 8.4)** | **기능 4/4, tool 9/9, needle 9/9(최대 244,898 토큰). 같은 날 측정(설계 문서 8.7, 2026-10-21)에서 FP8 A100 16장의 0.83-1.10x다(1K·8K 요청 1개 1.01-1.02x). 이 범위에는 KV 용량에 막힌 동시성 32의 8K·32K 지점이 포함되고, NVFP4는 GPU 메모리 0.92·최대 시퀀스 128(FP8은 0.90·64)로 실행했다. 이전 판의 교차 비교 값(1K 동시성 32까지 1.01-1.06x, 64-128 0.81-0.86x, 8K-32K 0.64-0.98x)과 Fig. 7 조건의 B200 대비 0.47-0.49x는 다른 날짜·리전·연결의 값과 맞댄 데다 같은 seed로 인한 prefix cache 재사용이 섞여 재측정 전까지 사용하지 않는다. 같은 날 D 기준으로는 1K에서 B200 노드의 0.47x(GPU당 약 절반)다. 출력 토큰당 비용 비교(38-56% 낮음)도 같은 이유로 사용하지 않는다** | expert는 W4A16으로 실행된다(A100에는 FP4 tensor core가 없다). vLLM 0.24 변경의 backport에 의존한다. KV cache는 254K-273K 토큰이다. 전체 품질 평가 |
| **검증 실행, 같은 16 x A100, 할당 한 번(설계 문서 8.1)** | **InfiniBand가 작동한다: 노드 간 20.6 GB/s이며, TCP로는 0.35이다. Tool calling 9/9(`--enable-auto-tool-choice` 필요). SKT의 needle 테스트 9/9(최대 252,728 토큰, native). 기술 보고서 Fig. 7 조건: 1K-8K 입력에서 B200 노드 한 대의 0.43-0.51x(2K-8K는 prefix cache 재사용으로 부풀었다. 2026-10-21 재측정 환산 8K 0.39x), 16K-120K에서 0.10-0.36x** | Batch-invariant 모드와 EAGLE3는 A100에서 사용할 수 없다 |
| **Expert parallelism, 모든 레이아웃에 같은 가중치와 벤치마크 사용(설계 문서 8.7)** | **`--enable-expert-parallel`은 스위치이다: EP 크기는 pipeline stage당 DP x TP이다. FP8 레이아웃 5개와 NVFP4 레이아웃 2개가 코드 변경 없이 native DSA를 서빙하며, 기능 4/4, tool 9/9이다. EP만으로는 처리량이 동시성 32에서 5-8%, 요청 1개에서 7-12% 감소한다. TP16 + EP는 동시성 32에서 TP8 x PP2의 0.83-0.98x이고, 단일 요청 TPOT는 1.4x, KV cache는 0.46x이다. 부하 상황에서 세 입력 모두 더 빠른 것은 DP2 x TP8 + EP뿐이다(출력 1.12-1.36x, EP 없는 TP16은 1K·8K에서 1.04x). 대신 단일 요청 TPOT가 약 2x이고 컨텍스트는 238,656 토큰이다** | 측정점마다 샘플이 하나이다. EPLB, round-robin expert 배치, DeepEP(이미지에 A100 빌드가 없음)는 테스트하지 않았다. 입력은 최대 32K이다 |
| **전체 688B, native DSA, 2 x ND96amsr(16 x A100 80GB), TP8 x PP2, 엔드포인트 하나** | **서빙 및 측정 완료**: 기능 6/6, reasoning, needle 12/12(최대 60K 토큰). 요청 1개에서 출력 약 49 tok/s(1K 입력), 1K-32K 컨텍스트에서 TPOT 19-21 ms(2026-10-03, 이탈리아 북부, 노드 간 TCP, 출력 256). 동시성 8-128의 처리량(최대 812 tok/s)은 같은 seed로 인한 prefix cache 재사용이 섞여 재측정 전까지 사용하지 않는다 | 전체 품질 평가, 운영 SLA, 비트 단위 결정성 |
| 같은 할당, dense 모드 | 기능 결과는 같다. 10개 조건 모두 더 빨랐다(배수는 같은 seed로 인한 cache 재사용이 섞여 재측정 필요). 정확한 결과는 컨텍스트 2,048 토큰까지만 보장된다 | 장문 컨텍스트 검색: 126K와 253K 토큰에서 needle을 하나도 찾지 못했다(8.1) |
| 실제 가중치 2-layer 절단본 대 공식 Transformers FP32 sparse, 14,618개 위치 | native는 `index_topk`를 넘어서도 BF16 하한에 머문다(내부 0.0116 대비 0.0111). dense는 벗어난다(0.0123, 최대 0.23) | 전체 깊이 logit 동등성 |
| A100에서 PR #38476 Triton 커널 | 테스트 94/94 통과 | PR의 upstream 채택 |
| Dense 모드 == `index_topk` 이내의 DSA(CPU, 축소판 AXK2) | 통과: prefill과 cached decode가 일치한다. 차이는 `index_topk`를 넘어서만 발생한다 | 2048 토큰을 넘는 구간의 품질 |
| 기본 vLLM 0.23.0 위의 fork overlay | 수정된 기본 파일 21개가 배포된 wheel과 바이트 단위로 같다. 추가로 Ampere MLA dtype 수정이 있다 | 다른 GPU 세대 |
| A100 VM 두 대에서 이전에 직접 작성한 블록 런타임 | parity 사례 7개 통과 | 서빙. vLLM 설계로 대체되었다 |

모든 실행에서 만든 Azure 리소스는 전부 삭제했다. 이력과 범위는
[인수인계 문서](docs/HANDOFF.md)를 본다.

## 저장소 구성

- `aml/`: 표준 스택의 Azure ML 배포이다. 리전 설정, job 템플릿, 노드별
  launcher(`src/entry.sh`), fork overlay, native-DSA 이식본
  (`src/dsa_port.json`), NVFP4 capability 이식본
  (`src/nvfp4_sm80_port.json`), dense 모드 설정, client 테스트,
  공식 참조 job, 결과 reader, 로그/artifact 수집기와
  다중 리전 용량 watcher가 있다. `docs/SERVING_DESIGN.md`를 본다.
- `experiments/`: 직접 작성한 모든 검증 스크립트, 참조 stage 코드,
  과거 Azure runner/provisioning/cleanup 스크립트, setup 스크립트가 있다.
- `evidence/`: 실패를 포함하여 선별한 결과와 진단을 고정해 둔다.
  Subscription ID는 가렸다. 수치 결과와 모델 revision은 남겨 두었다.
- `docs/deck/`: 발표 자료 묶음(구성안, 결과 요약, 그림, CSV 표, 그림 생성 스크립트)이다.
- `docs/report/`: 고객용 기술 보고서와 그림이다.
- `demo/`: 데모 웹앱, Foundry BYOM 게이트웨이, 인프라(Terraform) 자산이다.
- `experiments/hosted_ax/`: Hosted A.X 에이전트 실험 코드이다.
- `docs/HANDOFF.md`: 정확한 범위, 미해결 작업, 클라우드 안전, 이전(migration) 메모를 담는다.
- `tests/test_bundle.py`: 구문, evidence 일관성, 클라우드 비활성화 여부를 검사한다.

모델 가중치, 임시 SSH 키, 자격 증명, 원본 클라우드 로그, 내려받은
제3자 소스 트리는 포함하지 **않는다**.

## 새 환경에서 CPU 사전 점검 재현

Linux 또는 WSL2, Python 3.12, Git, CPU 빌드 PyTorch를 사용한다. 네이티브 Windows는
분산 지원 방식이 다르다. 기록된 실행은 Linux Gloo 프로세스를 사용했다.
고정된 의존성을 설치하려면 인터넷 접속이 필요하다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/validate_native_pipeline.py
.venv/bin/python experiments/validate_axk2_request_pipeline.py
.venv/bin/python experiments/diagnose_axk2_chunking.py
```

`uv`가 이미 설치되어 있으면 다음 방법도 쓸 수 있다.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-cpu.txt
```

첫 번째 pipeline 테스트는 직렬 실행과 중첩 실행을 비교한다.
AXK2 테스트는 요청별 cache, padding, 생성 토큰, CPU 연산 중첩을 검사한다.
이 테스트는 실제 checkpoint가 아니라 **축소판 무작위 가중치**를 사용한다.
진단 스크립트는 실패한 chunked-prefill 대조군을 의도적으로 기록한다. 프로세스가
성공으로 종료되면 진단이 실행되었다는 뜻이며, chunking이 통과했다는 뜻이 **아니다**.

`experiments/validate_dense_equivalence.py`는 축소판 AXK2(CPU 전용)에서 dense 모드의
전제를 검사한다. 컨텍스트가 `index_topk` 안에 들어가는 모든 위치에서 dense attention은
DSA와 같아야 하며, 이를 전체 prefill과 cached decode 양쪽에서 확인한다.

## A100 배포 실행(Azure ML)

`ml` extension이 있는 Azure CLI가 필요하고, Linux/WSL에서 실행해야 하며, 비용 지출에 대한
명시적 승인이 필요하다. Spot ND96amsr 노드 두 대는 리전에 따라 시간당 대략 USD 17-26이
들며, Spot 가격에는 상한을 걸 수 없다.

```bash
export AZ=az SUB=<subscription-id> RG=<resource-group> OWNER_TAG=<you>
bash aml/setup_region.sh italynorth mlw-axk2-itn       # repeat per candidate region
python3 aml/render_job.py aml/jobs/native-dsa-bench-nd96-hub.yml aml/jobs/oracle-hf-hub-cpu.yml
az ml job create --subscription $SUB -g $RG -w mlw-axk2-itn \
  -f aml/jobs/.rendered/native-dsa-bench-nd96-hub.yml --query name -o tsv
bash aml/first_capacity_wins.sh mlw-axk2-itn=<job> mlw-axk2-uks=<job> ...
AXK2_MLFLOW_URI=$(az ml workspace show -g $RG -n mlw-axk2-itn --query mlflow_tracking_uri -o tsv) \
  python3 aml/fetch_results.py show <job>
python3 aml/fetch_results.py compare <job> <oracle job> --out compare.json   # 2-layer check
```

할당 한 번에서 먼저 Triton 커널 테스트를 실행한다. 이어서 실제 가중치 2-layer 절단본을
native DSA와 dense 모드, 단일 노드 TP8, Ray 기반 native TP8 x PP2로 실행한다.
그동안 전체 checkpoint는 stall watchdog 감시 아래에서 내려받는다. 그 다음 `MODES`에
나열된 각 모드로 전체 모델을 서빙하고 측정한다.
`full-native-dense-nd96-hub.yml`은 전체 모델 단계만 실행한다.
`verify-remaining-nd96-hub.yml`은 InfiniBand를 TCP와 비교 점검한 뒤, `PHASE_PLAN`에
나열된 서버 구성을 `aml/src/verify_suites.py`로 실행한다. 대상은
tool calling, 256K까지의 SKT needle 테스트, 결정성, 기술 보고서의
Fig. 7 sweep, TP16, EAGLE3이다. 모든 다중 노드 템플릿은 `IB_PROBE=1`을 설정한다.
따라서 점검으로 InfiniBand가 확인되면 NCCL은 InfiniBand를 사용한다.
`nvfp4-a100-nd96.yml`은 capability 이식본(`NVFP4_SM80_PORT=1`)으로 SKT의 NVFP4
checkpoint를 노드 한 대에서 서빙한다. watcher는 `NODES=1`로 실행한다.
`ep-compare-fp8-nd96.yml`(노드 두 대)과 `ep-compare-nvfp4-nd96.yml`(노드 한 대)은
expert-parallel 레이아웃을 하나의 매트릭스(`verify_suites.py`의 `epmatrix`)로 비교한다.
이 템플릿들은 이 저장소의 고정 commit에서 `aml/src`를 내려받고
각 파일의 SHA-256을 검사한다. 렌더링 전에 commit하고 push한다. `render_job.py`는
push하지 않은 commit을 거부한다.
작업이 끝나면 resource group을 삭제한다. 클러스터는 0으로 축소되지만, workspace, storage,
managed network는 삭제할 때까지 남는다. 삭제한 workspace 이름은 14일 동안
예약된 상태로 남는다(soft delete).

새 출력은 실험 스크립트 옆에 기록되며 Git이 무시한다.
commit된 결과는 `evidence/`에 변경 없이 남는다.

## GPU 실험

`setup_axk2_cuda.sh`는 원래의 **게스트 Linux** 설정을 기록한다. PyTorch
2.11.0 CUDA 12.8과 고정된 Transformers를 사용한다. 이 스크립트는 `/opt`와 `/tmp` 아래에 쓴다.
해당 경로를 검토하지 않고 공유 개발 호스트에서 실행하지 않는다.
기록된 GPU 환경은 `evidence/a100-gpu-package-freeze.json`을 본다.

`axk2_dist_weights.py`, `axk2_dist_prompts.py`, `axk2_dist_pipeline.py`는
과거의 **2-블록** 실험이다. 이 스크립트들은 `/opt/axk2-dist`와 `RANK` 환경 변수를
사용한다. 전체 모델이 아니라 선택한 checkpoint tensor만 내려받는다. 이 pipeline은
의도적으로 순차 실행하는 정확성 기준선이며, **운영용 병렬 서빙 구현이 아니다**.

`axk2_request_stage.py`는 native `torch.distributed.pipelining.ScheduleGPipe`와 함께
쓰는 더 새로운 요청 격리 stage이다. CPU driver는 요청 중첩과
autoregressive cache 처리를 시연한다. 실제 가중치를 쓰는 전체
GPU 통합은 아직 끝나지 않았다.

## 클라우드 안전

**어떤 스크립트도 계정을 자동으로 다시 연결하거나 예전 VM을 다시 만들지 않는다.**
과거 Azure 스크립트는 `AXK2_ENABLE_LEGACY_CLOUD=1`과 명시적인 `AZURE_SUBSCRIPTION_ID`가
모두 설정되지 않으면 즉시 실패한다.
이 스크립트들은 PATH에서 `az`를 찾거나 `AZURE_CLI`를 사용한다.

이 스크립트들에는 고정된 만료 시험 기한과 과거 리소스
이름/소유 태그가 그대로 남아 있다. 이 스크립트들은 감사 기록이며, **이식 가능한 단일 명령
배포가 아니다**. 새 시험을 승인받기 전에 리소스 이름, 네트워킹, 기한,
cleanup 보호 장치, 지출 한도를 검토하고 교체한다. 기한 보호 장치만 제거해서는 절대
안 된다. opt-in flag는 지출이나 삭제에 대한 승인이 아니다.
Azure 오케스트레이션은 명시적으로 인증된 Linux/WSL 환경에서 실행한다.

이 저장소에는 예약된 재시도가 없다. 기존 앱/세션 상태와
인증은 Git으로 옮겨지지 않는다.

## 고정된 upstream 소스

- [모델](https://huggingface.co/skt/A.X-K2/tree/4237c6a60cec6ae449854870d01a42665a1166aa)
- [Transformers](https://github.com/huggingface/transformers/tree/7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae)
- [SKT vLLM](https://github.com/SKT-AI/vllm/tree/023bc76025459d6356c5885f271b9b95c50f93b6)

원래의 제3자 소스와 checkpoint에는 각 upstream 라이선스가 그대로 적용된다.
이 저장소는 해당 소스 트리나 모델 가중치를 vendoring하지 않는다.
