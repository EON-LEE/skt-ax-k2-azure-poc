# 개발 환경 인계

스냅샷: 2026-10-06. GPU를 할당하기 전에 이 문서를 읽는다.

## 이중 모델 채팅 구현

승인된 UI는 여덟 개 샘플, composer/업로드, thinking, 실제 MAF action card와
artifact 컨트롤을 유지한다. 기본 비교 화면은 공유 질문 하나와 실제 응답 열 두 개로
이루어진 채팅 타임라인이다. 일반 채팅에는 모델 선택기만 추가된다.
탭을 바꿔도 상태는 유지되며 실행/취소/전원에 대한 부수 효과는 없다.
독립 snapshot과 공통 컨텍스트 snapshot은 실제 입력 조건을 공유한다. 이어 하기는
각 모델의 실제 기록/파일을 사용하고, 컨텍스트가 다르면 이를 명시적으로 표시한다.
Actor는 별도의 capability와 변경 불가능한 입력 snapshot을 가진다. 새 채팅은 둘 다
취소하고 권한을 회수한다. 만료되면 manager 항목뿐 아니라 보관된 참조도 지운다.
지표는 추정 토큰이나 GPU 직접 측정 속도가 아니라 실제 서버 lifecycle/usage다.

FP8은 여전히 root 프로필이며 기존 영구 상태를 유지한다. NVFP4는
`/models/nvfp4`에 있고 기본값은 꺼짐이다. 자체 supervisor/queue/store를 가지며,
`demo-nvfp4-nd96.yml`과 검증된 SM80 포트/MP executor를 사용해 UKSouth 노드 하나만
대상으로 한다. 사용자가 승인한 것은 기존 LowPriority compute 최대값 2 -> 3과 NVFP4용
추가 노드 하나뿐이다. 최소값/우선순위/FP8 job은 바꾸지 않아야 한다. benchmark job을
시작하거나, 권한/설정을 바꾸거나, FP8을 교체하지 않는다. 실제 준비 상태는 할당 후에
확인해야 한다. 승인된 용량은 사용 가능한 Spot 재고가 있다는 증거가 아니다.

Bootstrap archive는 결정적 gzip과 `/home/data/sources` 아래의 변경 불가능한
해시 주소 파일을 사용한다. 재배포 전에 실행 중인 레거시 job archive와 job/해시
바인딩을 정확히 보존한다. 현재 소스를 이전에 pin된 checksum으로 서빙하지 않는다.
프런트엔드 전용 배포는 이 파일들과 실행 중인 job 상태를 유지해야 한다. archive
매핑이 없으면 명시적으로 실패한다. 프로필 경로, 출처, 비교 모드, 컨텍스트/지표
정의와 기존 OCR/sandbox 제한은 `demo/README.md`를 참조한다.

## 2026-11-09: 서빙 실험 검토와 정정

검토자 5명이 서빙 실험(2026-10-02~10-21)과 문서를 읽기 전용으로 검토했다. Azure와 GPU는 쓰지 않았다.
결과는 [검토 보고서](report/REVIEW_2026-11-09.md)에 있다. 아래 날짜별 항목의 일부 수치는 그 결과에 따라
"**2026-11-09 정정:**"으로 고쳤다. 주요 내용은 다음과 같다.

- 2026-10-03·10-04 측정은 모든 측정점에 같은 seed를 써서 prefix cache 재사용이 섞였다. 일부 처리량,
  Fig. 7 비율(2K-8K), 토큰당 비용은 재측정 전까지 사용하지 않는다.
- low-priority 클러스터는 2026-03-31부터 Spot으로 할당·과금된다. 이전 low-priority 요율은 적용되지 않는다.
- DSA 포트의 Triton 인덱서 디코드 커널에서 잠재 버그 1건을 소스 분석으로 찾았다. GPU에서 재현하지 않았다.
  수정은 `aml/src/dsa_port.json`에 넣었다(패치 파일이 5개에서 6개가 됨, SERVING_DESIGN.md 5b절). CPU의 Triton
  interpreter로만 확인했고, CUDA 회귀 테스트는 아직 GPU에서 실행하지 않았다. 수정은 이 변경 이후의 실행부터
  적용된다. 지금까지 기록한 결과는 모두 수정 전 커널로 실행했고, 버그가 그 결과에 준 영향은 확인하지 않았다
  (처리량, 기능·도구 점검에서는 드러나지 않았다).

## 2026-10-07: AX Microsoft Agent Framework 데모

프런트엔드는 이제 기존 역방향 링크 위에서 Microsoft Agent Framework Python 1.20.0
`Agent`, function invocation layer, function middleware를 사용한다. thinking/tool
결정/코드를 포함한 추론은 A.X K2 혼자 수행한다.
최소 채팅 화면은 실제 pending/running/success/error 이벤트, 파일, diff와 요청 시
표시되는 sanitize된 HTML/SVG artifact를 렌더링한다. capability 격리, 만료, 업로드
제한과 지원하는 tool의 정확한 목록은 `demo/README.md`를 참조한다.

데스크톱 후속 작업은 공유되는 최대 1200px 채팅/composer 폭, 18px 본문 텍스트,
42px 주요 컨트롤을 사용한다. 이제 네이티브 PDF 읽기는 명시적인 페이지
chunk/인용과 typed outcome을 사용한다. 새로 첨부된 PDF는 분류하기 전에 반드시 실제로
파싱해야 한다. 이제 custom relay는 MAF의 `options.instructions`(이전에는 누락됨)를
전달하며, 여기에는 tool 안내와 정확한 workspace 목록이 포함된다. 비공개 PDF가
스캔본인지를 파일 이름이나 크기로 추론하지 않는다. OCR은 여전히 사용할 수 없다.

같은 token의 네이티브 대화/tool replay는 RAM에만 있고 크기가 제한된다. 장기
메모리는 구현되지 않았다. 새 채팅은 실제 task를 취소하고 해당 workspace의 권한을
회수한다. 턴별 tools-off와 명확한 직접 검색/tool 금지는 등록 시점과 호출 시점에
강제된다. 실행 요청은 가짜 진행 표시가 아니라 제한된 A.X continuation을 사용한다.
새로 승인된 PDF는 실제 MAF `read_pdf`/middleware 첫 페이지 preflight를 먼저
사용하며, 여기에는 실제 실패 카드도 포함된다. 실패하거나 취소된 읽기는 확인됨으로
표시하지 않는다. 이 변경은 이전의 필수 PDF guard를 수정한다. 그 guard는 모델의
잘못된/누락된 호출 때문에 종료될 수 있었으며, 사용자의 비공개 문서에서 parser
실패가 입증된 것은 아니었다. 더 많은 페이지는 여전히 실제 읽기가 필요하다. 종료
상태는 이벤트 기반이다. stream이 끝난 뒤에는 어떤 작업도 계속되지 않는다.

Web IQ에는 실제 표준 MCP adapter가 있다. 이 adapter는 input-schema 탐색과
직접 `x-apikey` 인증을 지원하며, 기존에 승인된 `EON-LEE/tmap-webiq-poc` 선행
작업을 기준으로 검증했다. Foundry 추론은 호출하지 않는다.
일반 공개 질의를 지원한다. 업로드된 콘텐츠와 파일 이름은 자동으로 외부에 보낼 수
없다. 기존에 승인된 엔터프라이즈 key/access를 안전하게 구성한다
(`demo/README.md` 참조). connection을 만들거나 새 권한을 부여하지 않는다.
`/api/capabilities`는 실제 상태와 실제 질의가 성공했는지를 보고한다. 탐색 전에는
운영용 검색 tool을 등록하지 않는다.
프로토콜 fixture 테스트만으로는 실제 엔터프라이즈 접근을 입증할 수 없다.

원래의 로컬 Web IQ 입력은 2026-10-21에 worktree 밖으로 옮겼다.
이제 운영 도구는 WSL 비공개 파일
`~/.axk2-targets/<target>/private-credential-copies/webiq.env`
(디렉터리 `0700`, 파일 `0600`)를 읽는다. 서비스는 이 파일을 자동으로 로드하지 않는다.

Web IQ 활성화는 2026-10-07에 검증했다. 이때 사용자가 별도로 제공한 AX 전용 key
입력(`.env.webiq`, git에서 제외되고 번들에 절대 포함되지 않음)과, 승인된 메모리 전용
방식으로 보안 App Service 설정에 옮기는 절차를 사용했다. 실제 MCP
initialize/discovery와 제한된 passage 질의는 Microsoft Learn 출처를 반환한다.
기존 A.X job은 실제 MAF를 통해 `web_iq_search`를 실행했고, thinking을 stream으로
전달했으며, 실제로 반환된 URL을 인용해 답변을 종합했다.
`/api/capabilities`는 `ready`와 `verified_search: true`를 보고했다.
독립 브라우저에서 실제 검색 카드와 반환된 출처 링크를 확인했다. 이후의 공개 검색
확장은 원래의 다섯 주제 제한을 없애고, 실제로 광고된 MCP vertical 열 개를 모두
네이티브 MAF tool로 등록한다:
web/news/finance/places/autosuggest/browse/images/videos/sports/sonic.
각 tool은 자체적으로 탐색된 schema와 제한된 옵션을 가진다. 공개 provider smoke
호출은 열 개 모두 응답을 반환했다. Samsung finance는 symbol 005930, KRW, LSEG와
거래 timestamp를 반환했지만 exchange/delay 필드는 반환하지 않았다. Autosuggest는
제안 목록을 반환하지 않았다. sports는 인용 URL 없이 경기 정보를 반환했다.
출처를 지어내지 않고, 실시간 피드라고 주장하지 않으며, 미디어 임베드 권리도 없다.
첨부 파일이 있으면 현재 턴에 명시적인 공개 검색어가 있어야 한다. 업로드에서
파생된 질의는 차단된다. Browse는 색인된 결과만 사용하며 public-domain/DNS/credential
URL 검사를 거친다. 실시간 crawl이나 동적 렌더링은 없다. OCR과 격리된 코드 실행은
여전히 사용할 수 없다. GPU 변경은 없다.
테스트에는 승인된 외부 격리 sandbox가 필요하다. 업로드되거나 생성된 코드는 App
Service나 GPU에서 실행하지 않으며, 테스트 통과를 지어내지 않는다.
사용자 승인 없이 리소스를 프로비저닝하거나 권한을 부여하지 않는다.

**사용자가 명시적으로 달리 요청할 때까지 GPU는 켜 둔다.** 프런트엔드 전용
배포는 `AXK2_FRONTEND_ONLY=1`을 사용하며 job/weight/클러스터 설정을 바꾸거나
benchmark를 시작해서는 안 된다. 아래의 이전 종료 안내는 더 이상 유효하지 않다.
기존 benchmark 샘플은 여전히 부분적이며 동등성을 입증할 수 없다.

안정성 릴리스가 안정된 뒤, 2026-10-07 07:03:40–07:04:05 UTC에 별도의 제한된 실제
MAF HTTP 측정을 수행했다. 새 대화를 사용했고, tools/thinking은 껐으며, 실제 prompt
token 980개와 output token 67–108개(상한 256)를 사용했다.
c1/c2/c4에서 각각 두 번씩 실행한 결과 aggregate output은 38.091/50.184/94.805
tok/s였다. 성공한 자연 완료는 14건이었고, 프런트엔드 queue는 관찰되지 않았으며,
실제 취소 control 한 건에서 Gate가 drain되었다. 이는 GPU 직접 benchmark 결과나
운영 용량이 아니다. REPORT.md 9.1절과 그 원시 증거 JSON을 참조한다.
GPU/job/리소스/설정 변경은 없다.

실행 중인 기존 A.X job에서 검증한 항목은 다음과 같다: 정확한 `2^64-1`과 서울
시각, README/HTML 읽기, HTML 쓰기 + unified diff + sanitize된 미리보기 + 다운로드,
CSV 합계 + SVG 차트, 원래 릴리스에서 구성되지 않은 서비스에 대한 실제 오류 결과.
thinking을 켠 파일 작업은 실제 `reasoning_content` stream을 보존했다.
독립 브라우저 탭에서 모델 세 라운드, 카드, 요청 시 미리보기, 첨부 파일 읽기와
원래의 390px 모바일 레이아웃(상시 편집기/관리자 UI 없음)을 확인했다.
후속 요구 사항은 데스크톱 전용이며, 추가 모바일 작업은 아니다.
이는 기능 데모 검사이며, 새 모델 품질 benchmark가 아니다.

## 2026-10-06: 고객 데모와 model card benchmark

**데모.** `demo/`는 App Service(Korea Central, B1)에서 상시 실행되는 채팅 페이지를
제공한다. 백엔드는 2 x ND96amsr_A100_v4 서빙 job(FP8, native DSA, TP8 x PP2)이다.
배포와 운영은 `demo/README.md`에, 스크린샷은 `REPORT.md` 9절에 있다.

- **UI:** ChatGPT 스타일의 채팅 페이지 하나("A.X K2 고객 데모")다. thinking은 접을
  수 있는 섹션으로, tool call은 카드로 보여 준다. 클러스터 화면이나 결과 화면은
  없다. 그 수치는 `docs/report`에만 있다.
- **공개 데모:** `AXK2_OPEN_DEMO=1`이 `deploy.sh`의 기본값이다. 따라서 내부 데모
  요청대로 URL을 아는 사람은 누구나 로그인 없이 채팅할 수 있다. 데모 비밀번호를
  요구하려면 `AXK2_OPEN_DEMO=0`으로 재배포한다. `/admin`에는 항상 관리자 비밀번호가
  필요하다. 비밀번호는 저장소가 아니라 `~/.axk2-demo/passwords`(WSL)에 있다.
- **GPU:** `demo/ctl.sh on|off|status`. `on`은 uksouth, italynorth, francecentral에서
  low-priority job을 경쟁시켜 노드 2개를 먼저 얻은 job을 유지하며, 약 25분 후에
  준비된다. low-priority 노드 2개의 비용은 시간당 약 USD 16이다. **2026-11-09 정정:** low-priority는
  2026-03-31부터 Spot으로 과금된다. uksouth Spot 기준 노드 2개는 시간당 약 USD 26이다. **`ctl.sh off`는
  사용자가 명시적으로 요청할 때만 실행한다.**
- **데모에서 측정한 값:** 첫 토큰 0.4 s, 사용자당 45-49 tok/s.
- **배포 주의점:** Oryx가 아직 빌드 중인데 `deploy.sh`가 "Kudu Status 502"로 끝날 수
  있다. 그 뒤 사이트가 `No module named uvicorn`으로 중단되면 빌드가 중간에 끊긴
  것이다. `deploy.sh`를 다시 실행한다.

**Benchmark** (`evidence/a100-benchmarks.json`, run `ev20261006-095528`). 이 결과는
공개 데이터와 채점 방식을 사용하며, model card의 thinking 모드 설정(temperature
0.6, top_p 0.95)을 따른다. 각 suite는 무작위 샘플로 실행했다. 요청에 따라 실행을 약
55% 지점에서 중단했으므로 점수는 참고용이다:

| Suite | SKT | A100 x16 | 완료/계획 | 95% CI |
|---|---:|---:|---:|---:|
| AIME26 | 97.1 | 100.0 | 10/60 생성 | 작은 n |
| KoBALT | 73.0 | 74.3 | 109/200 | ±8.2 |
| CLIcK | 91.6 | 85.5 | 131/200 | ±6.1 |
| IFBench | 75.9 | 78.6 | 112/200 | ±7.6 |
| NIAH | 100 | 100 (9/9, 32K-256K) | 검증 실행 | - |

**2026-11-09 정정:** AIME26은 완료한 10문항이 모두 정답이라 95% CI가 [100, 100]으로 붕괴해 공개값과 비교할
수 없다. KoBALT·IFBench는 공개값이 CI 안에 있어 차이를 구별할 수 없다는 뜻이며, 품질 동등성의 근거가 아니다.

`python docs/report/make_figures.py --only eval`은 차트와 HF card 표
(`docs/report/figures/benchmark-table.md`)를 다시 만든다.

남은 일:

- 전체 n으로 eval을 마치거나 다시 실행한다. 주목적은 CLIcK 결과를 확정하는 것이다.
- 보고서의 PPTX를 만든다. Office MCP tool에는 Microsoft 365 계정 연결이 필요하다.
  2026-10-06에는 연결되어 있지 않았다.
- 고객 데이터로 품질 검사를 실행한다.
## 2026-10-04: A100 노드 하나에서 SKT의 NVFP4 체크포인트

아래 2026-10-03 항목은 NVFP4를 A100에서 실행할 수 없다고 했다. 이는 틀렸다.
필요한 compute capability 8.9는 vLLM 0.23의 소프트웨어 검사이며, upstream vLLM은
v0.24.0에서 이를 8.0으로 낮췄다. 이 변경을 backport하여
(`aml/src/nvfp4_sm80_port.json`, opt-in `NVFP4_SM80_PORT=1`) SKT의
`skt/A.X-K2-NVFP4`를 uksouth의 ND96amsr **한 대**(8 x A100, TP8, native DSA)에서
서빙했다. job은 `frank_nail_jp9txc21y1`이다. 결과는 SERVING_DESIGN.md 8.4절에,
증거는 `evidence/a100-nvfp4-single-node.json`에 있다.

- **검사:** 기능 4/4, tool call 9/9, SKT의 needle test 244,898 토큰까지 9/9.
  weight는 GPU당 48.5 GiB를 차지한다. KV cache는 254,016 토큰을 담는다. batched
  token 2,048개에서는 272,960 토큰을 담으므로 262,144 토큰 전체 컨텍스트가
  들어간다.
- **Kernel:** A100에는 FP4 tensor core가 없으므로 NVFP4 expert는 Marlin을 통해
  W4A16으로 실행된다(B200은 W4A4로 실행한다). FP8 레이어와 attention은 이전과
  같이 실행된다.
- **속도:** 같은 날(2026-10-21) 측정에서 16개 A100의 FP8 체크포인트의 0.83-1.10x다(1K·8K 요청 1개
  1.01-1.02x). 아래 이전 기록은 교차 비교이고 같은 seed 측정이라 재측정 전까지 사용하지 않는다. 이전 기록:
  1K 입력, 동시성 32까지는 같은 output tok/s(1.01-1.06x), 동시성 64-128에서는 0.81-0.86x, prefill이 더 느린
  8K-32K 입력에서는 0.64-0.98x. tech report의 Fig. 7 조건에서는 1K-4K 입력에서 B200 노드 하나의 0.47-0.49x,
  8K-32K에서는 0.20-0.35x. 같은 날 D 기준 1K는 0.47x(889.8/1,900)로 GPU당 B200의 약 절반이다.
- **비용:** 이전 기록은 출력 토큰 백만 개당 두 노드의 FP8보다 38-56% 적다고 했다.
- **2026-11-09 정정:** 위 이전 기록의 속도 비율(1.01-1.06x, 0.81-0.86x, 0.64-0.98x)은 다른 날짜·리전·연결(2026-10-03
  italynorth, TCP)의 FP8 값과 맞댄 교차 비교다. FP8(2026-10-03)과 NVFP4(2026-10-04) 작업 모두 모든 측정점에 같은
  seed를 써서 prefix cache 재사용이 섞였다. 비용 비교 38-56%도 같은 이유로 재측정 전까지 사용하지 않는다.
- **제한:** backport에 의존하므로 운영 환경에서는 vLLM 0.24 이상을 사용한다. 품질은
  위의 probe로만 확인했다. KV cache는 두 FP8 노드의 673,792 토큰 중 38-41%를 담는다.

Azure 상태: 모든 리소스를 다시 삭제했다(리소스 그룹
`rg-axk2-nvfp4-208d24c1`, 15:51Z). 이름에 `-r4-`가 들어간 workspace는 이제 14일
동안 soft-delete 상태다. 이 실행은 최대 약 2.4 노드-시간, 약 USD 20-31을 사용했다.

## 2026-10-03 (세 번째 업데이트): 검증 실행과 tech report의 속도

남아 있던 모든 항목을 같은 2 x ND96amsr의 할당 하나에서 테스트했다
(uksouth, job `wheat_yak_fq7rn5533b`). 결과는 SERVING_DESIGN.md 8.1절에, 증거는
`evidence/a100-verification-and-doc-speed.json`에 있다.

- **InfiniBand:** Azure ML 컨테이너 안에서 GPUDirect RDMA와 함께 동작한다. GPU
  쌍당 20.6 GB/s로, TCP의 0.35 GB/s와 대비된다. 이제 모든 멀티 노드 템플릿이
  InfiniBand를 probe하고(`IB_PROBE=1`) NCCL이 확인하면 사용한다.
- **Tool calling:** 두 모드 모두 9/9. `--enable-auto-tool-choice`가 필요한데, model
  card 명령에는 이 옵션이 없다. 이 옵션이 없으면 `tool_choice: auto`가 원시
  `<tool_call>` 텍스트를 반환한다.
- **긴 컨텍스트:** SKT의 needle test는 native 모드에서 32K, 128K, 256K(최대
  252,728 토큰)에서 9/9로 통과했다. dense 모드는 32K에서 3/3을 통과했지만 128K와
  256K에서는 여섯 개 모두 실패했다. 따라서 dense 모드는 약 60K 토큰으로 제한된다.
- **결정성:** 기본 kernel은 68-264자 이후에 출력이 달라진다.
  batch-invariant 모드는 A100에서 시작할 수 없다. 이 모드를 지원하는 SM80용 FP8 MoE
  kernel이 없다.
- **tech report Fig. 7 대비 속도**(B200 노드 하나, 동시성 32, 출력 1K): 16개
  A100의 native는 1K-8K 입력에서 0.43-0.51x, 16K-120K에서 0.10-0.36x에 이른다.
  16K-120K에서는 KV cache(673,792 토큰)가 동시성을 제한한다. dense는 같은 범위에서
  0.52-0.64x와 0.23-0.58x에 이른다. **2026-11-09 정정:** 이 sweep은 모든 측정점에 같은 seed를 써서
  2K-8K 값이 prefix cache 재사용으로 부풀었다. 2026-10-21 재측정으로 환산하면 native 8K는 0.39x다.
- **EAGLE3:** 사용할 수 없다. PP와 함께 실행할 수 없다. TP16에서는 A100 MLA decode
  kernel이 다중 토큰 draft를 검증할 수 없다. CUDA graph로 시작하면 실패하고, eager
  모드는 4-5배 느리다(**2026-11-09 정정:** CUDA graph를 쓴 TP16 대비 4.3-5.3배).
- **NVFP4:** A100에서 불가능하다고 보고했다(compute capability 8.9 필요).
  **2026-10-04 정정:** 이는 소프트웨어 검사이며, 체크포인트는 A100 노드 하나에서
  실행된다(위 항목 참조).

Azure 상태: 모든 리소스를 다시 삭제했다(리소스 그룹
`rg-axk2-aml-208d24c1`). 이름에 `-r3-`가 들어간 workspace는 이제 14일 동안
soft-delete 상태다. 이 실행에는 약 USD 80-125가 들었다.

## 2026-10-03 (두 번째 업데이트): A100에서 native DSA

이제 모델은 공개된 weight와 config, 그리고 자체 DeepSeek Sparse Attention 규칙으로
A100에서 실행된다. attention은 대체 Triton kernel로 계산한다. 이는 upstream vLLM
PR #38476(`TRITON_MLA_SPARSE`와 Triton indexer logits)을 해시 검사와 함께 포팅한
것이며, SERVING_DESIGN.md 5b절과 `aml/src/dsa_port.json`에 설명되어 있다. **2026-11-09 정정:** 이 포트는
이제 인덱서 디코드 커널의 버그 수정을 포함해 파일 6개를 패치한다. 이 항목의 결과는 수정 전 커널로 얻었다. 5c절은
참조용 단일 노드 배포와의 모든 차이를 나열한다. 여기에는 tool-call parser 미설정과
60K를 넘는 컨텍스트 미검증이라는 공백도 포함된다.

- **Kernel test:** A100에서 94/94 통과.
- **2-layer 실제 weight cut:** 공식 FP32 sparse 모델 대비, native는 `index_topk`를
  넘어서도 BF16 대 FP32 오차 하한에 머문다. 반면 dense 모드는 오차가 커진다.
- **전체 688B 모델:** 2 x ND96amsr(TP8 x PP2)의 할당 하나에서 native와 dense로
  서빙하고 측정했다. 두 모드 모두 기능 검사 6/6을 통과했고 60K 토큰까지 needle을
  12/12 찾았다. native는 동시성 1-128에서 50-812 output tok/s를 냈고, 1K부터 32K
  컨텍스트까지 TPOT는 19-21 ms였다. dense는 10-60% 빨랐지만 2,048 토큰까지만
  정확하다. 자세한 내용은 8.2절과 `evidence/a100-native-*.json`에 있다.
  **2026-11-09 정정:** 이 측정은 노드 간 TCP였고, 같은 seed로 인한 prefix cache 재사용이 섞였다.
  동시성 8 이상의 처리량과 dense 비율(이 작업 기준 11-64%)은 재측정 전까지 사용하지 않는다.

Azure 상태: 모든 리소스를 다시 삭제했다(리소스 그룹
`rg-axk2-aml-208d24c1`, 02:09Z). 삭제된 workspace 이름은 14일 동안 예약된 상태로
남으므로, 재사용하려면 새 이름이 필요하다. 첫 native 실행의 Hub 다운로드는 68%에서
멈췄다. 이제 `stage_weights.py`에 재시작 watchdog이 있다. 이 라운드에는
`evidence/a100-native-dsa-deployment-log.json`에 기록된 대로 약 USD 71이 들었다.

남은 일: 전체 품질 평가(AA-LCR/RULER 또는 고객 데이터), 32K 토큰보다 긴
시퀀스가 둘 이상인 batch에 대한 upstream 수정 #49139, NCCL용 InfiniBand, 비트 단위
재현성, 그리고 고객 구독의 on-demand 용량.

## 2026-10-03 업데이트: 표준 스택 재설계

방법이 바뀌었다. [SERVING_DESIGN.md](SERVING_DESIGN.md)를 참조한다. 아래의
수작업 block runtime은 서빙 방식으로서 대체되었고, 수치 oracle로만 보관한다.
목표는 SKT 자체의 방식, 즉 SKT fork에서 `vllm serve`를 실행하고 표준 방식으로
확장하는 것이다. 구성은 Ray 클러스터, 각 8xA100 노드 안의 TP8, 두 노드에 걸친 PP2,
endpoint 하나다. 용량은 VM quota가 아니라 리전당 300 vCPU인 별도의 Azure ML
low-priority quota에서 얻는다.

결과: 전체 모델을 italynorth의 2 x ND96amsr_A100_v4에서 서빙했다
(TP8 x PP2, endpoint 하나). 기능, 추론, needle, 부하 테스트를 통과했다. 2-layer 실제
weight cut은 공식 Transformers 구현과 일치했다. SERVING_DESIGN.md 8절과
`evidence/a100-*.json`을 참조한다.

Azure 상태: 실행 후 **모든 리소스를 삭제했다**(리소스 그룹
`rg-axk2-aml-208d24c1`: 정책을 준수하는 workspace 12개, 694 GB staged 사본을
포함한 storage, Key Vault, private endpoint, 클러스터). 유료 리소스나 대기 중인 job은
남아 있지 않다. 다시 실행하려면 README의 "Run the A100 deployment" 절을 따른다.
Spot 용량을 찾아다녀야 한다고 예상한다. 2026-10-02에는 swedencentral, westus2,
francecentral, polandcentral에서 2 x ND96amsr를 할당할 수 없었다. italynorth와
uksouth에서는 할당되었다.

이 실행에 필요했던 수정 사항이며, 모두 현재 `aml/`에 있다:

- Storage 정책이 비공개, key 없는 storage를 강제한다. managed VNet, identity
  datastore, compute identity, 환경 변수 스크립트 payload, MLflow tag 결과를
  사용한다.
- vLLM 0.23 이미지에는 Ray가 들어 있지 않다. 먼저 `ray[cgraph]`를 설치한다.
- MLA chunked-context prefill의 Ampere FP8 Marlin 버그(int32 cast)는
  `apply_overlay.py`가 패치한다.

## 목표와 타협할 수 없는 구분

더 작은 대체 모델이 아니라 **실제 A.X-K2 체크포인트 전체**를 H100보다 낮은 GPU에서
실행한다. 고객에게는 사용 가능한 GPU에 걸친 병렬 추론이 필요하며, 여기에는 이종
T4/A100 fleet 가능성도 포함된다. VM을 차례로 거쳐 hidden activation을 전달하는 것만으로는
유용한 동시 서빙의 증거가 되지 않는다.

필요한 최종 증거:

1. 사전 학습된 block 61개, embedding, 최종 normalization, LM head를 모두 로드한다.
2. 실제로 생성된 응답과 명시적인 요청/cache 정확성 검사.
3. 서로 다른 GPU에서 계산이 겹치는 것을 보여 주는 GPU 실행 trace.
4. 첫 토큰/토큰 간 지연 시간, 처리량, GPU 메모리, 전송 지연.
5. 제한된 시간과 정리 절차를 갖춘 최소 클러스터의 all-or-nothing 확보.

2026-10-03 기준으로 vLLM 배포는 native DSA와 dense 모드 모두에서 1, 2, 4, 5번
항목을 충족한다. 3번 항목은 두 pipeline stage에 걸친 동시 처리량으로 암시되지만,
명시적인 kernel overlap trace는 수집하지 않았다.
전송 지연은 측정하지 않았다.

## 검증한 내용

### 단일 A100

pin된 SKT vLLM sparse 경로는 SM80을 거부한다. Sparse MLA backend guard와
DeepGEMM indexer는 서로 독립적인 차단 요인이다. guard를 제거하는 것은 해결책이 아니다.
`evidence/axk2-source-results.json`을 참조한다.

정확성 우선의 custom Transformers 경로를 실제 A100 80GB에서 실행했다.
FP32로 계산하고, weight는 더 낮은 정밀도로 저장하며, projection은 필요할 때
확장했다. 선택한 full-width/random 두 block 사례와 실제 체크포인트 expert 하나가
통과했다. 이전의 random 모델 8K 사례를 포함한 엄격한 실패 기록은 보존한다.
`evidence/`의 `axk2-a100-resolution.json`과 `axk2_*_trials-results.json` 파일을
참조한다.

### 두 VM에 걸친 실제 체크포인트 block 두 개

block 0은 rank 0에서, 256개 expert를 모두 포함한 block 1은 rank 1에서 실행했다.
rank 1은 **직렬 reference만을 위해** block 0의 사본을 추가로 가졌다.
256/4K/8K random/이중 언어 사례 일곱 개에서 분산 대 직렬 prefill/decode 오차가
0이었다. 그 사례들에서 full 대 cache 대조 검사도 통과했다.

이 driver는 activation 전송마다 acknowledgment를 기다린다.
이는 **순차적 sharded 실행**이며, 겹치는 pipeline 서빙, tensor parallelism, expert
parallelism이 아니다. RDMA가 아니라 NCCL/TCP를 사용했다.
reference는 같은 custom loader/operator를 공유한다. 따라서 공식 H100 엔진과의
동등성을 독립적으로 입증하지 않는다.

필요했던 실제 체크포인트 loader 수정:

- 128x128 FP8 scaling에서 부분 가장자리 block을 처리한다. 원래의
  `kv_a_proj_with_mqa` 행렬은 576x7168이고 scale grid는 5x56이다.
- 초기화된 0이 아니라 원래의 `e_score_correction_bias` buffer를 로드한다.
- tensor 해시/shape, 완전한 소비, 남은 meta tensor가 없는지를 검사한다.

`evidence/axk2-distributed-resolution.json`과 `dist-rank*-result.json`을 참조한다.
과거 스크립트 해시는 저장소 패키징/줄 끝 정규화 이전의 원래 실행 artifact를
가리킨다. 현재 소스 파일의 바이트와 반드시 같지는 않다.

### CPU에서의 네이티브 overlapping scheduler

`validate_native_pipeline.py`: 로컬 프로세스 네 개, microbatch 여덟 개, `loss_fn=None`을
사용한 네이티브 PyTorch `ScheduleGPipe`. 직렬화된 대조군에서는 활성 stage가 하나로
관찰되었고, pipeline 대조군에서는 네 개로 관찰되었다. 출력 일치 검사를 통과했다.

`validate_axk2_request_pipeline.py`: 축소된 random **61-layer AXK2**,
길이가 다른 요청 여덟 개, 별도의 요청 cache, 최종 norm/head, greedy 출력 토큰 네 개.
CPU stage 네 개가 겹쳐 실행되었고 생성된 token ID가 공식 직렬 CPU 모델과
일치했다. 오래된 cache metadata는 거부된다.

이는 CPU 프로세스의 계산 구간이다. CUDA에서 stage의 wall-clock 구간은
**launch 구간일 뿐**이며, GPU kernel이 겹친다는 증거가 아니다.
고정된 요청 cohort는 운영용 continuous-batching scheduler가 아니다.
EOS 처리, 취소, 요청 admission, 장애 복구는 남은 작업이다.

### Chunked-prefill 실패

`diagnose_axk2_chunking.py`는 **수정하지 않은 공식 축소 CPU 모델 안에서**
whole-prefill 경로와 chunked-prefill 경로 사이의 불일치를 재현한다.
최대 오차는 약 0.1307이다. custom stage는 같은 chunk가 주어지면 공식 경로와
일치한다. 근본 원인은 밝혀지지 않았다.

메모리에 맞추려고 tolerance를 완화하거나 chunked prefill을 기본으로 켜지 않는다.
통과한 pipeline 테스트는 유효한 길이의 전체 prompt를 사용한다. 고정된 전송
padding은 cache 업데이트에서 제외된다.

## Azure 상태와 권한 경계

- 유료 단일/이중 A100 시험용 VM과 전용 그룹은 모두 삭제했다.
- 전체 모델용 GPU 클러스터는 만들지 않았다. 실행 중인 추론 endpoint는 없다.
- quota 재시도를 위한 앱 자동화는 남아 있지 않다.
- 원본 구독은 `Internal_2014-09-01`로 분류되었다. 이것이 내부 상태가 특정 실패의
  원인이라는 증거는 아니다.
- 후보 리전 일곱 곳은 확인한 범위에서 리전 Spot quota가 100 vCPU였고 일반
  A100/T4 family quota는 0이었다.
- Sweden `ND96amsr_A100_v4`는 SKU API에서 제한이 없는 것으로 보였지만, 물리적
  용량과 Spot 할당은 검증하지 않았다.
- Sweden Spot 192 vCPU 요청 두 건은 `RequestThrottled`(429)를 반환했다. 필수
  3600초 대기 후의 재시도도 포함된다.
- 수락된 요청 ID나 승인은 확인되지 않았다. 지원 escalation 증거는 준비했지만
  **지원 티켓으로 제출하지는 않았다**.

`quota-support-escalation.json`과 `regional-quota-diagnosis.json`을 참조한다.
commit된 증거의 구독 식별자는 모두 0인 UUID로 바꾸었다. 의도한 인증 구독을
명시적으로 사용한다. 환경의 기본값으로 계정을 추론하거나 전환하지 않는다.

사용자는 원래 세션에서 8-GPU A100 Spot VM 두 대의 제한된 Sweden 시험을
승인했다. 조건은 첫 할당부터 4시간, VM당 시간당 최대 USD12, compute USD96과
disk/network 비용이다. 이는 **과거의 범위**이며, 무제한이거나 자동으로 이전되는
배포 권한이 아니다. 이후의 T4/A100 혼합 fleet 논의는 fleet 프로비저닝이 아니라
조사를 요청한 것이다.

## 검토 중인 quota 증가 없는 대안

100 Spot vCPU pool은 리전 안의 GPU 유형 간에 공유된다. GPU family마다 100이 아니다.
리전별 pool을 합산해 리전 간 추론 클러스터로 계산하지 않는다.

| 구성 | vCPU | 명목 GPU 메모리 | 시스템 RAM |
| --- | ---: | ---: | ---: |
| 25 x NC4as_T4_v3 | 100 | 400 GB | 700 GiB |
| 19 x T4 + 1 x NC24ads_A100_v4 | 100 | 384 GB | 752 GiB |
| 13 x T4 + 2 x NC24ads_A100_v4 | 100 | 368 GB | 804 GiB |
| 4 x NC24ads_A100_v4 | 96 | 320 GB | 880 GiB |
| 1 x ND96amsr_A100_v4 | 96 | 640 GB | 1800 GiB |

원래 FP8 weight는 cache/workspace를 제외하고 약 656 GiB다.
이 명목 GPU 합계 중 어느 것도 모델이 들어간다는 것을 입증하지 않는다. host 메모리는
GPU 메모리와 자동으로 합쳐지지 않는다.

가능한 경로이며, 모두 end-to-end로 검증되지 않았다:

- host에 상주하는 FP8 weight와 제한된 GPU weight caching/prefetch를 사용하고, 실제
  요청 overlap을 활용하는 A100 멀티 VM pipeline.
- host offload를 사용하는 NVLink 연결 8-A100 노드 하나. 토폴로지는 더 쉽지만 멀티
  VM 증거는 **아니다**.
- 부하 인지 expert 배치, 통신/combine 스케줄링, host offload를 갖춘 이종 구성(T4는
  expert, A100은 그 밖의 작업). 이런 엔진은 아직 이 저장소에 없다. T4 SM75에는
  네이티브 BF16/FP8 연산이 없으므로 A100 호환성 증거를 그대로 옮길 수 없다.
- 추가 4-bit quantization은 별도의 정확도/kernel 프로젝트다. 무손실 저장 방식
  변경이 아니며, A.X-K2에서 지원이 알려진 구성도 아니다.
  이후 SKT가 `skt/A.X-K2-NVFP4`를 공개했다. SERVING_DESIGN.md 8.4절은 이를 A100
  노드 하나에서 서빙한다.

명시적인 기한 안에 완전한 최소 실행 가능 fleet을 확보한다. fleet을 구성할 수 없으면
부분 할당을 삭제한다. A100 용량을 기다리는 동안 많은 T4 VM을 무기한 보유하지 않는다.

## 유료 검증 전에 남은 작업

1. 합산 GPU 용량만이 아니라 기존 quota, 인터커넥트, 측정된 메모리를 바탕으로
   실현 가능한 토폴로지와 runtime을 선택한다.
2. 모든 expert weight를 동시에 확장하지 않고, 전체 체크포인트의 partition/다운로드/로딩과
   제한된 host/GPU 저장을 구현한다.
3. 실제 weight 실행을 요청별 KV 상태 및 네이티브 scheduler와 통합한다. 또는 선택한
   병렬화를 지원하는 검증된 대안과 통합한다.
4. EOS, admission, 전송 실패, profiling 동작을 완성한다.
5. 로컬 정확성 검사와 오류 경로 검사를 실행한다. 그다음 제한된 배포 범위를 승인받고
   최신 프로비저닝/정리 guard를 구현한다.
6. 실제 GPU overlap, 응답, 지연 시간, 처리량을 검증한다. 실패 기록을 보존한다. 더
   작은 모델이나 일부 block의 통과를 완료로 보고하지 않는다.

## 과거 도구

작성한 스크립트는 모두 `experiments/`에 있다. 클라우드 runner는 opt-in 방식이며
`cloud_context.py`로 비활성화되어 있다. 프로비저닝 스크립트에는 만료된 2026년 10월
2일 cutoff가 남아 있다. 검토된 대체 수단이 새 시간/비용 guard를 제공할 때까지 이
cutoff를 유지한다.
원격 guest는 Linux `/opt`와 `/tmp`를 전제로 한다. 이 경로는 host 사용자의 경로가
아니다.

source snapshot은 다시 가져올 수 있다. 지원하는 모드는
`python experiments/validate_axk2.py --help`로 확인한다.
이 도구는 pin된 공개 revision을 사용한다. CPU 테스트를 재현하려는 목적만으로 전체
체크포인트를 다운로드하지 않는다. 여러 과거 GPU 시험은 결과를 `/tmp`에 쓴다.
실행하기 전에 각 스크립트를 검사한다.

원래의 원시 세션 로그, 개인 키, 계정 cache, 다운로드한 weight는 의도적으로
제외했다. 선택한 수치 실패 증거는 보존한다.
Git 전송이 성공해도 Azure 인증, quota, 실행 중인 리소스, 앱 자동화, GPU 가용성
보장은 전송되지 않는다.
