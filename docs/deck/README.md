# A.X K2 A100 서빙 발표 자료

모든 서빙 패턴 실험을 PPT로 정리할 수 있게 구성안·그림·표를 한 폴더에 모았다. 실험 측정 기록은 저장소 루트 `evidence/`와 `docs/report/`에 있다.

## 폴더 구성

| 위치 | 내용 |
|---|---|
| `README.md` | 이 문서 |
| `01_슬라이드별_구성안.md` | 20장 + 부록 3장. 장마다 제목, 핵심 메시지, 넣을 글, 그림 파일, 발표자 노트, 주의, 근거 |
| `02_실험결과_요약.md` | 실험 9건 목록, 결과 표, 발견, 해석(측정 아님), 한계 |
| `figures/` | 슬라이드용 PNG 26장(1920×1080, 16:9) + 화면 캡처 3장(크기가 다르다) |
| `tables/` | CSV 11개. 그림의 원자료(차트를 새로 그릴 때 쓴다) |
| `scripts/` | 그림·표를 다시 만드는 파이썬 코드. 저장소 `evidence/`와 `docs/report/`를 읽는다 |

## 가장 빠르게 쓰는 방법

1. `01_슬라이드별_구성안.md`를 위에서부터 읽는다.
2. 장마다 빈 슬라이드에 `figures/`의 PNG를 꽉 차게 넣는다. 제목과 각주가 그림 안에 있다.
3. 발표자 노트에 구성안의 "발표자 노트"와 "주의"를 붙인다.
4. 3장은 `s03_architecture.png` 옆에 화면 캡처 2장을, 10장은 `s10_logprob-2layer-native.png`를 보조로 넣는다.
5. 부록 A1~A3은 그림이 없다. 구성안의 글과 표를 슬라이드에 옮긴다.

글자를 PPT 안에서 고치고 싶으면 PNG 대신 구성안의 "넣을 글"로 슬라이드를 만들고, `tables/`의 CSV로 차트를 새로 그린다.

## 그림 목록

| 장 | 파일 |
|---|---|
| 2 | `s02_price-and-memory.png` |
| 3 | `s03_architecture.png`, `s03_demo-chat-thinking.png`, `s03_demo-chat-tools.png` |
| 4 | `s04_kernel-swaps.png` |
| 5~8 | `s05_tp.png`, `s06_pp.png`, `s07_ep.png`, `s08_dp.png` |
| 9 | `s09a_summary-table.png`, `s09b_ratio-bars.png`, `s09c_kv-capacity.png` |
| 10 | `s10a_accuracy.png`, `s10b_logprob-2layer.png`, `s10_logprob-2layer-native.png` |
| 11 | `s11a_baseline-speed.png`, `s11b_fig7-b200-vs-a100.png` |
| 12~15 | `s12_tp16-vs-baseline.png`, `s13_tp16-ep16.png`, `s14_tp8pp2-ep8.png`, `s15_dp2tp8-ep16.png` |
| 16 | `s16a_nvfp4-vs-fp8.png`, `s16b_nvfp4-fig7.png` |
| 17 | `s17_nvfp4-ep8.png` |
| 18 | `s18a_dense-vs-native-speed.png`, `s18b_dense-needle.png` |
| 19 | `s19a_eagle3.png`, `s19b_status-table.png` |
| 20 | `s20_decision-table.png` |

## 꼭 알아둘 점

- **리전·연결이 실험마다 다르다.** 영국 남부 또는 이탈리아 북부, InfiniBand 또는 TCP다. 그림 각주와 구성안 A2 표를 지우지 않는다.
- **값은 조건당 1회 측정이다.** 반복 측정과 신뢰구간이 없다.
- **공식 구현이 아닌 포트로 돌렸다.** 병목 위치는 프로파일링하지 않았다.
- **정확도는 부분 실행이다.** 전체 모델 정확도나 공개 벤치마크 동등성을 말하지 않는다. 2층 log-prob은 앞 2층만 본 결과다.
- **Foundry 카탈로그 배포는 시험하지 않았다.** BYOM 호출 경로만 검증했다.
- **EAGLE3 비교는 두 요인이 섞여 있다.** 기준 TP16은 CUDA graph, EAGLE3는 eager다. "EAGLE3 자체가 느리다"고 말하지 않는다.
- **토큰당 비용은 비교하지 않는다.** 보고서 이전 판의 값은 서로 다른 작업의 처리량으로 계산했고 캐시 재사용이 섞여 철회했다.
- **2026-10-03·10-04 작업의 일부 값은 프리픽스 캐시 재사용으로 부풀었다.** 측정점마다 같은 시드를 썼기 때문이다. 해당 그림(s11b, s12, s16b, s18a, s20)의 글과 표에 표시했다. 재측정 전까지 사용하지 않는다.
- **측정하지 않은 해석은 "추정"으로 표시한다.** 목록은 `02_실험결과_요약.md` 4절이다.
- **검토 결과(2026-11-09)는 `../report/REVIEW_2026-11-09.md`에 있다.** 정정 내역과 재측정 계획(미실행)을 담았다.
- **그림은 `scripts/make_figures.py`로 다시 만들 수 있다.** 파이썬, matplotlib, numpy, Pillow와 맑은 고딕 글꼴이 필요하다. 실행하면 `figures/`와 `tables/`를 덮어쓴다.

## 공유 전에 확인할 것

- 저장소 루트 `evidence/`는 원본 측정 기록이다. 고객에게 주기 전에 포함 여부를 정한다.
- 슬라이드 그림에는 인증 정보와 내부 주소가 없다. 화면 캡처 3장도 확인했다.
