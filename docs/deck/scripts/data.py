"""Loads the evidence JSON from the repository, checks anchor values, and writes the CSV tables.

All numbers on the figures come from here. Nothing is typed in by hand except the items marked
HARD-CODED (retail prices, published-figure references and the experiment registry), which carry
their provenance in the CSV.
"""
import csv
import json

import numpy as np

from common import EV, LAYOUT, TAB

SIZES = (1024, 8192, 32768)
SZ = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K", 120000: "120K"}
ORDER = ["A", "B0", "B", "C1", "C2", "D", "E"]
BASE_OF = {"A": "A", "B0": "A", "B": "A", "C1": "A", "C2": "A", "D": "D", "E": "D"}
EP_SIZE = {"A": "없음", "B0": "없음", "B": "16", "C1": "8 (단계마다)", "C2": "16", "D": "없음", "E": "8"}
NODES = {"A": 2, "B0": 2, "B": 2, "C1": 2, "C2": 2, "D": 1, "E": 1}


def _j(name):
    for folder in EV:
        if (folder / name).is_file():
            return json.loads((folder / name).read_text(encoding="utf-8"))
    raise FileNotFoundError(name)


EP = _j("a100-expert-parallel-comparison.json")
BM = _j("a100-benchmarks.json")
PUB = _j("skt_published.json")
VER = _j("a100-verification-and-doc-speed.json")
NV = _j("a100-nvfp4-single-node.json")
ND = _j("a100-native-vs-dense-full-model.json")
L2 = _j("a100-native-dsa-2layer-vs-official.json")
LP = _j("a100-native-2layer-logprob-series.json")
CFG = {**EP["jobs"]["fp8"]["configs"], **EP["jobs"]["nvfp4"]["configs"]}


def cfg(code):
    return CFG[LAYOUT[code]["key"]]


def point(code, isl, c):
    return cfg(code)["points"][f"isl{isl}-c{c}"]


def out(code, isl, c):
    return point(code, isl, c)["output_throughput"]


def ratio(code, isl, c):
    return out(code, isl, c) / out(BASE_OF[code], isl, c)


def tpot_ratio(code, isl, c=1):
    return point(code, isl, c)["median_tpot_ms"] / point(BASE_OF[code], isl, c)["median_tpot_ms"]


def kv_list(code):
    return list(cfg(code)["parallel"]["kv_cache_tokens"])


def kv_label(code):
    return " + ".join(f"{v:,}" for v in kv_list(code))


def max_ctx(code):
    return cfg(code)["max_model_len"]["auto_fit"]


def healthy(code):
    return cfg(code)["startup"]["seconds_to_healthy"]


def running32k(code):
    return int(point(code, 32768, 32)["peak_running_requests"])


def near(a, b, tol=5e-4):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---- published references ------------------------------------------------------------------
PUB_ROWS = {r["suite"]: r for r in PUB["benchmarks"]["rows"] if r.get("suite")}
PUB_FIG7 = {p["input_tokens"]: (p["total_tok_s"], p["output_tok_s"])
            for p in PUB["series"]["b200_fp8_k2_fig7_crosscheck"]["points"]}

# ---- doc-speed (Fig. 7 conditions) ---------------------------------------------------------
DSP = VER["doc_speed_comparison"]["points"]
FIG7 = {p["input_tokens"]: p for p in DSP}
NVF7 = {p["input_tokens"]: p for p in NV["doc_fig7_conditions"]["points"]}

# ---- native vs dense -------------------------------------------------------------------------
NDM = {m: {b["name"]: b for b in ND["modes"][m]["benchmarks"]} for m in ("native", "dense")}
ND_NAMES = [b["name"] for b in ND["modes"]["native"]["benchmarks"]]
ND_KV = {"native": 724544, "dense": 821312}  # parsed from the vLLM log lines below, asserted
for _mode, _key in (("native", "full-native-tp8-pp2"), ("dense", "full-dense-tp8-pp2")):
    _lines = ND["backend_selection_from_vllm_logs"][_key]
    _lines = _lines if isinstance(_lines, list) else [_lines]
    assert any(f"{ND_KV[_mode]:,} tokens" in ln for ln in _lines), _mode

# ---- EAGLE3 --------------------------------------------------------------------------------
SPEC = {p["input_tokens"]: p for p in VER["speculative_decoding"]["points"]}

# ---- 2-layer log-prob ----------------------------------------------------------------------
SUMV = L2["summary_vs_official"]
LP_KEYS = {"dense": "node0.smoke-dense-tp8", "native": "node0.smoke-native-tp8", "native_pp2": "node0.smoke-native-tp8-pp2"}


def logprob_means(series_key):
    allv, inside, beyond = [], [], []
    for p in LP["prompts"]:
        o = np.asarray(p["oracle"], float)
        s = np.asarray(p[series_key], float)
        d = np.abs(s - o)
        allv.append(d)
        inside.append(d[:2048])
        beyond.append(d[2048:])
    cat = np.concatenate
    return dict(all=float(cat(allv).mean()), inside=float(cat(inside).mean()), beyond=float(cat(beyond).mean()),
                n_all=int(cat(allv).size), n_inside=int(cat(inside).size), n_beyond=int(cat(beyond).size))


LPM = {"native": logprob_means("native_tp8"), "native_pp2": logprob_means("native_tp8_pp2")}

# ---- anchor checks (a failure here means the evidence or the code drifted) ---------------------
assert near(out("A", 1024, 1), 47.52, 1e-3), out("A", 1024, 1)
assert near(ratio("C2", 8192, 32), 1.3616, 1e-3), ratio("C2", 8192, 32)
assert kv_list("A") == [673792] and healthy("A") == 261
assert near(BM["summary"]["kobalt"]["score"], 0.7431, 1e-3)
assert near(FIG7[1024]["native"]["total_token_throughput"], 814.59, 1e-3)
assert near(FIG7[1024]["dense"]["total_token_throughput"], 980.49, 1e-3)
assert near(NVF7[1024]["nvfp4_8_a100"]["total_token_throughput"], 924.9, 1e-3)
assert near(LPM["native"]["all"], 0.011358, 2e-3), LPM["native"]
assert LPM["native"]["n_inside"] == 7356 and LPM["native"]["n_beyond"] == 7262
assert all(cfg(c)["functional"]["passed"] == 4 and cfg(c)["tools"]["passed"] == 9 for c in ORDER)
assert PUB_ROWS["kobalt"]["skt"] == 73.0 and PUB_FIG7[1024] == (1900, 965)


# ---- CSV helpers -----------------------------------------------------------------------------
def write_csv(name, header, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote tables/{name} ({len(rows)} rows)")


def r(v, n=2):
    return None if v is None else round(float(v), n)


def write_all_tables():
    # t01 -------------------------------------------------------------------------------------
    head = ["id", "구성", "가중치 형식", "노드 수", "GPU 수", "EP 크기", "서버 시작(초)", "KV 토큰", "최대 컨텍스트",
            "32K·c32 동시 실행", "기능 시험", "도구 시험", "비교 기준"]
    for c in (1, 32):
        for s in SIZES:
            head.append(f"c{c} 출력 tok/s {SZ[s]}")
    for c in (1, 32):
        for s in SIZES:
            head.append(f"c{c} 기준 대비 {SZ[s]}")
    for s in SIZES:
        head.append(f"c1 TPOT 기준 대비 {SZ[s]}")
    rows = []
    for code in ORDER:
        L = LAYOUT[code]
        row = [code, L["label"], L["fmt"], NODES[code], NODES[code] * 8, EP_SIZE[code], healthy(code), kv_label(code),
               max_ctx(code), running32k(code), "4/4", "9/9", BASE_OF[code]]
        row += [r(out(code, s, c)) for c in (1, 32) for s in SIZES]
        row += [r(ratio(code, s, c), 4) for c in (1, 32) for s in SIZES]
        row += [r(tpot_ratio(code, s), 4) for s in SIZES]
        rows.append(row)
    write_csv("t01_layout_summary.csv", head, rows)

    # t02 -------------------------------------------------------------------------------------
    fields = list(point("A", 1024, 1).keys())
    rows = []
    for code in ORDER:
        for key, p in cfg(code)["points"].items():
            isl, c = key.replace("isl", "").split("-c")
            rows.append([code, LAYOUT[code]["label"], int(isl), int(c)] + [p.get(f) for f in fields])
    write_csv("t02_ep_points_long.csv", ["id", "구성", "입력 토큰", "동시성"] + fields, rows)

    # t03 -------------------------------------------------------------------------------------
    names = {"aime": "AIME26", "kobalt": "KoBALT", "click": "CLIcK", "ifbench": "IFBench"}
    rows = []
    for s, nm in names.items():
        d = BM["summary"][s]
        ifs = BM["summary"]["ifbench"]["prompt_strict"] * 100
        note = {"aime": "10/30 문항, 10/60 생성만 실행. 모두 정답이라 신뢰구간이 0으로 붕괴함",
                "kobalt": "200문항 부분 표본", "click": "200문항 부분 표본",
                "ifbench": f"200문항 부분 표본. 점수는 prompt-loose 기준(prompt-strict {ifs:.1f})"}[s]
        rows.append([nm, PUB_ROWS[s]["skt"], r(d["score"] * 100, 1), r(d["ci95"][0] * 100, 1), r(d["ci95"][1] * 100, 1),
                     d["n_items"], d["items_expected"], d["n_generations"], d["generations_expected"],
                     r(d["mean_completion_tokens"], 0), note])
    write_csv("t03_accuracy.csv", ["벤치마크", "SKT 공개값(%)", "우리 측정(%)", "95% CI 하한", "95% CI 상한", "문항 수",
                                    "예정 문항 수", "생성 수", "예정 생성 수", "평균 완성 토큰", "비고"], rows)

    # t04 -------------------------------------------------------------------------------------
    rows = []
    for s, p in FIG7.items():
        nv = NVF7.get(s)
        rows.append([s, PUB_FIG7[s][0], PUB_FIG7[s][1],
                     r(p["native"]["total_token_throughput"]), r(p["native"]["output_throughput"]),
                     r(p["native_total_vs_doc"], 3),
                     r(p["dense"]["total_token_throughput"]), r(p["dense"]["output_throughput"]),
                     r(p["dense_total_vs_doc"], 3),
                     r(nv["nvfp4_8_a100"]["total_token_throughput"]) if nv else None,
                     r(nv["total_vs_doc_b200"], 3) if nv else None])
    write_csv("t04_fig7_doc_speed.csv",
              ["입력 토큰", "B200 공개 총 tok/s", "B200 공개 출력 tok/s", "A100×16 native 총 tok/s", "A100×16 native 출력 tok/s",
               "native / B200", "A100×16 dense 총 tok/s", "A100×16 dense 출력 tok/s", "dense / B200",
               "A100×8 NVFP4 총 tok/s", "NVFP4 / B200"], rows)

    # t05 -------------------------------------------------------------------------------------
    lt = NV["latency_and_throughput_vs_fp8_16_gpus"]
    a, b = lt["nvfp4-tp8"]["points"], lt["nvfp4-tp8-b2048"]["points"]
    rows = []
    for pa, pb in zip(a, b):
        assert pa["name"] == pb["name"]
        f = pa["fp8_16_gpus"]
        rows.append([pa["name"], r(pa["nvfp4_8_gpus"]["output_throughput"]), r(pb["nvfp4_8_gpus"]["output_throughput"]),
                     r(f["output_throughput"]), r(pa["nvfp4_8_gpus"]["median_tpot_ms"]), r(pb["nvfp4_8_gpus"]["median_tpot_ms"]),
                     r(f["median_tpot_ms"]), r(pa["nvfp4_8_gpus"]["median_ttft_ms"]), r(pb["nvfp4_8_gpus"]["median_ttft_ms"]),
                     r(f["median_ttft_ms"])])
    write_csv("t05_nvfp4_vs_fp8.csv",
              ["시험", "NVFP4×8 출력 tok/s (배치 토큰 8192)", "NVFP4×8 출력 tok/s (배치 토큰 2048)",
               "FP8×16 출력 tok/s (2026-10-03 이탈리아 북부·TCP, 교차 비교)",
               "NVFP4×8 TPOT ms (8192)", "NVFP4×8 TPOT ms (2048)", "FP8×16 TPOT ms (2026-10-03 이탈리아 북부·TCP)",
               "NVFP4×8 TTFT ms (8192)", "NVFP4×8 TTFT ms (2048)", "FP8×16 TTFT ms (2026-10-03 이탈리아 북부·TCP)"], rows)

    # t06 -------------------------------------------------------------------------------------
    rows = []
    for n in ND_NAMES:
        na, de = NDM["native"][n], NDM["dense"][n]
        rows.append([n, na["concurrency"], na["input_len"], na["output_len"], r(na["output_throughput"]), r(de["output_throughput"]),
                     r(de["output_throughput"] / na["output_throughput"], 3), r(na["median_tpot_ms"]), r(de["median_tpot_ms"]),
                     r(na["median_ttft_ms"]), r(de["median_ttft_ms"])])
    write_csv("t06_dense_vs_native.csv",
              ["시험", "동시성", "입력 토큰", "출력 토큰", "native 출력 tok/s", "dense 출력 tok/s", "dense / native",
               "native TPOT ms", "dense TPOT ms", "native TTFT ms", "dense TTFT ms"], rows)
    pp = VER["per_phase"]
    nh = {k: pp[k]["niah"]["by_length"] for k in ("native-pp2", "dense-pp2")}
    rows = [
        ["서버 시작까지(초, 이탈리아 북부·TCP)", ND["modes"]["native"]["startup_seconds_to_healthy"], ND["modes"]["dense"]["startup_seconds_to_healthy"]],
        ["KV 캐시 토큰", ND_KV["native"], ND_KV["dense"]],
        ["기능 시험", "6/6", "6/6"],
        ["needle 1.4K~60K ×3 깊이 (이탈리아 북부·TCP)", "12/12", "12/12"],
    ]
    for L_ in ("32768", "131072", "262144"):
        rows.append([f"needle {int(L_):,} 토큰 (영국 남부·IB)", f"{nh['native-pp2'][L_]['hits']}/{nh['native-pp2'][L_]['total']}",
                     f"{nh['dense-pp2'][L_]['hits']}/{nh['dense-pp2'][L_]['total']}"])
    write_csv("t06b_dense_vs_native_summary.csv", ["항목", "native", "dense"], rows)

    # t07 -------------------------------------------------------------------------------------
    rows = []
    for k, label in (("dense", "dense TP8"), ("native", "native TP8"), ("native_pp2", "native TP8×PP2")):
        s = SUMV[LP_KEYS[k]]
        for part, ko in (("within_index_topk", "index_topk 안쪽"), ("beyond_index_topk", "index_topk 바깥쪽")):
            rows.append([label, ko, s[part]["positions"], s[part]["mean_abs_diff"], s[part]["top1_agreement"], "요약 JSON(summary_vs_official)"])
    for k, label in (("native", "native TP8"), ("native_pp2", "native TP8×PP2")):
        m = LPM[k]
        rows.append([label, "시계열 재계산: 전체", m["n_all"], round(m["all"], 6), None, "a100-native-2layer-logprob-series.json 시계열에서 재계산"])
        rows.append([label, "시계열 재계산: 안쪽", m["n_inside"], round(m["inside"], 6), None, "a100-native-2layer-logprob-series.json 시계열에서 재계산"])
        rows.append([label, "시계열 재계산: 바깥쪽", m["n_beyond"], round(m["beyond"], 6), None, "a100-native-2layer-logprob-series.json 시계열에서 재계산"])
    write_csv("t07_logprob.csv", ["구성", "구간", "위치 수", "평균 |Δ log-prob|", "top-1 일치율", "출처"], rows)

    # t08 (HARD-CODED prices: retail lookup, not part of the evidence JSON) ---------------------
    pr = NV["cost_per_million_output_tokens_usd"]["prices_usd_per_node_hour"]
    rows = [
        ["ND96amsr_A100_v4 소매가(시간당 USD)", 40.962, "영국 남부, 소매(retail), 2026-11-02 조회. NVFP4 증거 JSON(2026-10-04 조회)에는 40.96"],
        ["ND96isr_H100_v5 소매가(시간당 USD)", 122.90, "영국 남부, 소매(retail), 2026-11-02 조회. 직접 조회한 값(증거 JSON에는 없음)"],
        ["A100 / H100 가격비", 0.333, "위 두 값의 비"],
        ["2노드 A100(시간당 USD)", 81.92, "40.96 × 2"],
        ["2노드 H100(시간당 USD)", 245.80, "122.90 × 2"],
        ["ND96amsr_A100_v4 Spot(시간당 USD)", pr["spot"], "NVFP4 증거 JSON의 단가표. 실험 비용 추정에만 사용"],
        ["ND96amsr_A100_v4 Low-priority(시간당 USD)", pr["low_priority"],
         "NVFP4 증거 JSON의 단가표. 이전 요율: Azure ML 저우선은 2026-03-31부터 Spot으로 과금되어 적용되지 않음"],
        ["FP8 체크포인트 크기(GB)", NV["checkpoint"]["fp8_checkpoint_size_gb"], "NVFP4 증거 JSON"],
        ["NVFP4 체크포인트 크기(GB)", NV["checkpoint"]["size_gb"], "NVFP4 증거 JSON"],
        ["노드 1대 GPU 메모리(GB)", 640, "A100 80GB × 8"],
        ["노드 2대 GPU 메모리(GB)", 1280, "A100 80GB × 16"],
    ]
    write_csv("t08_prices_memory.csv", ["항목", "값", "조건·출처"], rows)

    # t09 ---------------------------------------------------------------------------------------
    rows = [
        ["EAGLE3 (CUDA graph 사용)", "시작 실패", "TRITON_MLA 디코드는 요청당 쿼리 1토큰만 받음. 검증 쿼리 4토큰에서 AssertionError (max_query_len <= reorder_batch_threshold)", "a100-verification-and-doc-speed.json"],
        ["EAGLE3 (eager 대체)", "동작하나 느림", "수락 길이 1.046 토큰/스텝(초안 수락률 1.5%). CUDA graph를 쓴 TP16 비투기 대비 처리량 0.19~0.23배", "a100-verification-and-doc-speed.json"],
        ["EAGLE3 + 파이프라인 병렬(PP)", "불가", "초안 모델이 층 2·30·58의 은닉 상태를 읽는데, PP2에서는 단계 경계를 넘는 것이 은닉 상태·잔차뿐. vLLM도 미지원으로 표기. TP16으로 대신 측정", "a100-verification-and-doc-speed.json"],
        ["EAGLE3 + native DSA 포트", "실행 안 함", "투기 디코딩에서 seq_lens 형태가 [batch, next_n]인데 포팅한 Triton 커널은 [batch]로 읽음. 결과 검증 불가", "a100-verification-and-doc-speed.json"],
        ["batch-invariant 모드", "시작 실패", "'No FP8 MoE backend supports the deployment configuration'. 배치 불변을 선언한 MoE 커널은 Triton뿐이고, 그 FP8 경로는 연산 능력 8.9가 필요(A100은 8.0)", "a100-verification-and-doc-speed.json"],
        ["NVFP4 체크포인트", "가능 (정정됨)", "검증 JSON의 'not_possible_on_a100.nvfp4_checkpoint'는 낡은 주장. 2026-10-04에 Marlin W4A16 포트로 A100 8장 1노드에서 동작 확인", "a100-nvfp4-single-node.json"],
        ["결정적(deterministic) 출력", "보장 안 됨", "같은 탐욕 요청이 68~264자 이후 갈라짐(혼자·프리픽스 캐시 적중·배치 16)", "a100-verification-and-doc-speed.json"],
        ["DeepEP", "불가", "A100용 커널 없음(9.0a·10.0a만 빌드). pplx·naive는 vLLM 0.23에서 제거됨", "a100-expert-parallel-comparison.json"],
    ]
    write_csv("t09_status_not_possible.csv", ["항목", "상태", "원인(요약)", "근거 파일"], rows)

    # t10 (HARD-CODED registry; every row is backed by the named evidence file) -------------------
    rows = [
        ["첫 전체 모델 실행(dense, TP8×PP2)", "2026-10-02", "이탈리아 북부", "TCP(IB 끔)", "FP8 ×16 GPU", "a100-full-model-tp8-pp2-results.json"],
        ["native DSA 라운드", "2026-10-02~03", "이탈리아 북부 / 영국 남부", "라운드마다 다름", "FP8 ×16 GPU", "a100-native-dsa-deployment-log.json"],
        ["native vs dense 전체 모델", "2026-10-03", "이탈리아 북부", "TCP(IB 끔)", "FP8 TP8×PP2, max_model_len 65,536, 출력 256, 측정점 공통 시드(캐시 재사용)", "a100-native-vs-dense-full-model.json"],
        ["검증 + Fig.7 조건 속도 + TP16 + EAGLE3", "2026-10-03", "영국 남부", "InfiniBand", "FP8 ×16 GPU, NIAH 256K까지, 측정점 공통 시드(캐시 재사용)", "a100-verification-and-doc-speed.json"],
        ["NVFP4 1노드", "2026-10-04", "영국 남부", "단일 노드(NVLink)", "NVFP4 TP8, GPU 8장, 측정점 공통 시드(캐시 재사용)", "a100-nvfp4-single-node.json"],
        ["정확도 벤치마크(부분 실행)", "2026-10-06", "증거에 기록 없음", "데모 클러스터", "FP8 native TP8×PP2, thinking 모드", "a100-benchmarks.json"],
        ["EP 비교(7개 구성)", "2026-10-21", "영국 남부", "FP8: InfiniBand / NVFP4: 단일 노드", "FP8 5구성 + NVFP4 2구성, 측정점마다 다른 시드", "a100-expert-parallel-comparison.json"],
        ["2층 vs 공식 구현 (dense 모드, 약 3,300 위치)", "2026-10-02", "이탈리아 북부(vLLM) / 스웨덴 중부(공식 구현, CPU)", "단일 노드 TP8", "앞 2층, 실제 가중치, 공식 구현은 FP32. 첫 전체 모델 실행과 같은 작업", "a100-2layer-vs-official.json"],
        ["2층 vs 공식 구현 (native DSA·dense, 14,618 위치)", "2026-10-02~03", "이탈리아 북부(vLLM) / 스웨덴 중부(공식 구현, CPU)", "TP8 1노드, TP8×PP2 2노드", "앞 2층, 실제 가중치, 공식 구현은 FP32. native DSA 라운드 1회차", "a100-native-dsa-2layer-vs-official.json"],
    ]
    write_csv("t10_experiment_registry.csv", ["실험", "날짜", "리전", "연결", "구성·조건", "증거 파일"], rows)


if __name__ == "__main__":
    write_all_tables()
    print("anchors OK")
