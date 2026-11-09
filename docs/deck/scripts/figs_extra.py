"""Slides 18-20: dense mode, EAGLE3 and batch-invariant status, and the decision table."""
import csv

from common import *
from common import _mark
import data as D
from figs_summary import trio, fmt_t, col_for, ep_pairs, KS, SIZES, GREY2


def rng(vals, f="{:.2f}"):
    return f"{f.format(min(vals))}~{f.format(max(vals))}"


def s18a():
    names = D.ND_NAMES
    nat = [D.NDM["native"][n] for n in names]
    den = [D.NDM["dense"][n] for n in names]
    ratio = [d["output_throughput"] / n["output_throughput"] for n, d in zip(nat, den)]
    assert len(names) == 10 and all(r > 1 for r in ratio)
    fig, ax = slide(
        "Dense는 10개 조건 모두 native보다 빨랐다 (배수는 재측정 필요)",
        "같은 클러스터에서 attention 방식만 바꿨다. 출력 tok/s, 이탈리아 북부·TCP, 10개 조건",
        "이탈리아 북부, TCP(InfiniBand 끔), 2026-10-03. 전체 688B 모델, FP8, TP8×PP2, max_model_len 65,536, 최대 128 seqs, 출력 256 토큰, 조건당 1회. "
        "dense = DSA 희소 선택을 건너뛰는 모드. 영국 남부 InfiniBand 결과와 절대값을 섞어 읽지 않는다. 출처: tables/t06_dense_vs_native.csv, tables/t06b_dense_vs_native_summary.csv",
    )
    assert all(n["concurrency"] == 1 for n in nat[:3])
    ticks = [f"{n['input_len'] // 1024}K" for n in nat[:3]] + [f"c{n['concurrency']}" for n in nat[3:]]
    assert ticks == ["1K", "8K", "32K", "c1", "c8", "c32", "c64", "c128", "c4", "c16"], ticks
    assert [n["input_len"] for n in nat] == [1024, 8192, 32768] + [1024] * 5 + [16384] * 2
    lx = legend_row(ax, [("native", NAVY), ("dense", SKY)], 4, 46.6, fs=12.5, gap=2.0)
    label(ax, lx, 46.6, "출력 tok/s (× = dense ÷ native)", 12.5, GREY, True)
    x0, w = 4.5, 61.0
    axc = panel(fig, x0, 20.0, w, 22.6)
    grouped_bars(axc, ticks, [("native", [n["output_throughput"] for n in nat], NAVY), ("dense", [d["output_throughput"] for d in den], SKY)],
                 fmt="{:,.0f}", ymax=max(d["output_throughput"] for d in den) * 1.32, fs_val=10, fs_tick=12.5,
                 ratios=[[None] * 10, ratio])
    for i0, i1, cap in ((0, 2, "한 요청 · 입력 길이"), (3, 7, "1K 입력 · 동시 요청"), (8, 9, "16K 입력 · 동시 요청")):
        xa, xb = x0 + (i0 + 0.1) / 10 * w, x0 + (i1 + 0.9) / 10 * w
        ax.plot([xa, xb], [18.0, 18.0], color=MID, lw=1.6, zorder=3)
        label(ax, (xa + xb) / 2, 16.6, cap, 12, GREY, ha="center")
    su = D.ND["modes"]["native"]["startup_seconds_to_healthy"], D.ND["modes"]["dense"]["startup_seconds_to_healthy"]
    card(ax, 3, 5.8, 62.5, 8.2, "10개 조건 모두 dense가 빨랐다 (재측정 전 참고값)",
         f"격차는 한 요청 1K {ratio[0]:.2f}배, 32K {ratio[2]:.2f}배, 16K·동시 16개 {ratio[9]:.2f}배다. "
         "측정점마다 같은 시드를 써서 캐시 재사용이 섞였으므로 이 비율은 재측정 전까지 쓰지 않는다. 정확도 손실(다음 슬라이드)과 같이 본다.",
         ec=VERM, hc=VERM, fs=11.5)

    st = D.ND["modes"]
    cols = [("항목", 12.5, "l"), ("native", 8.75, "c"), ("dense", 8.75, "c")]
    rows = [
        ["서버 시작", f"{su[0]} 초", f"{su[1]} 초"],
        ["KV 캐시 토큰", f"{D.ND_KV['native']:,}", f"{D.ND_KV['dense']:,}"],
        ["1K c1 TPOT", f"{nat[3]['median_tpot_ms']:.1f} ms", f"{den[3]['median_tpot_ms']:.1f} ms"],
        ["스트리밍 첫 토큰", f"{st['native']['streaming_single_user']['ttft_seconds']:.3f} s", f"{st['dense']['streaming_single_user']['ttft_seconds']:.3f} s"],
        ["기능 시험", f"{st['native']['summary']['functional_passed']}/{st['native']['summary']['functional_total']}",
         f"{st['dense']['summary']['functional_passed']}/{st['dense']['summary']['functional_total']}"],
    ]
    assert (round(nat[3]["median_tpot_ms"], 2), round(den[3]["median_tpot_ms"], 2)) == (19.23, 15.79)
    label(ax, 67, 45.2, "시작·메모리·첫 토큰", 14, INK, True)
    bottom = table(ax, 67, 43.4, cols, rows, fs=11, hfs=11, padx=0.7, pady=0.65)

    fig7 = [D.FIG7[s]["dense"]["total_token_throughput"] / D.FIG7[s]["native"]["total_token_throughput"] for s in sorted(D.FIG7)]
    assert 1.19 < min(fig7) < 1.21 and 2.36 < max(fig7) < 2.38, fig7
    assert fig7[0] == min(fig7) and fig7[-1] == max(fig7) and fig7 != sorted(fig7)
    card(ax, 67, 16.0, 30, bottom - 1.4 - 16.0, "영국 남부·IB의 Fig. 7 조건",
         f"동시 32개, 입력 1K~120K에서 dense÷native 총 tok/s는 {min(fig7):.2f}~{max(fig7):.2f}배다. 2K~8K는 같은 시드로 캐시 재사용이 섞였다.",
         ec=ORANGE, hc=VERM, fs=11.5)
    card(ax, 67, 5.8, 30, 9.0, "해석 (프로파일링하지 않음)",
         "attention의 희소 선택 단계를 건너뛰어 긴 입력에서 이득이 커 보이나, 시간 비중은 재지 않았다.", ec=GREY, hc=GREY, fs=11.5)
    save(fig, "s18a_dense-vs-native-speed.png")


def s18b():
    nd = {m: D.ND["modes"][m]["needle"] for m in ("native", "dense")}
    lens = sorted({n["target_tokens"] for n in nd["native"]})
    assert lens == [1400, 8000, 30000, 60000]
    low = {m: (sum(n["found"] for n in nd[m]), len(nd[m])) for m in nd}
    assert low["native"] == (12, 12) and low["dense"] == (12, 12)
    hi = {m: D.VER["per_phase"][k]["niah"]["by_length"] for m, k in (("native", "native-pp2"), ("dense", "dense-pp2"))}
    keys = ["32768", "131072", "262144"]
    assert all(hi["native"][k]["hits"] == 3 for k in keys)
    assert [hi["dense"][k]["hits"] for k in keys] == [3, 0, 0]
    fig, ax = slide(
        "Dense는 128K부터 needle을 못 찾는다 (0/3)",
        "문맥에 숨긴 코드를 찾는 needle 시험. native는 256K까지 통과, dense는 32K까지만 통과",
        "native·dense 모두 FP8 전체 모델, TP8×PP2, 2026-10-03. 60K 이하는 이탈리아 북부(TCP, max_model_len 65,536)의 12회, 32K 이상은 영국 남부(InfiniBand)의 별도 작업이다. "
        "두 작업의 설정이 달라 열마다 조건이 다르다. temperature 0, 출력 32토큰, thinking 끔. 출처: tables/t06b_dense_vs_native_summary.csv",
    )
    x0, tw, gap = 17.0, 19.0, 1.6
    cols = [("1.4K~60K", "이탈리아 북부·TCP"), ("32,768 토큰", "영국 남부·IB"), ("131,072 토큰", "영국 남부·IB"), ("262,144 토큰", "영국 남부·IB")]
    for i, (a, b) in enumerate(cols):
        cx = x0 + i * (tw + gap) + tw / 2
        label(ax, cx, 46.2, a, 14, INK, True, ha="center")
        label(ax, cx, 43.7, b, 11.5, GREY, ha="center")
    rows = [("native", 34.0, "native\n(TP8×PP2)"), ("dense", 24.3, "dense\n(TP8×PP2)")]
    for m, y, nm in rows:
        for j, ln in enumerate(nm.split("\n")):
            label(ax, 3.0, y + 5.2 - j * 2.4, ln, 14 if j == 0 else 11.5, INK if j == 0 else GREY, j == 0)
        cells = [low[m]] + [(hi[m][k]["hits"], hi[m][k]["total"]) for k in keys]
        for i, (h, t) in enumerate(cells):
            full = h == t
            x = x0 + i * (tw + gap)
            box(ax, x, y, tw, 8.6, "", fc=PALE_GREEN if full else PALE_RED, ec=GREEN if full else VERM, lw=2.0)
            label(ax, x + tw / 2, y + 5.3, f"{h}/{t}", 26, GREEN if full else VERM, True, ha="center")
            label(ax, x + tw / 2, y + 2.0, f"통과 / 시행 {t}회", 11, GREY, ha="center")
    card(ax, 3, 5.8, 46.0, 15.8, "측정한 것",
         "한 번에 한 요청, 깊이 3곳씩 넣은 needle 시험이다. native는 32K·128K·256K를 모두 통과했다(9/9). "
         "dense는 32K만 통과하고 128K·256K는 한 번도 찾지 못했다(0/6). 60K 이하는 두 모드가 모두 12/12다. "
         "60K와 128K 사이는 시험하지 않았다.", ec=NAVY, fs=12)
    card(ax, 51, 5.8, 46.5, 15.8, "해석 (측정 아님)",
         "dense는 top-2,048 토큰 선택을 건너뛴다. 문맥이 2,048 토큰 이하면 두 모드는 같은 결과를 낸다. "
         "모델이 선택 방식으로 학습돼 긴 문맥에서 어긋나는 것으로 보지만 원인은 검증하지 않았다. 실패가 시작되는 길이도 모른다. "
         "그래서 dense는 입력+출력이 약 60K 이하일 때만 쓸 만하다.", ec=GREY, hc=GREY, fs=12)
    save(fig, "s18b_dense-needle.png")


def s19a():
    sizes = sorted(D.SPEC)
    assert sizes == [1024, 4096, 8192]
    base = [D.SPEC[s]["tp16_dense"] for s in sizes]
    eg = [D.SPEC[s]["tp16_dense_eagle3"] for s in sizes]
    r = [e["total_token_throughput"] / b["total_token_throughput"] for b, e in zip(base, eg)]
    assert all(x < 0.25 for x in r) and 0.18 < min(r) < 0.2 and 0.22 < max(r) < 0.24, r
    doc = [D.SPEC[s]["doc_b200_eagle3_speedup_total"] for s in sizes]
    assert all(d > 1 for d in doc)
    e3 = D.VER["per_phase"]["dense-tp16-eagle3"]
    assert e3["eager_fallback"] == {"eager": True}
    assert "enforce_eager=True" in e3["engine_config"] and "enforce_eager=False" in D.VER["per_phase"]["dense-tp16"]["engine_config"]
    cnt = e3["specstats"]["counters"]
    drafts, dtok, acc = (cnt["vllm:spec_decode_num_drafts_total"], cnt["vllm:spec_decode_num_draft_tokens_total"],
                         cnt["vllm:spec_decode_num_accepted_tokens_total"])
    acc_len = 1 + acc / drafts
    assert abs(acc_len - 1.046) < 0.001 and abs(acc / dtok - 0.015) < 0.001
    fig, ax = slide(
        f"EAGLE3(eager): 일반 TP16의 {min(r):.2f}~{max(r):.2f}배",
        "기준 TP16은 CUDA graph 사용, EAGLE3는 eager(그래프 없음). 속도 차이에는 두 요인이 섞여 있다",
        "영국 남부, InfiniBand, 2026-10-03. dense 모드, TP16(노드 2개), FP8, max_model_len 32,768, 동시 32개, 출력 1,024 토큰, 조건당 1회. EAGLE3는 vLLM이 파이프라인 병렬과 같이 쓰기를 거부해 TP16으로 측정했다. "
        "B200 값은 SKT 기술보고서 Fig. 8 ÷ Fig. 7 막대 라벨. 출처: tables/t09_status_not_possible.csv, evidence/a100-verification-and-doc-speed.json",
    )
    legend_row(ax, [("TP16 일반", SKY), ("TP16 + EAGLE3 (eager)", VERM)], 4, 46.6, fs=12.5)
    label(ax, 4, 44.0, "총 처리량 (입력+출력 tok/s)", 14, INK, True)
    axc = panel(fig, 5.5, 22.8, 40, 19.6)
    grouped_bars(axc, ["1K", "4K", "8K"], [("TP16 일반", [b["total_token_throughput"] for b in base], SKY), ("EAGLE3", [e["total_token_throughput"] for e in eg], VERM)],
                 fmt="{:,.0f}", ymax=max(b["total_token_throughput"] for b in base) * 1.38, fs_val=11, fs_tick=13, ratios=[[None] * 3, r])
    cols = [("입력", 17.5, "l"), ("1K", 9.5, "c"), ("4K", 9.5, "c"), ("8K", 9.5, "c")]
    rows = [
        [("TP16 일반", {"b": True})] + [f"{b['total_token_throughput']:,.0f}" for b in base],
        [("+ EAGLE3 (eager)", {"b": True})] + [f"{e['total_token_throughput']:,.0f}" for e in eg],
        [("A100: EAGLE3 ÷ 일반", {"b": True})] + [(f"{x:.2f}", {"c": VERM, "b": True}) for x in r],
        [("B200 공개값 (×)", {"b": True})] + [(f"{x:.2f}", {"c": GREEN, "b": True}) for x in doc],
        [("TPOT 일반 → EAGLE3 (ms)", {"b": True})] + [f"{b['median_tpot_ms']:.0f} → {e['median_tpot_ms']:.0f}" for b, e in zip(base, eg)],
    ]
    label(ax, 51, 44.0, "A100과 B200 공개값 비교", 14, INK, True)
    table(ax, 51, 42.2, cols, rows, fs=11.5, hfs=11.5, padx=0.7, pady=0.65)
    card(ax, 3, 5.8, 46.0, 14.4, "CUDA graph로는 시작하지 못했다",
         "TRITON_MLA 디코드는 요청당 쿼리 1토큰만 받는다. EAGLE3 검증은 요청당 4토큰을 넣어, "
         "mla_attention.py:1827의 assertion(max_query_len <= reorder_batch_threshold)에서 멈췄다. 그래서 eager 모드로만 측정했다.", ec=VERM, fs=12)
    card(ax, 51, 5.8, 46.5, 14.4, "eager로 돌렸더니 수락이 거의 없다",
         f"초안 토큰 {dtok:,.0f}개 중 {acc:,.0f}개만 수락됐다(수락률 {100 * acc / dtok:.1f}%, 스텝당 {acc_len:.3f}토큰). "
         f"SKT는 B200에서 스텝당 2.24토큰을 보고했다. 수락이 낮은 원인은 규명하지 않았다.", ec=ORANGE, hc=VERM, fs=12)
    save(fig, "s19a_eagle3.png")


SHORT = {
    "EAGLE3 (CUDA graph 사용)": "디코드가 요청당 1토큰만 받는데 검증 쿼리는 4토큰이라 assertion 실패",
    "EAGLE3 (eager 대체)": "수락 길이 1.046 토큰/스텝. TP16 일반(CUDA graph) 대비 0.19~0.23배",
    "EAGLE3 + 파이프라인 병렬(PP)": "초안 모델이 중간 층 은닉 상태를 읽는데 PP2는 경계 너머로 일부만 넘긴다. vLLM도 미지원",
    "EAGLE3 + native DSA 포트": "투기 디코딩의 seq_lens 형태를 포팅한 Triton 커널이 읽지 못해 검증 불가",
    "batch-invariant 모드": "배치 불변 MoE 커널은 Triton뿐이고 그 FP8 경로는 연산 능력 8.9 필요(A100은 8.0)",
    "NVFP4 체크포인트": "검증 JSON의 '불가' 기록은 낡았다. 10-04 Marlin W4A16 포트로 A100 8장 1노드에서 동작",
    "결정적(deterministic) 출력": "같은 탐욕 요청이 68~264자 뒤 갈라졌다(단독, 프리픽스 캐시 적중, 배치 16)",
    "DeepEP": "A100용 커널이 없다(9.0a·10.0a만 빌드). pplx·naive는 vLLM 0.23에서 제거",
}
STATUS_COLOR = {"시작 실패": VERM, "동작하나 느림": ORANGE, "불가": VERM, "실행 안 함": GREY, "가능 (정정됨)": GREEN, "보장 안 됨": ORANGE}


def s19b():
    with open(TAB / "t09_status_not_possible.csv", encoding="utf-8-sig", newline="") as f:
        src = list(csv.DictReader(f))
    assert [r["항목"] for r in src] == list(SHORT), [r["항목"] for r in src]
    assert all(r["상태"] in STATUS_COLOR for r in src)
    fig, ax = slide(
        "A100 제약 7건과 정정 1건",
        "시작 실패·불가·제한 7건과 정정 1건(NVFP4). 원인은 로그와 코드 메시지에 근거한 사실이다",
        "영국 남부(2026-10-03), NVFP4 정정은 2026-10-04, DeepEP은 2026-10-21 작업. 상태·원인의 원문과 근거 파일은 tables/t09_status_not_possible.csv에 있다. "
        "이 표의 항목은 A100(연산 능력 8.0) 기준이며 H100·B200에서는 다를 수 있다.",
    )
    cols = [("항목", 25, "l"), ("상태", 13, "c"), ("원인 (요약)", 56, "l")]
    rows = [[(r["항목"], {"b": True}), (r["상태"], {"c": STATUS_COLOR[r["상태"]], "b": True}), SHORT[r["항목"]]] for r in src]
    bottom = table(ax, 3, 46.6, cols, rows, fs=11.5, hfs=12, padx=0.9, pady=1.15)
    label(ax, 3, bottom - 2.0, "탐욕(greedy) 요청의 출력이 갈라지므로, 같은 입력의 결과가 항상 같다고 약속할 수 없다.", 12, INK, max_w=94)
    save(fig, "s19b_status-table.png")


def s20():
    A = lambda s, c: D.point("A", s, c)["median_tpot_ms"]
    dD = lambda s, c: D.point("D", s, c)["median_tpot_ms"]
    a_tp = [A(s, 1) for s in SIZES]
    d_tp = [dD(s, 1) for s in SIZES]
    c2_c32 = [D.ratio("C2", s, 32) for s in SIZES]
    c2_tp = [D.tpot_ratio("C2", s) for s in SIZES]
    assert all(x > 1 for x in c2_c32) and all(x > 1.8 for x in c2_tp)
    d_c1 = [D.out("D", s, 1) / D.out("A", s, 1) for s in SIZES]
    d_c32 = [D.out("D", s, 32) / D.out("A", s, 32) for s in SIZES]
    ep = ep_pairs()
    assert len(ep) == 18 and all(x < 1 for x in ep)
    b0_c1 = [D.ratio("B0", s, 1) for s in SIZES]
    b0_c32 = [D.ratio("B0", s, 32) for s in SIZES]
    assert [x > 1 for x in b0_c32] == [True, True, False]
    kv_b0 = D.kv_list("B0")[0] / D.kv_list("A")[0]
    spec =  [D.SPEC[s]["tp16_dense_eagle3"]["total_token_throughput"] / D.SPEC[s]["tp16_dense"]["total_token_throughput"] for s in sorted(D.SPEC)]
    kva, kvd = D.kv_list("A")[0], D.kv_list("D")[0]
    fig, ax = slide(
        "정리: 목표에 따라 고르는 구성",
        "측정한 범위 안에서의 선택표다. 모든 값은 조건당 1회 측정이다",
        "FP8 구성과 NVFP4는 영국 남부 2026-10-21 작업(출력 1,024 토큰, c1 = 한 요청, c32 = 32개 동시, 입력 1K/8K/32K). dense 속도는 2026-10-03 이탈리아 북부·TCP, "
        "needle·EAGLE3는 2026-10-03 영국 남부·IB 작업. 배수는 기준 대비이며 FP8은 A(TP8×PP2), NVFP4는 D(NVFP4 TP8) 기준이다. 출처: tables/t01_layout_summary.csv, t06, t09",
    )
    cols = [("목표·상황", 20, "l"), ("고를 구성", 20, "l"), ("근거와 주의", 54, "l")]
    rows = [
        [("한 요청의 지연", {"b": True}), ("A TP8×PP2 또는 D NVFP4 TP8", {"b": True, "c": GREEN}),
         f"c1 TPOT {trio(a_tp, '{:.1f}')} ms(A), {trio(d_tp, '{:.1f}')} ms(D)."],
        [("긴 문맥, KV 용량", {"b": True}), ("A TP8×PP2", {"b": True, "c": GREEN}),
         f"KV {kva:,}토큰, 컨텍스트 262,144, 32K 동시 실행 {D.running32k('A')}개. D는 KV {kvd:,}토큰, 동시 {D.running32k('D')}개."],
        [("동시 요청이 많고 입력 32K 이하", {"b": True}), ("C2 DP2×TP8+EP16", {"b": True, "c": GREEN}),
         f"c32 출력 ×{trio(c2_c32)}(1회 측정, 조건 혼재). 단 c1 TPOT ×{trio(c2_tp)}, 컨텍스트 {D.max_ctx('C2'):,}."],
        [("GPU 8장만 쓸 때", {"b": True}), ("D NVFP4 TP8 (노드 1개)", {"b": True, "c": GREEN}),
         f"A 대비 c1 ×{trio(d_c1)}, c32 ×{trio(d_c32)}. 정확도 동등성은 기능·도구·needle만 확인."],
        [("EP만 켜기", {"b": True}), ("이득 없음(5~12% 느림)", {"b": True, "c": VERM}),
         f"18개 비교 모두 켠 쪽이 느리다(×{rng(ep)}, 3쌍 각 1회). round-robin 배치·EPLB는 시험하지 않았다(DeepEP은 A100 미지원)."],
        [("TP16 (노드 2개에 걸침)", {"b": True}), ("A와 비슷하거나 낮다", {"b": True, "c": VERM}),
         f"c1 출력 ×{trio(b0_c1)}, c32 ×{trio(b0_c32)}, KV ×{kv_b0:.2f}."],
        [("dense 모드", {"b": True}), ("입력+출력 약 60K 이하만", {"b": True, "c": ORANGE}),
         "10개 조건 모두 native보다 빨랐으나(배수는 재측정 필요), 128K·256K에서 needle 0/3. 60K~128K는 미시험."],
        [("EAGLE3, batch-invariant", {"b": True}), ("A100에서 이득 확인 못 함", {"b": True, "c": VERM}),
         f"EAGLE3(eager)는 CUDA graph 일반 TP16의 ×{rng(spec)}(4K·8K는 같은 시드 측정). batch-invariant는 시작 실패."],
    ]
    bottom = table(ax, 3, 46.8, cols, rows, fs=11.5, hfs=12, padx=0.8, pady=0.62)
    card(ax, 3, 4.6, 94, max(bottom - 1.2 - 4.6, 5.6), "다음 단계",
         "① 목표 지연·처리량, 입력 길이, 동시 요청 수를 먼저 합의한다.\n"
         "② 그 조건으로 같은 구성을 반복 측정한다(지금은 조건당 1회).\n"
         "③ 정확도는 전체 문항으로 다시 평가한다(지금은 일부 문항).",
         ec=NAVY, fs=12.5, hfs=13)
    save(fig, "s20_decision-table.png")


ALL = [s18a, s18b, s19a, s19b, s20]
