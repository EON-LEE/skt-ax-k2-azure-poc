"""Slides 9-11: experiment summary, accuracy, 2-layer log-prob, baseline speed, Fig. 7 comparison."""
from common import *
from common import _mark
import data as D

GREY2 = "#9AA5B1"
SIZES = D.SIZES
KS = [D.SZ[s] for s in SIZES]
SIZE_COLORS = [SKY, BLUE, NAVY]
# Points of the 2026-10-03/10-04 sweeps that reused earlier points' prompts from the prefix cache (shared seed).
TAINTED = (2048, 4096, 8192)


def trio(vals, f="{:.2f}"):
    return " / ".join(f.format(v) for v in vals)


def fmt_t(ms):
    return f"{ms:,.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def col_for(vals, higher_good=True):
    if all(v >= 1 for v in vals):
        return GREEN if higher_good else VERM
    if all(v < 1 for v in vals):
        return VERM if higher_good else GREEN
    return INK


def ep_pairs():
    """EP-only effect: same layout with EP on divided by EP off, for every input size and c1/c32."""
    pairs = (("B", "B0"), ("C1", "A"), ("E", "D"))
    return [D.out(on, s, c) / D.out(off, s, c) for on, off in pairs for c in (1, 32) for s in SIZES]


def ep_tpot_pairs():
    """EP on divided by EP off, single-request median TPOT, for the three layout pairs and every input size."""
    pairs = (("B", "B0"), ("C1", "A"), ("E", "D"))
    return [D.point(on, s, 1)["median_tpot_ms"] / D.point(off, s, 1)["median_tpot_ms"] for on, off in pairs for s in SIZES]


def fig7_clean_ratios():
    """B200 ratio of the 2026-10-21 baseline A (own seed per point), converted to total tok/s: {input: ratio}."""
    return {s: D.out("A", s, 32) * (s + 1024) / 1024 / D.PUB_FIG7[s][0] for s in SIZES}


def s09a():
    fig, ax = slide(
        "실험 7개 구성: 속도·용량·시작 시간 한눈에",
        "출력 tok/s는 기준 구성 대비 배수. 입력 1K / 8K / 32K 순, 조건당 1회 측정",
        "FP8 5구성은 A(TP8×PP2) 대비, NVFP4 2구성은 D(NVFP4 TP8) 대비. 영국 남부, FP8은 InfiniBand, 2026-10-21. c1 = 한 요청, c32 = 32개 동시, 출력 1,024 토큰. "
        "초록 = 세 입력 모두 기준 이상(TPOT는 이하), 주황 = 모두 기준 미만(TPOT는 초과). 시작(초)은 1회 측정이며 실행 순서·캐시 영향을 받는다. 출처: tables/t01_layout_summary.csv",
    )
    cols = [("구성", 14.5, "l"), ("GPU", 4.5, "c"), ("시작\n(초)", 5.5, "c"), ("KV 캐시 토큰", 14, "c"), ("최대\n컨텍스트", 8.5, "c"),
            ("32K·c32\n동시 실행", 8, "c"), ("c1 출력\n(기준 대비 ×)", 13, "c"), ("c1 TPOT\n(기준 대비 ×)", 13, "c"), ("c32 출력\n(기준 대비 ×)", 13, "c")]
    rows = []
    for code in D.ORDER:
        L = LAYOUT[code]
        base = D.BASE_OF[code] == code
        kv = " + ".join(f"{v:,}" for v in D.kv_list(code))
        ro1 = [D.ratio(code, s, 1) for s in SIZES]
        ro32 = [D.ratio(code, s, 32) for s in SIZES]
        rt = [D.tpot_ratio(code, s) for s in SIZES]
        if base:
            c1 = ("기준 " + trio([D.out(code, s, 1) for s in SIZES], "{:.1f}") + " tok/s", {"b": True})
            tp = ("기준 " + trio([D.point(code, s, 1)["median_tpot_ms"] for s in SIZES], "{:.1f}") + " ms", {"b": True})
            c32 = ("기준 " + trio([D.out(code, s, 32) for s in SIZES], "{:.0f}") + " tok/s", {"b": True})
        else:
            c1 = (trio(ro1), {"c": col_for(ro1), "b": True})
            tp = (trio(rt), {"c": col_for(rt, False), "b": True})
            c32 = (trio(ro32), {"c": col_for(ro32), "b": True})
        rows.append([(f"{code} · {L['short']}", {"b": True}), str(D.NODES[code] * 8),         str(D.healthy(code)), kv,
                     f"{D.max_ctx(code):,}", str(D.running32k(code)), c1, tp, c32])
    bottom = table(ax, 3, 46.8, cols, rows, fs=11.5, hfs=11.5, padx=0.8, pady=0.6)

    ep = ep_pairs()
    assert len(ep) == 18 and max(ep) < 1.0
    fp8_c1_best = all(D.out("A", s, 1) > D.out(c, s, 1) for s in SIZES for c in ("B0", "B", "C1", "C2"))
    c2_all = all(D.ratio("C2", s, 32) > 1 for s in SIZES)
    others_c32 = [c for c in ("B0", "B", "C1") if all(D.ratio(c, s, 32) > 1 for s in SIZES)]
    assert fp8_c1_best and c2_all and not others_c32
    c2r = [D.ratio("C2", s, 32) for s in SIZES]
    c2c1 = [D.ratio("C2", s, 1) for s in SIZES]
    b0r = [D.ratio("B0", s, 32) for s in SIZES]
    assert b0r[0] > 1 and b0r[1] > 1 and b0r[2] < 1, b0r
    tp_ep = ep_tpot_pairs()
    w, gap, h = 30.0, 2.0, 12.0
    y0 = 5.8
    card(ax, 3, y0, w, h, "EP만 켜면 5~12% 느렸다",
         f"EP 켬/끔 3쌍(각 1회)의 18개 비교가 모두 1.0 미만이다 ({min(ep):.3f}~{max(ep):.3f}). 한 요청 TPOT는 "
         f"{100 * (min(tp_ep) - 1):.0f}~{100 * (max(tp_ep) - 1):.0f}% 늘었다.", ec=ORANGE, fs=12, hc=VERM)
    card(ax, 3 + w + gap, y0, w, h, "한 요청은 FP8 중 TP8×PP2가 가장 빠르다",
         f"FP8 5구성 중 한 요청 출력 tok/s가 세 입력 모두 가장 높다 (1K {D.out('A', 1024, 1):.1f}). NVFP4 TP8은 GPU 8장으로 {D.out('D', 1024, 1):.1f}.", ec=NAVY, fs=12)
    card(ax, 3 + 2 * (w + gap), y0, w, h, "동시 32개에서 세 입력 모두 앞선 것은 DP2×TP8+EP16뿐",
         f"세 입력에서 ×{min(c2r):.2f}~{max(c2r):.2f}. TP16은 1K·8K에서 ×{b0r[0]:.2f}·{b0r[1]:.2f}다. 대신 한 요청은 ×{min(c2c1):.2f}~{max(c2c1):.2f}, "
         f"컨텍스트 상한은 {D.max_ctx('C2'):,}.",
         ec=GREEN, fs=12, hc=GREEN)
    assert bottom > y0 + h + 0.4, bottom
    save(fig, "s09a_summary-table.png")


def s09b():
    fig, ax = slide(
        "기준 대비 속도: EP는 느리고, 동시 32에서 세 입력 모두 앞선 것은 DP2뿐",
        "출력 tok/s를 기준 구성으로 나눈 값. 1.00보다 크면 기준보다 빠르다",
        "기준: FP8은 A(TP8×PP2), NVFP4는 D(NVFP4 TP8). 영국 남부, FP8은 InfiniBand, 2026-10-21. 입력 1K/8K/32K, 출력 1,024 토큰, 조건당 1회. "
        "TP16은 EP 없는 구성(B0), TP16+EP16은 B. 출처: tables/t01_layout_summary.csv",
    )
    codes = ["B0", "B", "C1", "C2", "E"]
    names = {"B0": "TP16", "B": "TP16\n+EP16", "C1": "TP8×PP2\n+EP8", "C2": "DP2×TP8\n+EP16", "E": "NVFP4 TP8\n+EP8 (D 대비)"}
    for x, c, ttl in ((4.5, 1, "한 요청 (c1)"), (53.5, 32, "32개 동시 (c32)")):
        label(ax, x, 45.2, ttl, 15, INK, True)
        axc = panel(fig, x + 1.5, 17.5, 42, 24.5)
        series = [(D.SZ[s], [D.ratio(code, s, c) for code in codes], SIZE_COLORS[i]) for i, s in enumerate(SIZES)]
        grouped_bars(axc, [names[k] for k in codes], series, fmt="{:.2f}", ymax=1.55, fs_val=10, fs_tick=11)
        axc.set_yticks([0, 0.5, 1.0, 1.5])
        axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        axc.axhline(1.0, color=INK, ls="--", lw=1.6, zorder=2)
        axc.text(len(codes) - 0.52, 1.1, "기준 1.00", ha="right", va="bottom", fontsize=11, color=INK, fontweight="bold")
    legend_row(ax, [("입력 1K", SIZE_COLORS[0]), ("입력 8K", SIZE_COLORS[1]), ("입력 32K", SIZE_COLORS[2])], 6, 10.4, fs=12.5)
    label(ax, 6, 7.6, "세 입력 모두 1.00을 넘는 구성은 32개 동시에서 DP2×TP8+EP16뿐이다. 나머지는 한 요청·32개 동시 모두 1.00 근처이거나 그 아래다.", 12.5, INK, max_w=90)
    save(fig, "s09b_ratio-bars.png")


def s09c():
    fig, ax = slide(
        "KV 캐시 용량이 구성마다 다르다",
        "서버가 시작할 때 확보한 KV 캐시 풀(토큰 수). 풀이 작으면 긴 요청을 동시에 많이 받지 못한다",
        "KV 캐시는 BF16(MLA). 영국 남부, FP8은 InfiniBand, 2026-10-21. 막대 길이는 서버 로그의 KV 캐시 토큰 수. 원인 줄은 구조에서 읽은 해석이며 측정값이 아니다. "
        "출처: tables/t01_layout_summary.csv, evidence/a100-expert-parallel-comparison.json",
    )
    x0, scale = 27.0, 40.0 / 700000
    top = 43.0
    label(ax, 3, top + 2.2, "구성", 12.5, GREY, True)
    label(ax, x0, top + 2.2, "KV 캐시 토큰 (막대) · 점선 = 모델 최대 컨텍스트 262,144", 12.5, GREY, True)
    label(ax, 76.0, top + 2.2, "최대\n컨텍스트".replace("\n", " "), 12.5, GREY, True, ha="center")
    label(ax, 90.0, top + 2.2, "32K·c32 동시", 12.5, GREY, True, ha="center")
    step, bh = 4.2, 2.7
    for i, code in enumerate(D.ORDER):
        yc = top - 1.5 - i * step
        if i % 2 == 0:
            ax.add_patch(Rectangle((2.5, yc - step / 2), 95, step, fc=LIGHT, ec="none", zorder=0.5))
        L = LAYOUT[code]
        label(ax, 3.2, yc, f"{code} · {L['short']}", 12.5, INK, True, max_w=22.5)
        kv = D.kv_list(code)
        xx = x0
        for j, v in enumerate(kv):
            wv = v * scale
            ax.add_patch(Rectangle((xx, yc - bh / 2), wv, bh, fc=L["color"], ec="white", lw=1.2, zorder=3, alpha=1.0 if j == 0 else 0.62))
            xx += wv + 0.15
        label(ax, xx + 0.8, yc, " + ".join(f"{v:,}" for v in kv) + (" (복제본 2개)" if len(kv) > 1 else ""), 11.5, INK)
        label(ax, 76.0, yc, f"{D.max_ctx(code):,}", 12, INK, ha="center")
        label(ax, 90.0, yc, f"{D.running32k(code)} / 32", 12, INK, ha="center")
        _mark(ax, yc - step / 2)
    xl = x0 + 262144 * scale
    ax.plot([xl, xl], [top - 1.5 - 6 * step - step / 2, top - 0.2], color=VERM, ls="--", lw=1.8, zorder=4)
    b0 = D.kv_list("B0")[0] / D.kv_list("A")[0]
    assert abs(b0 - 0.46) < 0.005
    w_c2, w_b0 = (D.cfg(c)["parallel"]["model_loading_gib"]["node0"][0] for c in ("C2", "B0"))
    assert 3.0 < w_c2 - w_b0 < 3.2, (w_c2, w_b0)
    y1 = 12.4
    label(ax, 3, y1, f"TP16은 KV가 TP8×PP2의 {b0:.2f}배다. 해석: MLA KV 캐시가 TP 랭크마다 복제되고, TP16에서는 각 랭크가 61층 전체를 맡는다.", 12.5, INK, max_w=94)
    label(ax, 3, y1 - 2.4, f"DP2×TP8+EP16은 GPU당 가중치가 TP16보다 {w_c2 - w_b0:.1f} GiB 많고(대부분 GPU마다 복제한 공유 전문가, 추정) "
          f"그만큼 KV가 줄어 컨텍스트 상한이 {D.max_ctx('C2'):,}이다.", 12.5, INK, max_w=94)
    label(ax, 3, y1 - 4.8, f"NVFP4는 노드 1대(GPU 8장)라서 KV 풀이 {D.kv_list('D')[0]:,} 토큰이고, 최대 컨텍스트도 그 값으로 제한된다.", 12.5, INK, max_w=94)
    save(fig, "s09c_kv-capacity.png")


def s10a():
    names = [("aime", "AIME26"), ("kobalt", "KoBALT"), ("click", "CLIcK"), ("ifbench", "IFBench")]
    fig, ax = slide(
        "정확도: 부분 실행한 벤치마크 4종",
        "SKT 공개값과 우리 측정값(FP8 native TP8×PP2, thinking 모드). 주황 선은 우리 측정의 95% 신뢰구간",
        "thinking 모드, T 0.6, top_p 0.95. 요청으로 일찍 멈춘 부분 실행이다. 측정 리전은 증거에 기록되지 않았고, 데모 클러스터에서 2026-10-06에 실행했다. "
        "공개값은 SKT 모델 카드(thinking 모드 표) 값이다. 출처: tables/t03_accuracy.csv, evidence/a100-benchmarks.json",
    )
    label(ax, 4.5, 45.4, "점수 (%)", 15, INK, True)
    axc = panel(fig, 5.5, 14.5, 52, 28.0)
    ticks = []
    for i, (k, nm) in enumerate(names):
        d = D.BM["summary"][k]
        pub = float(D.PUB_ROWS[k]["skt"])
        ours = d["score"] * 100
        lo, hi = (v * 100 for v in d["ci95"])
        axc.bar(i - 0.2, pub, 0.36, color=GREY2, zorder=3)
        axc.bar(i + 0.2, ours, 0.36, color=NAVY, zorder=3)
        axc.errorbar(i + 0.2, ours, yerr=[[ours - lo], [hi - ours]], fmt="none", ecolor=ORANGE, elinewidth=2.4, capsize=7, capthick=2.4, zorder=5)
        axc.text(i - 0.2, pub + 1.2, f"{pub:.1f}", ha="center", va="bottom", fontsize=12, color=INK)
        axc.text(i + 0.2, hi + 1.2, f"{ours:.1f}" + (" ※" if k == "aime" else ""), ha="center", va="bottom", fontsize=12, color=NAVY, fontweight="bold")
        ticks.append(f"{nm}\n{d['n_items']}/{d['items_expected']}문항")
    axc.set_xticks(range(4))
    axc.set_xticklabels(ticks, fontsize=12.5, color=INK)
    axc.set_ylim(0, 118)
    axc.set_yticks([0, 20, 40, 60, 80, 100])
    legend_row(ax, [("SKT 공개값", GREY2), ("우리 측정", NAVY), ("95% 신뢰구간", ORANGE)], 6, 10.0, fs=12.5)

    kb, cl, ifb, ai = (D.BM["summary"][k] for k in ("kobalt", "click", "ifbench", "aime"))
    in_ci = lambda k, d: d["ci95"][0] * 100 <= float(D.PUB_ROWS[k]["skt"]) <= d["ci95"][1] * 100
    assert in_ci("kobalt", kb) and in_ci("ifbench", ifb) and not in_ci("click", cl)
    gap = float(D.PUB_ROWS["click"]["skt"]) - cl["ci95"][1] * 100
    assert 0.03 < gap < 0.07, gap
    x, w = 62.0, 35.0
    card(ax, x, 33.6, w, 11.4, "부분 실행이다",
         f"예정 200문항 중 KoBALT {kb['n_items']}, CLIcK {cl['n_items']}, IFBench {ifb['n_items']}문항만 채점했다. AIME26은 30문항 중 {ai['n_items']}문항이다.", ec=ORANGE, fs=12, hc=VERM)
    card(ax, x, 15.4, w, 17.0, "공개값과의 거리",
         f"KoBALT·IFBench는 공개값이 신뢰구간 안에 있다. CLIcK은 공개값 {float(D.PUB_ROWS['click']['skt']):.1f}이 구간 상한 {cl['ci95'][1] * 100:.2f}보다 {gap:.2f}%p 높아 사실상 경계이고, 우리 값은 {cl['score'] * 100:.1f}이다. "
         "※ AIME26은 10문항이 모두 정답이라 신뢰구간이 0으로 붕괴해 비교할 수 없다.", ec=NAVY, fs=12)
    card(ax, x, 5.8, w, 8.4, "조건 확인 필요", "공개값과 같은 평가 설정인지는 검증하지 않았다.", ec=GREY, fs=12, hc=GREY)
    save(fig, "s10a_accuracy.png")


def beyond_max(key):
    vals = [v["beyond_index_topk"]["chosen_logprob"]["max_abs_diff"] for v in D.L2["per_prompt_vs_official"][key].values() if "beyond_index_topk" in v]
    return max(vals)


def s10b():
    fig, ax = slide(
        "2층 log-prob: 2,048 위치 밖에서 dense만 차이가 커진다",
        "2층만 · 전체 모델 정확도 아님. 공식 구현(FP32) 대비 정답 토큰 log-prob의 차이 |Δ|",
        "앞 2층만 실제 가중치로 돌렸다. 공식 구현은 FP32·CPU(스웨덴 중부), vLLM은 이탈리아 북부. 14,618 위치(안쪽 7,356 · 바깥쪽 7,262). "
        "출처: tables/t07_logprob.csv, evidence/a100-native-dsa-2layer-vs-official.json, evidence/a100-native-2layer-logprob-series.json",
    )
    keys = [("dense", "dense TP8", GREY2), ("native", "native TP8", NAVY), ("native_pp2", "native TP8×PP2", ORANGE)]
    label(ax, 4.5, 45.6, "평균 |Δ log-prob| (작을수록 공식 구현에 가깝다)", 14.5, INK, True)
    legend_row(ax, [(n, c) for _, n, c in keys], 5.5, 42.8, fs=12)
    axc = panel(fig, 6, 14.5, 44, 25.5)
    series = []
    for k, nm, colr in keys:
        s = D.SUMV[D.LP_KEYS[k]]
        series.append((nm, [s["within_index_topk"]["mean_abs_diff"], s["beyond_index_topk"]["mean_abs_diff"]], colr))
    grouped_bars(axc, ["안쪽 (앞 2,048 위치)", "바깥쪽 (2,048 이후)"], series, fmt="{:.4f}", ymax=0.0155, fs_val=11, fs_tick=12.5)
    axc.set_yticks([0, 0.005, 0.01, 0.015])
    axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))

    cols = [("구성", 15, "l"), ("안쪽\ntop-1 일치", 9, "c"), ("바깥쪽\ntop-1 일치", 9, "c"), ("바깥쪽\n최대 |Δ|", 9, "c")]
    rows = []
    for k, nm, _ in keys:
        s = D.SUMV[D.LP_KEYS[k]]
        rows.append([(nm, {"b": True}), f"{s['within_index_topk']['top1_agreement'] * 100:.1f}%", f"{s['beyond_index_topk']['top1_agreement'] * 100:.1f}%",
                     f"{beyond_max(D.LP_KEYS[k]):.3f}"])
    bottom = table(ax, 54, 46.2, cols, rows, fs=12, hfs=11.5, padx=0.8, pady=0.65)
    dm, nm_ = beyond_max(D.LP_KEYS["dense"]), beyond_max(D.LP_KEYS["native"])
    assert 0.2 < dm < 0.25 and 0.12 < nm_ < 0.14, (dm, nm_)
    d_in = D.SUMV[D.LP_KEYS["dense"]]["within_index_topk"]["mean_abs_diff"]
    d_out = D.SUMV[D.LP_KEYS["dense"]]["beyond_index_topk"]["mean_abs_diff"]
    n_out = D.SUMV[D.LP_KEYS["native"]]["beyond_index_topk"]["mean_abs_diff"]
    card(ax, 54, 6.0, 43, bottom - 7.6, "읽는 법",
         f"안쪽에서는 dense와 native가 같은 함수를 계산한다. 공식 구현과의 거리 {d_in:.4f}는 BF16 대 FP32 수치 차이다.\n"
         f"바깥쪽에서 native는 안쪽과 같은 수준이고({n_out:.4f}), dense는 {d_out:.4f}로 native보다 {(d_out / n_out - 1) * 100:.0f}% 크다. 최대값은 {dm:.3f} 대 {nm_:.3f}이다.\n"
         "top-1 일치율은 세 구성 모두 97%대로 거의 같다.\n"
         "전체 61층 모델은 FP32 CPU 기준 구현으로 돌릴 수 없어 비교하지 못했다.", ec=NAVY, fs=12)
    save(fig, "s10b_logprob-2layer.png")


def s11a():
    fig, ax = slide(
        "기준 A의 속도: 입력이 길면 동시성이 떨어진다",
        "A = TP8×PP2, FP8, GPU 16장. 한 요청(c1)과 32개 동시(c32), 입력 1K / 8K / 32K",
        "영국 남부, InfiniBand, 2026-10-21, 출력 1,024 토큰, 조건당 1회. TPOT = 출력 토큰 1개를 만드는 시간(중앙값), TTFT = 첫 토큰까지 걸리는 시간(중앙값). "
        "출처: tables/t01_layout_summary.csv, tables/t02_ep_points_long.csv",
    )
    for x, c, ttl, fmt, ymax in ((4.0, 1, "한 요청 (c1) 출력 tok/s", "{:.1f}", 62), (32.0, 32, "32개 동시 (c32) 출력 tok/s", "{:,.0f}", 480)):
        label(ax, x, 45.4, ttl, 14, INK, True)
        axc = panel(fig, x + 1.5, 18.0, 23, 24)
        grouped_bars(axc, KS, [("A", [D.out("A", s, c) for s in SIZES], NAVY)], fmt=fmt, ymax=ymax, fs_val=12.5, fs_tick=13)
    cols = [("입력", 6, "c"), ("c1\nTPOT", 7, "c"), ("c1\nTTFT", 7.5, "c"), ("c32\nTPOT", 7, "c"), ("c32\nTTFT", 7.5, "c")]
    rows = []
    for s in SIZES:
        p1, p32 = D.point("A", s, 1), D.point("A", s, 32)
        rows.append([(D.SZ[s], {"b": True}), f"{p1['median_tpot_ms']:.1f} ms", fmt_t(p1["median_ttft_ms"]), f"{p32['median_tpot_ms']:.0f} ms", fmt_t(p32["median_ttft_ms"])])
    label(ax, 61, 45.4, "지연 시간 (중앙값)", 14, INK, True)
    bottom = table(ax, 61, 43.4, cols, rows, fs=12, hfs=12, padx=0.7, pady=0.7)
    p = D.point("A", 32768, 32)
    card(ax, 61, 18.4, 36, bottom - 1.6 - 18.4, "32K 입력에서 일어난 일",
         f"32개 동시 요청 중 동시 실행은 최대 {p['peak_running_requests']:.0f}개, 대기는 최대 {p['peak_waiting_requests']:.0f}개였다 (각각의 최대치). 첫 토큰까지 중앙값은 {p['median_ttft_ms'] / 1000:.0f}초다.",
         ec=ORANGE, fs=12, hc=VERM)
    p1k, p32k = D.point("A", 1024, 1), D.point("A", 32768, 1)
    for q in (p1k, p32k):
        est = 1024 / (q["median_ttft_ms"] / 1000 + 1023 * q["median_tpot_ms"] / 1000)
        assert abs(est / q["output_throughput"] - 1) < 0.02, (est, q["output_throughput"])
    card(ax, 61, 6.0, 36, 11.0, "출력 tok/s에는 첫 토큰 대기가 든다",
         f"c1에서 1K→32K로 가면 {p1k['output_throughput']:.1f}→{p32k['output_throughput']:.1f} tok/s로 줄지만, 줄어든 몫은 대부분 첫 토큰까지 걸리는 시간"
         f"({p1k['median_ttft_ms'] / 1000:.2f}초→{p32k['median_ttft_ms'] / 1000:.1f}초)이다.",
         ec=NAVY, fs=12)
    inc = (p32k["median_tpot_ms"] / p1k["median_tpot_ms"] - 1) * 100
    card(ax, 3, 5.8, 56.5, 8.0, f"한 요청 TPOT는 입력이 길어도 {inc:.0f}%만 늘어난다",
         f"TPOT는 {p1k['median_tpot_ms']:.1f} ms에서 {p32k['median_tpot_ms']:.1f} ms로 늘었다. 동시 32개에서는 처리량이 {D.out('A', 1024, 32):.0f}에서 {D.out('A', 32768, 32):.0f} tok/s로 줄었다.",
         ec=NAVY, fs=12)
    save(fig, "s11a_baseline-speed.png")


def s11b():
    fig, ax = slide(
        "Fig. 7 조건: B200 공개값 대비 A100 16장",
        "동시 32개, 출력 1,024 토큰, 입력 1K~120K. 총 처리량(입력+출력 tok/s) 기준",
        "B200: SKT 기술보고서 Fig. 7 막대 라벨 값(FP8). A100×16: 같은 조건으로 측정, 영국 남부, InfiniBand, 2026-10-03, 조건당 1회. "
        "입력이 길면 KV 풀 한계로 동시 실행이 줄어든다(native 기준 32K 20개, 64K 10개, 120K 5개). 출처: tables/t04_fig7_doc_speed.csv",
    )
    sizes = sorted(D.FIG7)
    pub = [D.PUB_FIG7[s][0] for s in sizes]
    nat = [D.FIG7[s]["native"]["total_token_throughput"] for s in sizes]
    den = [D.FIG7[s]["dense"]["total_token_throughput"] for s in sizes]
    assert [D.FIG7[s]["native"]["peak_running_requests"] for s in (32768, 65536, 120000)] == [20.0, 10.0, 5.0]
    series = [("B200 공개값 (8장)", pub, GREY2), ("A100×16 native", nat, NAVY), ("A100×16 dense", den, SKY)]
    legend_row(ax, [(n, c) for n, _, c in series], 24, 46.6, fs=12.5)
    label(ax, 3, 46.6, "총 tok/s", 12.5, GREY, True)
    axc = panel(fig, 23.0, 29.8, 74.0, 15.0)
    n = len(sizes)
    bw = 0.82 / 3
    for si, (_, vals, colr) in enumerate(series):
        axc.bar(np.arange(n) + (si - 1) * bw, vals, bw * 0.92, color=colr, zorder=3)
    axc.set_xlim(-0.5, n - 0.5)
    axc.set_xticks([])
    axc.set_ylim(0, 13500)
    axc.set_yticks([0, 4000, 8000, 12000])

    colw = 74.0 / n
    cols = [("입력 토큰", 20, "l")] + [(D.SZ[s] + ("†" if s in TAINTED else ""), colw, "c") for s in sizes]
    nat_r = [a / b for a, b in zip(nat, pub)]
    den_r = [a / b for a, b in zip(den, pub)]
    rows = [
        [("B200 공개값", {"b": True})] + [f"{v:,.0f}" for v in pub],
        [("A100×16 native", {"b": True})] + [f"{v:,.0f}" for v in nat],
        [("  B200 대비 ×", {"c": GREY})] + [(f"{v:.2f}", {"c": GREY}) for v in nat_r],
        [("A100×16 dense", {"b": True})] + [f"{v:,.0f}" for v in den],
        [("  B200 대비 ×", {"c": GREY})] + [(f"{v:.2f}", {"c": GREY}) for v in den_r],
    ]
    bottom = table(ax, 3, 29.3, cols, rows, fs=11.5, hfs=11.5, padx=0.6, pady=0.5)
    clean = fig7_clean_ratios()
    assert abs(clean[1024] - nat_r[0]) < 0.01 and clean[8192] < nat_r[sizes.index(8192)] - 0.1, (clean, nat_r)
    label(ax, 3, bottom - 2.2, f"native는 B200의 {min(nat_r):.2f}~{max(nat_r):.2f}배, dense는 {min(den_r):.2f}~{max(den_r):.2f}배다(† 포함). 입력이 길수록 격차가 벌어진다.", 12.5, INK, max_w=94)
    label(ax, 3, bottom - 4.6, "†: 같은 시드로 앞 지점의 프롬프트를 캐시에서 재사용해 부풀었다. 재측정 전까지 쓰지 않는다. "
          f"2026-10-21 재측정 환산(native): 1K {clean[1024]:.2f}, 8K {clean[8192]:.2f}, 32K {clean[32768]:.2f}배.", 12.5, VERM, max_w=94)
    label(ax, 3, bottom - 7.0, "dense는 DSA 희소 선택을 건너뛰어 더 빠르지만, 긴 문맥에서는 정확하지 않았다 (슬라이드 18).", 12.5, INK, max_w=94)
    save(fig, "s11b_fig7-b200-vs-a100.png")


ALL = [s09a, s09b, s09c, s10a, s10b, s11a, s11b]
