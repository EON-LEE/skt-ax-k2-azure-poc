"""Slides 12-17: each parallel layout against its baseline (TP16, TP16+EP16, TP8xPP2+EP8, DP2xTP8+EP16, NVFP4 TP8, NVFP4 TP8+EP8)."""
from common import *
from common import _mark
import data as D
from figs_summary import trio, fmt_t, col_for, ep_pairs, KS, SIZES, GREY2, TAINTED

LOG_NOTE = ("영국 남부, 2026-10-21, 출력 1,024 토큰, 조건당 1회라 몇 % 차이는 반복하면 달라질 수 있다. TPOT = 출력 토큰 1개당 시간(중앙값), 가중치/KV = GPU당 GiB. "
            "출처: tables/t01_layout_summary.csv, tables/t02_ep_points_long.csv")


def rel(code, base, isl, c):
    return D.out(code, isl, c) / D.out(base, isl, c)


def rels(code, base, c):
    return [rel(code, base, s, c) for s in SIZES]


def slower(rs):
    """Percent range by which every ratio in rs is below 1."""
    assert all(r < 1 for r in rs), rs
    return round(100 * (1 - max(rs))), round(100 * (1 - min(rs)))


def faster(rs):
    assert all(r > 1 for r in rs), rs
    return round(100 * (min(rs) - 1)), round(100 * (max(rs) - 1))


def gib(code):
    p = D.cfg(code)["parallel"]
    return max(v[0] for v in p["model_loading_gib"].values()), max(v[0] for v in p["kv_cache_gib"].values())


def n_gpu(code):
    return 8 * D.NODES[code]


def spec_cells(code):
    w, k = gib(code)
    tp = [D.point(code, s, 1)["median_tpot_ms"] for s in SIZES]
    return [f"{D.healthy(code)} 초", f"{w:.1f} / {k:.1f}", D.kv_label(code).replace(" + ", "\n+ "),
            f"{D.max_ctx(code):,}", f"{D.running32k(code)}개", "/".join(f"{v:.1f}" for v in tp)]


SPEC_HEADS = ["서버 시작", "가중치/KV GiB", "KV 캐시 토큰", "최대 컨텍스트", "동시 실행\n(32K·c32)", "c1 TPOT (ms)\n1K/8K/32K"]


def compare_slide(code, base, fname, title, subtitle, note, interp, side, lab_base=None, lab_this=None, side_ec=ORANGE, side_hc=VERM):
    """Two bar panels (c1, c32) of this layout against its baseline, a spec table, a side card and an interpretation card."""
    fig, ax = slide(title, subtitle, note)
    a, b = D.LAYOUT[base], D.LAYOUT[code]
    lab_base, lab_this = lab_base or a["label"], lab_this or b["label"]
    legend_row(ax, [(lab_base, a["color"]), (lab_this, b["color"])], 4, 46.6, fs=12.5)
    for x, c, ttl, fmt in ((4.0, 1, "한 요청 (c1) 출력 tok/s", "{:.1f}"), (33.0, 32, "32개 동시 (c32) 출력 tok/s", "{:,.0f}")):
        label(ax, x, 44.0, ttl, 14, INK, True)
        vb = [D.out(base, s, c) for s in SIZES]
        vc = [D.out(code, s, c) for s in SIZES]
        axc = panel(fig, x + 1.5, 18.4, 23, 23.2)
        grouped_bars(axc, KS, [(lab_base, vb, a["color"]), (lab_this, vc, b["color"])], fmt=fmt, ymax=max(vb + vc) * 1.3,
                     fs_val=11.5, fs_tick=13, ratios=[[None] * 3, rels(code, base, c)])
    label(ax, 61, 44.0, "구성 비교", 14, INK, True)
    cols = [("항목", 11.5, "l"), (b["short"], 12.25, "c"), (a["short"], 12.25, "c")]
    cb, ca = spec_cells(code), spec_cells(base)
    rows = [[(h, {"b": True}), (x, {"b": True}), y] for h, x, y in zip(SPEC_HEADS, cb, ca)]
    bottom = table(ax, 61, 42.2, cols, rows, fs=11, hfs=11, padx=0.6, pady=0.5)
    h = bottom - 1.4 - 5.8
    assert h >= 8.0, ("side card too small", h)
    card(ax, 61, 5.8, 36, h, side[0], side[1], ec=side_ec, fs=11.5, hc=side_hc)
    card(ax, 3, 5.8, 56.5, 9.6, interp[0], interp[1], ec=NAVY, fs=11.5)
    save(fig, fname)


def s12():
    r1, r32 = rels("B0", "A", 1), rels("B0", "A", 32)
    lo, hi = slower(r1)
    kvr = D.kv_list("B0")[0] / D.kv_list("A")[0]
    assert abs(kvr - 0.46) < 0.005
    tp = [D.tpot_ratio("B0", s) for s in SIZES]
    assert 1.23 < min(tp) and max(tp) < 1.27, tp
    dense = {s: D.SPEC[s]["tp16_dense"]["total_token_throughput"] / D.FIG7[s]["dense"]["total_token_throughput"] for s in (1024, 4096, 8192)}
    compare_slide(
        "B0", "A", "s12_tp16-vs-baseline.png",
        f"TP16: 한 요청은 {lo}~{hi}% 느리고 KV는 {kvr:.2f}배",
        "TP16(노드 2개에 걸친 텐서 병렬) 대 기준 A(TP8×PP2). 둘 다 FP8, GPU 16장",
        LOG_NOTE,
        ("해석 (프로파일링하지 않음)",
         "로그에 “Custom allreduce is disabled because this process group spans across nodes” 줄이 남았다(사실). "
         f"토큰마다 노드 사이 all-reduce가 드는 것이 TPOT 증가(+{100 * (min(tp) - 1):.0f}~{100 * (max(tp) - 1):.0f}%)의 원인으로 보이나 측정하지 않았다. "
         "KV가 줄어든 것은 MLA KV가 TP 랭크마다 복제되기 때문으로 보인다."),
        ("참고: dense TP16은 다른 작업",
         f"같은 10-03 작업의 dense 모드에서 TP16은 TP8×PP2보다 c32 총 tok/s가 1K {dense[1024]:.2f}배, 4K {dense[4096]:.2f}배, 8K {dense[8192]:.2f}배였다. "
         "4K·8K는 같은 시드로 캐시를 재사용해 부푼 값이 섞였다. 모드와 지표가 달라 직접 비교하지 않는다."))
    assert max(r32) > 1 and min(r32) < 1


def s13():
    r1, r32 = rels("B", "B0", 1), rels("B", "B0", 32)
    lo = round(100 * (1 - max(r1 + r32)))
    hi = round(100 * (1 - min(r1 + r32)))
    assert max(r1 + r32) < 1
    a1, a32 = rels("B", "A", 1), rels("B", "A", 32)
    compare_slide(
        "B", "B0", "s13_tp16-ep16.png",
        f"TP16+EP16: EP를 켜면 TP16보다 {lo}~{hi}% 느리다",
        "같은 TP16에서 EP 켬/끔만 비교한다. 기준은 B0(TP16), 둘 다 FP8, GPU 16장",
        LOG_NOTE,
        ("해석 (프로파일링하지 않음)",
         "이 구성은 DP=1이라 all2all을 쓰지 않는다. 로그에 “MoEPrepareAndFinalizeNoDPEPModular”가 남았다(사실). "
         "expert가 랭크에 나뉘어도 routed+shared 합을 TP 그룹에서 all-reduce하는 것으로 보인다(코드 해석). "
         "랭크 간 expert 부하 불균형 가설은 측정하지 않았다."),
        ("기준 A와 비교하면",
         f"B는 A보다 c1에서 {trio(a1)}배, c32에서 {trio(a32)}배다(입력 1K/8K/32K). "
         "TP16 자체의 손실이 이미 있어, EP 효과만 보려면 B0와 비교해야 한다."),
        side_ec=NAVY, side_hc=None)


def s14():
    r1, r32 = rels("C1", "A", 1), rels("C1", "A", 32)
    lo = round(100 * (1 - max(r1 + r32)))
    hi = round(100 * (1 - min(r1 + r32)))
    assert max(r1 + r32) < 1
    assert D.max_ctx("C1") == D.max_ctx("A")
    compare_slide(
        "C1", "A", "s14_tp8pp2-ep8.png",
        f"TP8×PP2+EP8: 기준보다 {lo}~{hi}% 느리다",
        "TP8×PP2의 각 단계 안에서 EP8을 켠다. 기준은 A, 둘 다 FP8, GPU 16장",
        LOG_NOTE,
        ("해석 (프로파일링하지 않음)",
         "EP8이 한 노드 안(NVLink)에서만 이뤄지는데도 8~12% 느렸다. 손실을 노드 사이 통신 탓으로만 돌릴 수 없다. "
         "DP=1이라 all2all 없이 NoDPEP 경로를 쓴다(로그, 사실). 검토 추정 원인은 그룹 라우팅과 linear 전문가 배치가 겹친 부하 불균형이다."),
        ("KV·컨텍스트는 그대로다",
         f"KV 캐시 토큰({D.kv_label('C1')} 대 {D.kv_label('A')})과 최대 컨텍스트({D.max_ctx('C1'):,})는 같다. "
         f"서버 시작 {D.healthy('C1')}초(A {D.healthy('A')}초)는 1회 측정이고 실행 순서·캐시 영향을 받아 구성끼리 비교하지 않는다."),
        side_ec=NAVY, side_hc=None)


def s15():
    r1, r32 = rels("C2", "A", 1), rels("C2", "A", 32)
    lo, hi = faster(r32)
    slo, shi = slower(r1)
    for other in ("B0", "B", "C1"):
        assert min(rels(other, "A", 32)) < 1, other
    tt = [D.point("C2", s, 32)["median_ttft_ms"] / D.point("A", s, 32)["median_ttft_ms"] for s in SIZES]
    assert min(tt) < 1 < max(tt), tt
    compare_slide(
        "C2", "A", "s15_dp2tp8-ep16.png",
        f"DP2×TP8+EP16: 동시 32개는 {lo}~{hi}% 빠르다",
        f"그러나 한 요청(c1)은 {slo}~{shi}% 느리고 최대 컨텍스트는 {D.max_ctx('C2'):,} 토큰이다",
        LOG_NOTE,
        ("해석 (프로파일링하지 않음)",
         "로그에 “AgRsAll2AllManager”가 남았다(사실): MoE 층마다 복제본 둘이 allgather·reducescatter로 묶인다. "
         "요청이 많으면 일을 나누어 빠르고, 한 요청이면 노드 사이 통신과 쉬는 복제본의 빈 실행 탓에 느린 것으로 보이나 측정하지 않았다. "
         "API 서버 2개·배치 한도 합계 2배(엔진 2개)도 섞였다."),
        ("c32 첫 토큰 대기는 뒤섞인다",
         f"A 대비 c32 첫 토큰 중앙값은 {trio(tt)}배(1K/8K/32K)다. c32에서 세 입력 모두 A를 앞선 FP8 구성은 이것뿐이다."))


def s16a():
    r1, r32 = rels("D", "A", 1), rels("D", "A", 32)
    allr = r1 + r32
    lo, hi = min(allr), max(allr)
    assert abs(lo - 0.83) < 0.005 and abs(hi - 1.10) < 0.005, (lo, hi)
    wd, kd = gib("D")
    wa, ka = gib("A")
    compare_slide(
        "D", "A", "s16a_nvfp4-vs-fp8.png",
        f"NVFP4 TP8: GPU 8장으로 FP8 16장의 {lo:.2f}~{hi:.2f}배",
        "교차 할당 비교: NVFP4는 노드 1개(GPU 8장), 기준 A는 FP8 노드 2개(GPU 16장)",
        "영국 남부, 2026-10-21, 출력 1,024 토큰, 조건당 1회라 몇 % 차이는 반복하면 달라질 수 있다. FP8은 InfiniBand로 노드 2개를 연결했다. "
        "TPOT = 출력 토큰 1개당 시간(중앙값), 가중치/KV = GPU당 GiB. 출처: tables/t01_layout_summary.csv, tables/t02_ep_points_long.csv",
        ("해석 (프로파일링하지 않음)",
         f"GPU당 가중치가 NVFP4 {wd:.1f} GiB, FP8 {wa:.1f} GiB로 비슷하다. 디코딩은 가중치 읽기가 병목이라 TPOT가 비슷한 것으로 보인다. "
         "A100은 FP8·FP4 연산 유닛이 없어 두 경우 모두 Marlin 커널(W8A16, W4A16)을 쓴다."),
        ("측정하지 않은 것",
         "NVFP4와 FP8의 정확도 동등성(기능 4/4, 도구 9/9, 바늘 9/9만 확인). "
         "“NVFP4 2복제본은 약 2배”는 추정이며 측정한 적 없다."),
        lab_base="FP8 TP8×PP2 (GPU 16장)", lab_this="NVFP4 TP8 (GPU 8장)")


def s16b():
    sizes = sorted(D.NVF7)
    k = [D.SZ[s] for s in sizes]
    pub = [D.PUB_FIG7[s][0] for s in sizes]
    fp8 = [D.NVF7[s]["fp8_16_a100_native_ib"]["total_token_throughput"] for s in sizes]
    nv = [D.NVF7[s]["nvfp4_8_a100"]["total_token_throughput"] for s in sizes]
    for s, v in zip(sizes, fp8):
        assert abs(v - D.FIG7[s]["native"]["total_token_throughput"]) < 0.5, s
    r = [a / b for a, b in zip(nv, fp8)]
    assert r[0] > 1 and r[1] > 1 and 0.95 < r[2] < 1.05 and all(x < 0.9 for x in r[3:]), r
    run_nv = int(D.NVF7[32768]["nvfp4_8_a100"]["peak_running_requests"])
    run_fp = int(D.NVF7[32768]["fp8_16_a100_native_ib"]["peak_running_requests"])
    same_day_8k = D.out("D", 8192, 32) / D.out("A", 8192, 32)
    assert abs(same_day_8k - 0.90) < 0.01 and r[sizes.index(8192)] < 0.75, (same_day_8k, r)
    fig, ax = slide(
        "Fig. 7 조건: NVFP4×8과 FP8×16 (작업이 다른 교차 비교, 재측정 전 참고값)",
        "동시 32개, 출력 1,024 토큰, 입력 1K~32K. 총 처리량(입력+출력 tok/s) 기준",
        "영국 남부. NVFP4×8(노드 1개, 2026-10-04 작업)과 FP8×16 native(InfiniBand, 2026-10-03 작업)는 작업이 달라 조건이 완전히 같지는 않다. "
        "B200은 SKT 기술보고서 Fig. 7 막대 라벨 값. 조건당 1회. 출처: tables/t04_fig7_doc_speed.csv",
    )
    series = [("B200 공개값 (8장)", pub, GREY2), ("A100 FP8×16 native", fp8, NAVY), ("A100 NVFP4×8", nv, GREEN)]
    legend_row(ax, [(n, c) for n, _, c in series], 24, 46.6, fs=12.5)
    label(ax, 3, 46.6, "총 tok/s", 12.5, GREY, True)
    axc = panel(fig, 23.0, 29.0, 74.0, 15.8)
    n = len(sizes)
    bw = 0.82 / 3
    for si, (_, vals, colr) in enumerate(series):
        axc.bar(np.arange(n) + (si - 1) * bw, vals, bw * 0.92, color=colr, zorder=3)
    axc.set_xlim(-0.5, n - 0.5)
    axc.set_xticks([])
    axc.set_ylim(0, 9000)
    axc.set_yticks([0, 3000, 6000, 9000])
    colw = 74.0 / n
    cols = [("입력 토큰", 20, "l")] + [(x, colw, "c") for x in k]
    rows = [
        [("B200 공개값", {"b": True})] + [f"{v:,.0f}" for v in pub],
        [("A100 FP8×16", {"b": True})] + [f"{v:,.0f}" + ("†" if s in TAINTED else "") for s, v in zip(sizes, fp8)],
        [("A100 NVFP4×8", {"b": True})] + [f"{v:,.0f}†" for v in nv],
        [("  NVFP4÷FP8 (배)", {"c": GREY})] + [(f"{v:.2f}†", {"c": GREY}) for v in r],
        [("  GPU당 환산 (배)", {"c": GREY})] + [(f"{2 * v:.2f}†", {"c": GREY}) for v in r],
    ]
    bottom = table(ax, 3, 28.4, cols, rows, fs=11.5, hfs=11.5, padx=0.6, pady=0.5)
    same_day = rels("D", "A", 1) + rels("D", "A", 32)
    label(ax, 3, bottom - 2.0, f"비율은 순위 판단에 쓰지 않는다. 같은 날 같은 조건 비교(s16a)에서 NVFP4 TP8은 FP8 16장의 "
          f"{min(same_day):.2f}~{max(same_day):.2f}배였다.", 12.5, INK, max_w=94)
    label(ax, 3, bottom - 4.2, f"†: 두 작업(FP8 10-03, NVFP4 10-04) 모두 지점마다 같은 시드를 써 처리량이 높게 나왔다. "
          f"8K {r[sizes.index(8192)]:.2f}배가 같은 날 {same_day_8k:.2f}배와 다른 이유다. 재측정 전까지 쓰지 않는다.", 12.5, VERM, max_w=94)
    label(ax, 3, bottom - 6.4, f"32K 동시 실행은 NVFP4 {run_nv}개, FP8 {run_fp}개다(NVFP4 KV 풀 {D.kv_label('D')} 토큰 한계로 보인다). "
          "GPU당 환산은 GPU 수(8 대 16)로 나눈 산술값이고, 정확도 동등성은 측정하지 않았다.", 12.5, INK, max_w=94)
    save(fig, "s16b_nvfp4-fig7.png")


def s17():
    r1, r32 = rels("E", "D", 1), rels("E", "D", 32)
    lo = round(100 * (1 - max(r1 + r32)))
    hi = round(100 * (1 - min(r1 + r32)))
    assert max(r1 + r32) < 1
    ep = ep_pairs()
    assert len(ep) == 18 and max(ep) < 1.0
    compare_slide(
        "E", "D", "s17_nvfp4-ep8.png",
        f"NVFP4 TP8+EP8: EP를 켜면 {lo}~{hi}% 느리다",
        "같은 NVFP4 TP8에서 EP 켬/끔만 비교한다. 기준은 D(NVFP4 TP8), 노드 1개",
        "영국 남부, 2026-10-21, 노드 1개(GPU 8장), 출력 1,024 토큰, 조건당 1회라 몇 % 차이는 반복하면 달라질 수 있다. "
        "TPOT = 출력 토큰 1개당 시간(중앙값), 가중치/KV = GPU당 GiB. 출처: tables/t01_layout_summary.csv, tables/t02_ep_points_long.csv",
        ("해석 (프로파일링하지 않음)",
         "GPU 8장이 모두 한 노드 안(NVLink)에 있는데도 EP를 켜면 느려졌다. DP=1이라 all2all은 없다(로그, 사실). "
         "랭크마다 expert 부하가 달라 가장 바쁜 랭크를 기다리는 것으로 볼 수 있으나 랭크별 부하는 측정하지 않았다."),
        ("EP 효과는 어디서나 같은 방향이다",
         f"EP를 켠 비교 {len(ep)}개(3쌍×입력 3개×c1/c32)가 모두 1.0 미만, 범위 {min(ep):.3f}~{max(ep):.3f}배다. "
         f"서버 시작 EP 켬 {D.healthy('E')}초, 끔 {D.healthy('D')}초는 1회 측정이고 실행 순서(D 다음 E)·캐시 영향을 받는다."),
        side_ec=NAVY, side_hc=None)


ALL = [s12, s13, s14, s15, s16a, s16b, s17]
