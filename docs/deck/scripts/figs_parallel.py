"""Slides 5-8: the four parallelism ideas (TP, PP, EP, DP), each with headline numbers from the EP round."""
from matplotlib.patches import Rectangle

from common import *
import data as D

GREY2 = "#9AA5B1"


def pct(r):
    return f"{(r - 1) * 100:+.0f}%"


def tpot(code, isl):
    return D.point(code, isl, 1)["median_tpot_ms"]


def sect(text):
    st = {"b": True, "fc": PALE_BLUE, "c": NAVY}
    return [(text, st), ("", st), ("", st), ("", st)]


def s05():
    fig, ax = slide(
        "TP: 한 층을 GPU 8장이 나눠 계산한다",
        "노드 안(NVLink)에서는 빠르다. 노드를 넘기면(InfiniBand) 층마다 걸리는 통신이 느려진다",
        "측정: FP8, 영국 남부, InfiniBand, 2026-10-21. 한 요청(c1)과 32개 동시(c32), 입력 1K/8K/32K, 1,024 토큰 생성, 조건당 1회. "
        "왼쪽 그림은 개념도이며 측정값이 아니다. 출처: tables/t01_layout_summary.csv, evidence/a100-expert-parallel-comparison.json",
    )
    label(ax, 3, 46.4, "개념도: TP8 (노드 1대, GPU 8장)", 14.5, INK, True)
    box(ax, 3, 38.6, 46, 4.8, "층 n 입력: 모든 GPU가 같은 토큰 상태를 가진다", fc=PALE_BLUE, ec=BLUE, fs=12.5)
    gw, gp = 5.0, 0.6
    x0 = 3 + (46 - (8 * gw + 7 * gp)) / 2
    for i in range(8):
        gx = x0 + i * (gw + gp)
        box(ax, gx, 25.6, gw, 8.8, f"GPU\n{i}\n가중치\n1/8", fc=PALE_GREEN, ec=GREEN, fs=11, pad=0.3, radius=0.5)
        arrow(ax, gx + gw / 2, 38.5, gx + gw / 2, 34.7, BLUE, lw=1.6, ms=10)
        arrow(ax, gx + gw / 2, 25.5, gx + gw / 2, 22.9, GREEN, lw=1.6, ms=10)
    box(ax, 3, 17.6, 46, 5.2, "all-reduce: GPU 8장의 부분 결과를 합친다 (NVLink)", fc=PALE_AMBER, ec=ORANGE, fs=12.5)
    arrow(ax, 26, 17.5, 26, 14.9, NAVY, lw=2, ms=12)
    box(ax, 3, 10.0, 46, 4.8, "층 n+1 입력", fc=PALE_BLUE, ec=BLUE, fs=12.5)
    for i, ln in enumerate(wrap("층마다 이 합산이 필요하다. 그래서 TP는 NVLink가 있는 노드 안에서 쓰는 것이 기본이다.", 46, 12.5)):
        label(ax, 3, 8.0 - i * pitch(12.5), ln, 12.5, GREY)

    label(ax, 53, 46.4, "같은 GPU 16장을 TP16으로 묶으면 (측정)", 14.5, INK, True)
    cols = [("지표", 18.3, "l"), ("TP8×PP2 (기준)", 11.6, "c"), ("TP16", 7.6, "c"), ("변화", 6.5, "c")]
    rows = [sect("한 요청(c1) · TPOT")]
    for isl in D.SIZES:
        a, b = tpot("A", isl), tpot("B0", isl)
        rows.append([f"입력 {D.SZ[isl]}", f"{a:.1f} ms", f"{b:.1f} ms", (pct(b / a), {"c": VERM, "b": True})])
    rows.append(sect("32개 동시(c32) · 출력 tok/s"))
    for isl in D.SIZES:
        a, b = D.out("A", isl, 32), D.out("B0", isl, 32)
        rows.append([f"입력 {D.SZ[isl]}", f"{a:,.0f}", f"{b:,.0f}", (f"×{b / a:.2f}", {"c": GREEN if b >= a else VERM, "b": True})])
    rows.append(sect("용량"))
    ka, kb = D.kv_list("A")[0], D.kv_list("B0")[0]
    rows.append(["KV 캐시 토큰", f"{ka:,}", f"{kb:,}", (f"×{kb / ka:.2f}", {"c": VERM, "b": True})])
    bottom = table(ax, 53, 44.2, cols, rows, fs=12, hfs=12, pady=0.5)
    y = bottom - 1.6
    for ln in wrap("TPOT: 토큰 1개를 만드는 데 걸리는 시간(중앙값). 낮을수록 좋다.", 44, 11.5):
        label(ax, 53, y, ln, 11.5, INK)
        y -= pitch(11.5)
    y -= 0.5
    for ln in wrap("원인(로그와 구조에서 읽은 것, 프로파일링 값은 아님): 노드를 넘는 TP는 층마다 all-reduce가 IB를 건너고, "
                   "그 그룹에서는 custom all-reduce가 꺼진다. KV는 TP 랭크마다 복제되고 랭크가 61층 전체를 맡아 용량이 0.46배다.", 44, 11.5):
        label(ax, 53, y, ln, 11.5, GREY)
        y -= pitch(11.5)
    save(fig, "s05_tp.png")


def s06():
    fig, ax = slide(
        "PP: 층을 둘로 나눠 노드 두 대에 올린다",
        "속도를 위한 것이 아니다. FP8 가중치 694 GB가 노드 1대(640 GB)에 들어가지 않아서 쓴다",
        "개념도는 측정값이 아니다. 측정: FP8 TP8×PP2(기준 A), 영국 남부, InfiniBand, 2026-10-21, 입력 1K, 한 요청. "
        "연결 속도는 NCCL 측정(GPU 쌍당)이다. 출처: docs/report/REPORT.md, evidence/a100-expert-parallel-comparison.json",
    )
    label(ax, 3, 46.4, "개념도: TP8×PP2 (노드 2대, GPU 16장)", 14.5, INK, True)
    box(ax, 3, 28.5, 21, 14, "단계 0 · 노드 0\n층 0–30\nTP8 (NVLink)", fc=PALE_GREEN, ec=GREEN, fs=13)
    box(ax, 31, 28.5, 21, 14, "단계 1 · 노드 1\n층 31–60 + LM head\nTP8 (NVLink)", fc=PALE_GREEN, ec=GREEN, fs=13)
    arrow(ax, 24.1, 35.5, 30.9, 35.5, NAVY, lw=2.6, ms=16)
    label(ax, 27.5, 39.6, "은닉 상태", 11.5, NAVY, True, ha="center")
    label(ax, 27.5, 38.0, "+ 잔차", 11.5, NAVY, True, ha="center")
    label(ax, 27.5, 32.8, "InfiniBand", 11.5, NAVY, ha="center")
    card(ax, 3, 13.0, 23.5, 12.8, "왜 나누나: 메모리",
         "FP8 가중치 694 GB는 노드 1대의 640 GB보다 크다. 노드 2대(1,280 GB)에 층을 나눠 올린다.", ec=NAVY, fs=12.5)
    card(ax, 28.5, 13.0, 23.5, 12.8, "노드 사이로 오가는 것",
         "층 경계의 은닉 상태와 잔차뿐이다. 경계가 하나뿐이라 스텝마다 한 번 건넌다. 연결 속도(NCCL 측정, GPU 쌍당): InfiniBand 약 20 GB/s, TCP 약 0.3 GB/s.", ec=NAVY, fs=12)

    label(ax, 56, 46.4, "개념도: 요청 수에 따라 두 단계가 일하는 모습", 14.5, INK, True)
    slot, sx = 4.0, 66.0
    A_, B_ = BLUE, ORANGE

    def lane(y, name, seq):
        label(ax, 56, y + 1.3, name, 12, INK, True)
        for i, who in enumerate(seq):
            if who:
                box(ax, sx + i * (slot + 0.3), y, slot, 2.6, who, fc=A_ if who == "A" else B_, ec="none", fs=11.5, tc="white", bold=True, radius=0.3)
            else:
                box(ax, sx + i * (slot + 0.3), y, slot, 2.6, "", fc="white", ec=MID, ls="--", lw=1.0, radius=0.3)

    label(ax, 56, 42.0, "요청 1개", 13, NAVY, True)
    lane(38.4, "단계 0", ["A", None, "A", None, "A", None])
    lane(35.0, "단계 1", [None, "A", None, "A", None, "A"])
    label(ax, 56, 32.6, "디코딩은 토큰을 하나씩 만들어서 한 번에 한 단계만 일한다.", 12, GREY)
    label(ax, 56, 28.6, "요청 여러 개", 13, NAVY, True)
    lane(25.0, "단계 0", ["A", "B", "A", "B", "A", "B"])
    lane(21.6, "단계 1", [None, "A", "B", "A", "B", "A"])
    for i, ln in enumerate(wrap("단계 0이 다음 묶음을 처리하는 동안 단계 1이 이전 묶음을 처리해, 두 단계가 겹쳐 일한다. A, B는 서로 다른 요청이다.", 41, 12)):
        label(ax, 56, 19.2 - i * pitch(12), ln, 12, GREY)
    a1 = tpot("A", 1024)
    card(ax, 56, 6.6, 41, 8.6, "측정 (기준 A, 입력 1K)",
         f"한 요청 TPOT {a1:.1f} ms · KV 캐시 {D.kv_list('A')[0]:,} 토큰 · 최대 컨텍스트 {D.max_ctx('A'):,} · 서버 시작까지 {D.healthy('A')}초",
         ec=NAVY, fs=12)
    save(fig, "s06_pp.png")


def ep_grid(ax, x0, y_top, cell, ep, title, sub):
    per = 256 // ep
    rows_per = per // 16
    tones = [BLUE, SKY]
    label(ax, x0, y_top + 3.3, title, 13.5, NAVY, True)
    label(ax, x0, y_top + 1.6, sub, 11.5, GREY)
    for e in range(256):
        r_, c_ = divmod(e, 16)
        ax.add_patch(Rectangle((x0 + c_ * cell, y_top - (r_ + 1) * cell), cell * 0.88, cell * 0.88, fc=tones[(e // per) % 2], ec="none", zorder=3))
    for k in range(ep):
        yc = y_top - (k * rows_per + rows_per / 2) * cell
        label(ax, x0 + 16 * cell + 0.5, yc, f"GPU {k}" if ep == 8 else f"{k}", 11 if ep == 8 else 9.5, GREY)
    if ep == 16:
        label(ax, x0 + 16 * cell + 0.5, y_top + 0.5, "GPU", 9.5, GREY)


def s07():
    fig, ax = slide(
        "EP: 전문가(expert)를 GPU에 나눠 담는다",
        "라우팅되는 전문가 256개를 어떻게 나눌지의 문제다. EP 크기는 병렬 구성에서 정해진다",
        "개념도는 측정값이 아니다. EP 규칙은 vLLM 0.23 소스를 읽은 결과다. 측정: 같은 구성에서 EP만 켠 값 ÷ 끈 값(TP16+EP16÷TP16, TP8×PP2+EP8÷TP8×PP2, NVFP4 TP8+EP8÷NVFP4 TP8), "
        "c1과 c32, 입력 1K/8K/32K, 조건당 1회. 출처: evidence/a100-expert-parallel-comparison.json (how_vllm_0_23_forms_ep, findings)",
    )
    cell = 1.15
    ep_grid(ax, 3.5, 40.4, cell, 8, "EP8 · GPU당 전문가 32개", "행 2개 = GPU 1장")
    ep_grid(ax, 30.5, 40.4, cell, 16, "EP16 · GPU당 전문가 16개", "행 1개 = GPU 1장")
    y = 40.4 - 16 * cell - 1.2
    for ln in wrap("한 칸이 전문가 1개이고, 한 MoE 층의 전문가 256개(번호 0~255, 왼쪽에서 오른쪽, 위에서 아래)를 나눈 모습이다. 색이 같은 칸은 같은 GPU가 가진다.", 50, 11.5):
        label(ax, 3.5, y, ln, 11.5, GREY)
        y -= pitch(11.5)

    label(ax, 56, 46.4, "EP 크기 = DP × PCP × TP (파이프라인 단계 안에서)", 14.5, INK, True, max_w=41)
    for i, ln in enumerate(wrap("--enable-expert-parallel은 켜는 스위치일 뿐이고, 크기를 지정하는 옵션은 없다.", 41, 12.5)):
        label(ax, 56, 43.6 - i * pitch(12.5), ln, 12.5, INK)
    rows = []
    for code, name in (("B", "TP16+EP16"), ("C1", "TP8×PP2+EP8"), ("C2", "DP2×TP8+EP16"), ("E", "NVFP4 TP8+EP8")):
        pl = D.cfg(code)["parallel"]["expert_placement"][0]
        rows.append([name, D.EP_SIZE[code], str(pl["local_experts"])])
    bottom = table(ax, 56, 39.6, [("구성", 17, "l"), ("EP 크기", 13, "c"), ("GPU당 전문가", 11, "c")], rows, fs=12.5, hfs=12, pady=0.6)
    for i, ln in enumerate(wrap("DeepEP는 SM90·SM100용(9.0a·10.0a)으로만 빌드되어, A100(SM80)용 커널은 없다.", 41, 11.5)):
        label(ax, 56, bottom - 1.6 - i * pitch(11.5), ln, 11.5, GREY)

    pairs = (("B", "B0"), ("C1", "A"), ("E", "D"))
    allr = [D.out(a, s, c) / D.out(b, s, c) for a, b in pairs for s in D.SIZES for c in (1, 32)]
    assert max(allr) < 1.0
    lo, hi = (1 - max(allr)) * 100, (1 - min(allr)) * 100
    tp = [tpot(a, s) / tpot(b, s) for a, b in pairs for s in D.SIZES]
    tlo, thi = (min(tp) - 1) * 100, (max(tp) - 1) * 100
    w, ch, cy = 30.5, 10.4, 6.2
    card(ax, 3, cy, w, ch, "옮겨지는 것은 라우팅 전문가뿐",
         "라우팅 전문가 256개만 GPU에 나눠 담고 어텐션은 그대로다. 공유 전문가는 DP=1이면 TP 배치 그대로, DP2에서는 GPU마다 복제된다.", fs=12)
    card(ax, 3 + w + 1.5, cy, w, ch, "DP가 1이면 토큰을 주고받지 않는다",
         "각 GPU가 자기 전문가로 모든 토큰을 계산하고, 합계를 TP 그룹에서 all-reduce한다. EP를 안 켰을 때와 같은 통신이다.", fs=12)
    card(ax, 3 + 2 * (w + 1.5), cy, w, ch, "측정: 켜도 빨라지지 않았다",
         f"EP 켬/끔 3쌍(각 1회)에서 출력 tok/s가 {lo:.0f}~{hi:.0f}% 줄고(18개 조건 모두 1.0 미만), 한 요청 TPOT가 {tlo:.0f}~{thi:.0f}% 늘었다.",
         ec=VERM, hc=VERM, fs=12)
    save(fig, "s07_ep.png")


def s08():
    fig, ax = slide(
        "DP: 어텐션을 두 벌 두고 요청을 나눠 받는다",
        "DP2×TP8+EP16: 노드마다 TP8 복제본 1개. MoE 층에서는 두 노드가 전문가를 나눠 가진다",
        "개념도는 측정값이 아니다. 측정: FP8, 영국 남부, InfiniBand, 2026-10-21, 기준 A(TP8×PP2) 대비, 조건당 1회. 해석 상자는 측정이 아닌 추정이다(전문가·복제본별 부하는 측정하지 않았다). "
        "출처: tables/t01_layout_summary.csv, evidence/a100-expert-parallel-comparison.json (interpretation_not_measured)",
    )
    kv = D.kv_list("C2")
    label(ax, 3, 46.4, "개념도: DP2×TP8+EP16 (노드 2대, GPU 16장)", 14.5, INK, True)
    box(ax, 3, 38.6, 56, 4.6, "요청이 들어오면 두 복제본이 절반씩 나눠 받는다", fc=PALE_BLUE, ec=BLUE, fs=12.5)
    arrow(ax, 16, 38.5, 16, 36.4, BLUE, lw=2, ms=12)
    arrow(ax, 46, 38.5, 46, 36.4, BLUE, lw=2, ms=12)
    box(ax, 3, 24.0, 26, 12.2, f"복제본 0 · 노드 0\n어텐션 TP8 (NVLink)\nKV 캐시 {kv[0]:,} 토큰", fc=PALE_GREEN, ec=GREEN, fs=12.5)
    box(ax, 33, 24.0, 26, 12.2, f"복제본 1 · 노드 1\n어텐션 TP8 (NVLink)\nKV 캐시 {kv[1]:,} 토큰", fc=PALE_GREEN, ec=GREEN, fs=12.5)
    arrow(ax, 16, 23.9, 16, 21.4, NAVY, lw=2.2, ms=14, style="<|-|>")
    arrow(ax, 46, 23.9, 46, 21.4, NAVY, lw=2.2, ms=14, style="<|-|>")
    box(ax, 3, 9.6, 56, 11.6, "MoE 층마다 (두 노드 사이 InfiniBand)\n토큰 모으기(gather) → 전문가 계산 → 나눠 주기(scatter)\n전문가 256개를 GPU 16장이 16개씩 나눠 가진다 (EP16)",
        fc=PALE_AMBER, ec=ORANGE, fs=12.5)

    label(ax, 62, 46.4, "측정: 기준 A(TP8×PP2) 대비", 14.5, INK, True)
    t = [D.tpot_ratio("C2", s) for s in D.SIZES]
    o1 = [D.ratio("C2", s, 1) for s in D.SIZES]
    o32 = [D.ratio("C2", s, 32) for s in D.SIZES]
    ttft = [D.point("C2", s, 32)["median_ttft_ms"] / D.point("A", s, 32)["median_ttft_ms"] for s in D.SIZES]
    bad, good, mixed = {"c": VERM, "b": True}, {"c": GREEN, "b": True}, {"b": True}
    rows = [
        ["한 요청 TPOT", (" / ".join(f"×{x:.2f}" for x in t), bad)],
        ["한 요청 출력 tok/s", (" / ".join(f"×{x:.2f}" for x in o1), bad)],
        ["32개 동시 출력 tok/s", (" / ".join(f"×{x:.2f}" for x in o32), good)],
        ["32개 동시 TTFT", (" / ".join(f"×{x:.2f}" for x in ttft), mixed)],
        ["최대 컨텍스트", f"{D.max_ctx('C2'):,} (A: {D.max_ctx('A'):,})"],
        ["GPU당 가중치", f"{D.cfg('C2')['parallel']['model_loading_gib']['node0'][0]:.1f} GiB (A: {D.cfg('A')['parallel']['model_loading_gib']['node0'][0]:.1f})"],
    ]
    bottom = table(ax, 62, 44.2, [("지표", 15.5, "l"), ("입력 1K / 8K / 32K 순", 19.5, "l")], rows, fs=12, hfs=12, pady=0.55)
    card(ax, 62, 9.6, 35, bottom - 1.6 - 9.6, "해석 (측정 아님)",
         "요청이 많으면 어텐션이 NVLink 위 TP8 두 그룹으로 요청을 절반씩 나눠 처리해 유리하다. 요청이 1개면 MoE 층마다 IB를 건너 모으고 나누는 비용을 치르고, 쉬는 복제본도 같은 박자로 빈 스텝을 돈다. "
         "A와의 비교에는 API 서버 2개·배치 한도 합계 2배도 섞였다.",
         ec=GREY, hc=GREY, fs=12)
    save(fig, "s08_dp.png")


ALL = [s05, s06, s07, s08]
