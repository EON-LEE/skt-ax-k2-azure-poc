import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = REPORT_DIR / "figures"
INPUTS_FIG7 = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 120000]
# Azure ML low priority has been billed as Spot since 2026-03-31, so its old meter is left out of the cost outputs.
PRICE_METERS = ["on_demand", "reserved_1_year", "reserved_3_years", "spot"]
PRICE_LABELS_KO = {
    "on_demand": "종량제",
    "reserved_1_year": "1년 예약",
    "reserved_3_years": "3년 예약",
    "spot": "Spot",
    "low_priority": "Low priority",
}
EP_EVIDENCE = ROOT / "evidence" / "a100-expert-parallel-comparison.json"
# Points of the 2026-10-03/10-04 sweeps that reused earlier points' prompts from the prefix cache (one seed for all points).
TAINTED_INPUTS = (2048, 4096, 8192)
REVIEW_NOTE_KO = (
    "2026-11-09 검토에 따라 주의를 붙였다. 이 표의 A100 값은 2026-10-03(FP8, 영국 남부, InfiniBand)과 "
    "2026-10-04(NVFP4, 영국 남부, 노드 1대) 작업에서 나왔다. 두 작업은 모든 측정점에 같은 난수 시드를 써서, "
    "뒤 측정점이 앞 측정점의 프롬프트를 프리픽스 캐시에서 재사용했다. † 표시 값은 재측정 전까지 사용하지 않는다. "
    "자세한 내용은 [검토 보고서](../REVIEW_2026-11-09.md)에 있다."
)
LOGPROB_EVIDENCE = ROOT / "evidence" / "a100-native-2layer-logprob-series.json"
LOGPROB_GROUPS = {
    "native_tp8": ("Native TP8 (1노드)", "#1f77b4", "node0.smoke-native-tp8.series"),
    "native_tp8_pp2": ("Native TP8×PP2 (2노드)", "#d62728", "node0.smoke-native-tp8-pp2.series"),
}
LOGPROB_LENGTHS = {
    "ko_short": 12, "en_short": 14, "code": 15, "mixed_1500": 1171,
    "mixed_2600": 2076, "readme_tail": 4165, "readme_full": 7165,
}
LOGPROB_TITLE = "공식 구현 대비 대체 커널 확률 차이 (토큰 14,618개)"
LOGPROB_CAPTION = (
    "A.X K2 61개 레이어 중 앞 2개 레이어(실제 가중치), 프롬프트 7개, "
    "선택 토큰 log-prob 비교, 평균 |Δ| 약 0.011/0.012, "
    "전체 모델 정확도·공개 벤치마크 동등성 미측정"
)


def validate_logprob(data):
    if data["schema_version"] != 1:
        raise ValueError("Unsupported log-prob evidence schema")
    prompts = data["prompts"]
    if [p["name"] for p in prompts] != list(LOGPROB_LENGTHS):
        raise ValueError("Expected the seven prompts in length order")
    fields = {"oracle", *LOGPROB_GROUPS}
    for prompt in prompts:
        name = prompt["name"]
        if set(prompt) != {"name", *fields}:
            raise ValueError(f"Unexpected series fields for {name}")
        for field in fields:
            values = prompt[field]
            if not isinstance(values, list) or len(values) != LOGPROB_LENGTHS[name]:
                raise ValueError(f"Incorrect series length: {name}.{field}")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v > 0
                for v in values
            ):
                raise ValueError(f"Expected finite, nonpositive log-prob values: {name}.{field}")
    return data


def extract_logprob(oracle_path, native_path, out_path=LOGPROB_EVIDENCE):
    """Whitelist only the plotted numeric series, never raw report metadata."""
    paths = {"oracle": Path(oracle_path), "native": Path(native_path)}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    reports = {
        key: json.loads(value.decode("utf-8-sig"))["reports"]
        for key, value in raw.items()
    }
    specs = {"oracle": ("oracle", "oracle.series")}
    specs.update({key: ("native", spec[2]) for key, spec in LOGPROB_GROUPS.items()})
    sources = {
        key: {
            "file": "oracle-long-full.json" if report == "oracle" else "magenta_engine_vgjw1yh229.json",
            "series": series,
            "sha256": hashlib.sha256(raw[report]).hexdigest(),
        }
        for key, (report, series) in specs.items()
    }
    for key, (report, series) in specs.items():
        names = {
            name.removesuffix(".chosen_logprob")
            for name in reports[report][series] if name.endswith(".chosen_logprob")
        }
        if names != set(LOGPROB_LENGTHS):
            raise ValueError(f"Unexpected prompt set in {key}")
    data = {
        "schema_version": 1,
        "sources": sources,
        "prompts": [
            {
                "name": name,
                **{
                    key: reports[report][series][name + ".chosen_logprob"]
                    for key, (report, series) in specs.items()
                },
            }
            for name in LOGPROB_LENGTHS
        ],
    }
    validate_logprob(data)
    Path(out_path).write_text(
        json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return data


def load_logprob(path=LOGPROB_EVIDENCE):
    return validate_logprob(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def logprob_summary(data):
    validate_logprob(data)
    summary = {}
    for key in LOGPROB_GROUPS:
        inside, outside = [], []
        for prompt in data["prompts"]:
            delta = [abs(a - b) for a, b in zip(prompt["oracle"], prompt[key])]
            inside.extend(delta[:2048])
            outside.extend(delta[2048:])
        summary[key] = {
            "positions": len(inside) + len(outside),
            "within_first_2048": len(inside),
            "beyond_first_2048": len(outside),
            "mean_abs_delta": math.fsum(inside + outside) / (len(inside) + len(outside)),
            "within_mean_abs_delta": math.fsum(inside) / len(inside),
            "beyond_mean_abs_delta": math.fsum(outside) / len(outside),
        }
    return summary


def build_logprob_figure(data):
    import numpy as np
    from matplotlib import font_manager

    summary = logprob_summary(data)
    plt, _ = _import_matplotlib()
    # This figure must preserve its Korean title, not fall back to English/tofu.
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if family in fonts:
            plt.rcParams["font.family"] = family
            break
    else:
        raise RuntimeError("Install a Korean font (Malgun Gothic or Noto Sans CJK) to render this figure")
    oracle = np.concatenate([p["oracle"] for p in data["prompts"]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    means = []
    for key, (label, color, _) in LOGPROB_GROUPS.items():
        native = np.concatenate([p[key] for p in data["prompts"]])
        axes[0].scatter(oracle, native, s=4, alpha=0.3, color=color, label=label)
        delta = np.abs(oracle - native)
        axes[1].hist(
            np.clip(delta, 0, 0.1), bins=np.linspace(0, 0.1, 41),
            histtype="step", lw=2, color=color, label=label,
            linestyle="-" if key == "native_tp8" else "--",
        )
        means.append(f"{label}: 평균 |Δ| {summary[key]['mean_abs_delta']:.3f}")
    low = min(float(collection.get_offsets()[:, 1].min()) for collection in axes[0].collections)
    low = min(float(oracle.min()), low)
    axes[0].plot([low, 0], [low, 0], "k--", lw=1, label="같은 값")
    axes[0].set_xlabel("공식 구현(FP32) 선택 토큰 log-prob")
    axes[0].set_ylabel("대체 커널 선택 토큰 log-prob")
    axes[0].set_title("① 값 비교")
    axes[0].legend(loc="upper left", markerscale=3)
    axes[1].set_xlabel("|Δ log-prob| (0.1 초과는 마지막 구간에 합산)")
    axes[1].set_ylabel("토큰 위치 수 (실험군별 14,618개)")
    axes[1].set_title("② 차이 분포")
    axes[1].legend(loc="upper right")
    axes[1].text(
        0.97, 0.7, "\n".join(means), transform=axes[1].transAxes,
        ha="right", va="top", fontsize=9, bbox=dict(fc="w", ec="gray"),
    )
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(LOGPROB_TITLE, fontsize=14)
    fig.text(
        0.5, 0.045,
        LOGPROB_CAPTION.replace("선택 토큰 log-prob 비교, ", "선택 토큰 log-prob 비교,\n"),
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    return fig


def render_logprob(out_dir, data=None):
    fig = build_logprob_figure(load_logprob() if data is None else data)
    fig.savefig(Path(out_dir) / "logprob-two.png", dpi=150)
    from matplotlib import pyplot as plt
    plt.close(fig)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_verify_constants():
    path = ROOT / "aml" / "src" / "verify_suites.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"DOC_FIG7", "DOC_FIG8_TOTAL"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"Missing constants in {path}: {sorted(missing)}")
    return found


def _points_to_map(points, total_key="total_tok_s"):
    return {int(p["input_tokens"]): p for p in points}


def _series_from_pairs(name, label, hardware, gpu_count, source, conditions, pairs):
    return {
        "name": name,
        "label": label,
        "hardware": hardware,
        "gpu_count": gpu_count,
        "source": source,
        "conditions": conditions,
        "points": [
            {"input_tokens": int(k), "total_tok_s": float(v[0]), "output_tok_s": float(v[1])}
            for k, v in sorted(pairs.items())
        ],
    }


def load_throughput():
    constants = _extract_verify_constants()
    fp8 = _read_json(ROOT / "evidence" / "a100-verification-and-doc-speed.json")
    nvfp4 = _read_json(ROOT / "evidence" / "a100-nvfp4-single-node.json")
    published = _read_json(REPORT_DIR / "skt_published.json")

    doc_fig7 = constants["DOC_FIG7"]
    doc_fig8 = constants["DOC_FIG8_TOTAL"]
    series = {}
    series["b200_fp8_k2"] = _series_from_pairs(
        "b200_fp8_k2",
        "B200 A.X K2 FP8 (8 GPU)",
        "B200",
        8,
        "aml/src/verify_suites.py DOC_FIG7, SKT tech report Fig. 7",
        "random dataset, concurrency 32, 1,024 output tokens, FP8 weights, BF16 KV cache",
        doc_fig7,
    )
    series["b200_fp8_k2_eagle3_total"] = {
        "name": "b200_fp8_k2_eagle3_total",
        "label": "B200 A.X K2 FP8 + EAGLE3 (Fig. 8 total)",
        "hardware": "B200",
        "gpu_count": 8,
        "source": "aml/src/verify_suites.py DOC_FIG8_TOTAL, SKT tech report Fig. 8",
        "conditions": "same serving benchmark family; total throughput only",
        "points": [{"input_tokens": int(k), "total_tok_s": float(v)} for k, v in sorted(doc_fig8.items())],
    }

    for key, item in published["series"].items():
        series[key] = {
            "name": key,
            "label": key.replace("_", " "),
            "hardware": item["hardware"],
            "gpu_count": item["gpu_count"],
            "source": item["source"],
            "conditions": item["conditions"],
            "points": item["points"],
        }

    fig7_points = fp8["doc_speed_comparison"]["points"]
    series["a100_fp8_native"] = {
        "name": "a100_fp8_native",
        "label": "A100x16 FP8 native DSA (demo)",
        "hardware": "A100",
        "gpu_count": 16,
        "source": "evidence/a100-verification-and-doc-speed.json doc_speed_comparison.points[].native",
        "conditions": fp8["doc_speed_comparison"]["this_run"],
        "points": [
            {"input_tokens": p["input_tokens"], **p["native"]}
            for p in fig7_points
        ],
    }
    series["a100_fp8_dense"] = {
        "name": "a100_fp8_dense",
        "label": "A100x16 FP8 dense",
        "hardware": "A100",
        "gpu_count": 16,
        "source": "evidence/a100-verification-and-doc-speed.json doc_speed_comparison.points[].dense",
        "conditions": fp8["doc_speed_comparison"]["this_run"],
        "points": [
            {"input_tokens": p["input_tokens"], **p["dense"]}
            for p in fig7_points
        ],
    }

    nv_points = nvfp4["doc_fig7_conditions"]["points"]
    series["a100_nvfp4"] = {
        "name": "a100_nvfp4",
        "label": "A100x8 NVFP4",
        "hardware": "A100",
        "gpu_count": 8,
        "source": "evidence/a100-nvfp4-single-node.json doc_fig7_conditions.points[].nvfp4_8_a100",
        "conditions": nvfp4["doc_fig7_conditions"]["this_run"],
        "points": [
            {"input_tokens": p["input_tokens"], **p["nvfp4_8_a100"]}
            for p in nv_points
        ],
    }

    return {
        "schema_version": 1,
        "sources": {
            "verify_suites": "aml/src/verify_suites.py",
            "a100_fp8": "evidence/a100-verification-and-doc-speed.json",
            "a100_nvfp4": "evidence/a100-nvfp4-single-node.json",
            "skt_published": "docs/report/skt_published.json",
        },
        "series": series,
        "cost_per_million_output_tokens_usd": nvfp4["cost_per_million_output_tokens_usd"],
    }


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    korean = font_path.exists()
    if korean:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt, korean


def _fmt_input(value):
    if value == 120000:
        return "120K"
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return f"{value // 1000}K"


def _fmt_int(value):
    return f"{int(round(value)):,}"


def _point_map(data, key):
    return _points_to_map(data["series"][key]["points"])


def _total(point):
    return point.get("total_tok_s", point.get("total_token_throughput"))


def _output(point):
    return point.get("output_tok_s", point.get("output_throughput"))


def _line_plot(ax, xs, values, label, color, marker="o", linewidth=2.2, highlight=False, label_offset=(0, 7)):
    ax.plot(xs, values, marker=marker, label=label, color=color, linewidth=linewidth + (0.8 if highlight else 0), markersize=6)
    offsets = label_offset if isinstance(label_offset, list) else [label_offset] * len(values)
    for x, y, offset in zip(xs, values, offsets):
        va = "bottom" if offset[1] >= 0 else "top"
        ax.annotate(_fmt_int(y), (x, y), textcoords="offset points", xytext=offset, ha="center", va=va, fontsize=8, color=color)


def _save(fig, path):
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=155, bbox_inches="tight")
    fig.clf()


def render_fig7_overlay(data, out_dir):
    plt, ko = _import_matplotlib()
    inputs = INPUTS_FIG7
    xs = list(range(len(inputs)))
    labels = [_fmt_input(v) for v in inputs]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    specs = [
        ("b200_fp8_k2", "B200 A.X K2 FP8", "#1f77b4", "s", False, [(0, -11), (0, -11), (0, -11), (0, -11), (0, -11), (0, 8), (0, 8), (0, 8)]),
        ("b200_fp8_k1_fig7", "B200 A.X K1", "#7f7f7f", "o", False, [(0, 8), (0, 8), (0, 8), (0, 8), (0, 8), (0, -11), (0, -11), (0, -11)]),
        ("a100_fp8_native", "A100x16 FP8 native DSA (demo)", "#d62728", "D", True, (0, -11)),
        ("a100_fp8_dense", "A100x16 FP8 dense", "#ff7f0e", "^", False, (0, 8)),
    ]
    for key, label, color, marker, highlight, offset in specs:
        points = _point_map(data, key)
        _line_plot(ax, xs, [_total(points[i]) for i in inputs], label, color, marker, highlight=highlight, label_offset=offset)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("총 토큰 처리량 (tok/s)" if ko else "Total token throughput (tok/s)")
    ax.set_xlabel("입력 토큰" if ko else "Input tokens")
    ax.set_title("Fig. 7 조건 처리량: B200 공개값 vs A100 측정값" if ko else "Fig. 7 throughput: published B200 vs measured A100")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=2, fontsize=9)
    foot = ("조건: random dataset, concurrency 32, 출력 1,024 tokens, FP8/BF16 KV. 출처: SKT Fig. 7, verify_suites.py, evidence/*.json.\n"
            "2K~8K의 A100 값은 같은 시드로 앞 측정점의 프롬프트를 캐시에서 재사용해 부풀었다(재측정 전까지 사용하지 않음)."
            if ko else
            "Conditions: random dataset, concurrency 32, 1,024 output tokens, FP8/BF16 KV. Sources: SKT Fig. 7, verify_suites.py, evidence/*.json.\n"
            "A100 values at 2K-8K reused earlier points' prompts from the prefix cache (one seed); do not use until re-measured.")
    fig.text(0.01, 0.015, foot, fontsize=8, color="#444", va="bottom")
    _save(fig, out_dir / "fig7-overlay.png")


def render_fig9_overlay(data, out_dir):
    plt, ko = _import_matplotlib()
    b200 = _point_map(data, "b200_nvfp4_k2_fig9")
    a100 = _point_map(data, "a100_nvfp4")
    inputs = [i for i in sorted(a100) if i in b200]
    xs = list(range(len(inputs)))
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    _line_plot(ax, xs, [_total(b200[i]) for i in inputs], "B200 NVFP4 (4 GPU, Fig. 9)", "#1f77b4", "s")
    _line_plot(ax, xs, [_total(a100[i]) for i in inputs], "A100x8 NVFP4 (ours)", "#d62728", "D", highlight=True)
    ax.set_xticks(xs, [_fmt_input(v) for v in inputs])
    ax.set_ylabel("총 토큰 처리량 (tok/s)" if ko else "Total token throughput (tok/s)")
    ax.set_xlabel("입력 토큰" if ko else "Input tokens")
    ax.set_title("NVFP4 처리량: SKT Fig. 9 B200 vs A100 1노드" if ko else "NVFP4 throughput: SKT Fig. 9 B200 vs A100 one node")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(fontsize=9)
    foot = "Fig. 9 이미지는 GPU 수를 표시하지 않아 SERVING_DESIGN의 4 x B200(TP4+EP) 설명을 사용." if ko else "Fig. 9 image does not show GPU count; using SERVING_DESIGN assumption: 4 x B200 (TP4+EP)."
    fig.text(0.01, 0.015, foot, fontsize=8, color="#444")
    _save(fig, out_dir / "fig9-overlay.png")


def render_per_gpu(data, out_dir):
    plt, ko = _import_matplotlib()
    keys = ["b200_fp8_k2", "a100_fp8_native", "a100_fp8_dense", "a100_nvfp4", "b200_nvfp4_k2_fig9"]
    labels = ["B200 FP8 K2", "A100 FP8 native", "A100 FP8 dense", "A100 NVFP4", "B200 NVFP4"]
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#a23a3a", "#2ca02c"]
    common = set(_point_map(data, keys[0]))
    for key in keys[1:]:
        common &= set(_point_map(data, key))
    inputs = [i for i in [1024, 4096, 8192, 16384, 32768] if i in common]
    xbase = list(range(len(inputs)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for idx, (key, label, color) in enumerate(zip(keys, labels, colors)):
        points = _point_map(data, key)
        gpu_count = data["series"][key]["gpu_count"]
        vals = [_total(points[i]) / gpu_count for i in inputs]
        xs = [x + (idx - 2) * width for x in xbase]
        bars = ax.bar(xs, vals, width=width, label=label, color=color, alpha=0.9)
        ax.bar_label(bars, labels=[_fmt_int(v) for v in vals], fontsize=7, padding=2, rotation=90)
    ax.set_xticks(xbase, [_fmt_input(v) for v in inputs])
    ax.set_ylabel("GPU당 총 tok/s" if ko else "Total tok/s per GPU")
    ax.set_xlabel("입력 토큰" if ko else "Input tokens")
    ax.set_title("GPU당 처리량 비교" if ko else "Per-GPU throughput comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    foot = ("작업이 다른 값을 맞댔다: A100 FP8 2026-10-03, A100 NVFP4 2026-10-04, B200은 SKT 공개 그림 값. "
            "B200 NVFP4의 GPU 수 4장은 그림에 표시되지 않은 가정이다.\n"
            "A100 FP8(2026-10-03)과 NVFP4(2026-10-04)는 모든 측정점에 같은 시드를 써서 캐시 재사용이 섞였다. 처리량이 높게 나왔다"
            "(FP8 4K·8K가 특히 크다). 재측정 전까지 사용하지 않는다."
            if ko else
            "Values from different jobs: A100 FP8 2026-10-03, A100 NVFP4 2026-10-04, B200 from SKT's published figures. "
            "The B200 NVFP4 GPU count (4) is an assumption; the figure does not show it.\n"
            "Both A100 runs used one seed for every point, so prompts were reused from the prefix cache and throughput "
            "reads high (most for FP8 at 4K and 8K); do not use until re-measured.")
    fig.text(0.01, 0.015, foot, fontsize=8, color="#444", va="bottom")
    _save(fig, out_dir / "per-gpu-throughput.png")


def render_cost(data, out_dir):
    plt, ko = _import_matplotlib()
    cost = data["cost_per_million_output_tokens_usd"]
    configs = [
        ("concurrency_32_input_1k_output_1k", "nvfp4_1_node", "C32 1K/1K NVFP4 1 node", "#d62728"),
        ("concurrency_32_input_1k_output_1k", "fp8_2_nodes", "C32 1K/1K FP8 2 nodes", "#ff7f0e"),
        ("concurrency_128_input_1k_output_256", "nvfp4_1_node_batched_2048", "C128 1K/256 NVFP4 1 node", "#a23a3a"),
        ("concurrency_128_input_1k_output_256", "fp8_2_nodes_batched_2048", "C128 1K/256 FP8 2 nodes", "#1f77b4"),
    ]
    xbase = list(range(len(PRICE_METERS)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for idx, (workload, key, label, color) in enumerate(configs):
        vals = [cost[workload][key][m] for m in PRICE_METERS]
        xs = [x + (idx - 1.5) * width for x in xbase]
        bars = ax.bar(xs, vals, width=width, label=label, color=color, alpha=0.9)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2, rotation=90)
    ax.set_xticks(xbase, [PRICE_LABELS_KO[m] if ko else m.replace("_", " ") for m in PRICE_METERS])
    ax.set_ylabel("USD / 1M 출력 토큰" if ko else "USD per 1M output tokens")
    ax.set_title("출력 100만 토큰당 비용 (이전 값, 재측정 전까지 사용하지 않음)" if ko else "Cost per 1M output tokens (superseded; do not use until re-measured)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    foot = ("서로 다른 작업의 처리량으로 계산한 교차 비교이고, C128은 같은 시드로 인한 캐시 재사용이 섞였다. "
            "저우선 요율은 2026-03-31부터 Spot 과금이라 뺐다.\n식: " + cost["formula"]
            if ko else
            "Cross-job comparison; the C128 throughputs include prefix-cache reuse (one seed). The low-priority meter is left out: "
            "Azure ML bills it as Spot since 2026-03-31.\nFormula: " + cost["formula"])
    fig.text(0.01, 0.015, foot, fontsize=8, color="#444", va="bottom")
    _save(fig, out_dir / "cost-per-1m-output-tokens.png")


def write_throughput_json(data, out_dir):
    with (out_dir / "throughput.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _ratio(num, den):
    if den == 0 or den is None:
        return ""
    return f"{num / den:.2f}"


def _rerun_c32_output(path=EP_EVIDENCE):
    """Concurrency-32 output tok/s of the 2026-10-21 re-measurement (own seed per point) for FP8 baseline A and NVFP4 baseline D."""
    jobs = _read_json(Path(path))["jobs"]
    out = {}
    for name, key in (("fp8", "epc-fp8-tp8pp2"), ("nvfp4", "epc-nvfp4-tp8")):
        points = jobs[name]["configs"][key]["points"]
        out[name] = {isl: points[f"isl{isl}-c32"]["output_throughput"] for isl in (1024, 8192, 32768)}
    return out


def _mark(text, input_tokens):
    """Flag a value from a point that reused earlier prompts through the prefix cache."""
    return text + ("†" if input_tokens in TAINTED_INPUTS else "")


def write_tables(data, out_dir):
    rerun = _rerun_c32_output()
    lines = ["# 처리량 표", "", REVIEW_NOTE_KO, ""]
    b200 = _point_map(data, "b200_fp8_k2")
    native = _point_map(data, "a100_fp8_native")
    dense = _point_map(data, "a100_fp8_dense")
    lines += [
        "## Fig. 7 조건: FP8 총 tok/s",
        "",
        "| 입력 | B200 K2 FP8 | A100x16 native | native/B200 | A100x16 dense | dense/B200 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i in INPUTS_FIG7:
        bt, nt, dt = _total(b200[i]), _total(native[i]), _total(dense[i])
        lines.append(f"| {_fmt_input(i)} | {_fmt_int(bt)} | {_mark(_fmt_int(nt), i)} | {_mark(_ratio(nt, bt), i)} | "
                     f"{_mark(_fmt_int(dt), i)} | {_mark(_ratio(dt, bt), i)} |")
    clean = {i: rerun["fp8"][i] * (i + 1024) / 1024 / _total(b200[i]) for i in rerun["fp8"]}
    lines += ["", "†: 2K~8K는 앞 측정점의 프롬프트를 캐시에서 재사용해 부풀었다. 같은 native 구성을 측정점마다 다른 시드로 다시 잰 "
              f"2026-10-21 값으로 환산하면 B200 대비 1K {clean[1024]:.2f}, 8K {clean[8192]:.2f}, 32K {clean[32768]:.2f}다."]

    nv = _point_map(data, "a100_nvfp4")
    b200_nv = _point_map(data, "b200_nvfp4_k2_fig9")
    lines += ["", "## NVFP4: A100 측정값과 B200 공개값", "", "| 입력 | A100x8 NVFP4 | vs B200 FP8 Fig.7 | B200 NVFP4 Fig.9 | A100/B200 NVFP4 |", "| ---: | ---: | ---: | ---: | ---: |"]
    for i in sorted(nv):
        at = _total(nv[i])
        lines.append(f"| {_fmt_input(i)} | {_fmt_int(at)} | {_ratio(at, _total(b200[i]))} | {_fmt_int(_total(b200_nv[i]))} | {_ratio(at, _total(b200_nv[i]))} |")
    drift = {i: _output(nv[i]) / rerun["nvfp4"][i] - 1 for i in rerun["nvfp4"]}
    lines += ["", "NVFP4 값은 2026-10-04 작업이다. 같은 시드로 측정했고, 2026-10-21 재측정과의 동시 32 출력 차이는 "
              f"1K {drift[1024]:+.1%}, 8K {drift[8192]:+.1%}, 32K {drift[32768]:+.1%}다."]

    per_keys = ["b200_fp8_k2", "a100_fp8_native", "a100_fp8_dense", "a100_nvfp4", "b200_nvfp4_k2_fig9"]
    common = set(_point_map(data, per_keys[0]))
    for key in per_keys[1:]:
        common &= set(_point_map(data, key))
    per_inputs = [i for i in [1024, 4096, 8192, 16384, 32768] if i in common]
    lines += ["", "## GPU당 총 tok/s", "", "| 입력 | B200 FP8 K2 | A100 FP8 native | A100 FP8 dense | A100 NVFP4 | B200 NVFP4 |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for i in per_inputs:
        cells = [_fmt_int(_total(_point_map(data, key)[i]) / data["series"][key]["gpu_count"]) for key in per_keys]
        cells[1], cells[2] = _mark(cells[1], i), _mark(cells[2], i)
        lines.append(f"| {_fmt_input(i)} | " + " | ".join(cells) + " |")
    lines += ["", "열마다 작업이 다르다. A100 FP8 native·dense는 2026-10-03, A100 NVFP4는 2026-10-04 작업이고, B200은 SKT 공개 그림 값이다. "
              f"B200 NVFP4의 GPU 수({data['series']['b200_nvfp4_k2_fig9']['gpu_count']}장)는 그림에 표시되지 않은 가정이다"
              "(SERVING_DESIGN의 4 x B200 설명). †는 위 Fig. 7 표와 같다."]

    cost = data["cost_per_million_output_tokens_usd"]
    lines += ["", "## 출력 100만 토큰당 비용 (USD, 이전 값)", "",
              "| 워크로드 | 구성 | " + " | ".join(PRICE_LABELS_KO[m] for m in PRICE_METERS) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in PRICE_METERS) + " |"]
    rows = [
        ("C32, 1K in, 1K out", "NVFP4 1 node", cost["concurrency_32_input_1k_output_1k"]["nvfp4_1_node"]),
        ("C32, 1K in, 1K out", "FP8 2 nodes", cost["concurrency_32_input_1k_output_1k"]["fp8_2_nodes"]),
        ("C128, 1K in, 256 out", "NVFP4 1 node", cost["concurrency_128_input_1k_output_256"]["nvfp4_1_node_batched_2048"]),
        ("C128, 1K in, 256 out", "FP8 2 nodes", cost["concurrency_128_input_1k_output_256"]["fp8_2_nodes_batched_2048"]),
    ]
    for workload, config, row in rows:
        lines.append(f"| {workload} | {config} | " + " | ".join(f"{row[m]:.2f}" for m in PRICE_METERS) + " |")
    lines += ["", "이 표는 기록으로만 남기고 재측정 전까지 사용하지 않는다. FP8과 NVFP4 처리량은 서로 다른 작업에서 측정했다"
              "(C32 행: FP8 2026-10-03 영국 남부·InfiniBand, NVFP4 2026-10-04 영국 남부. C128 행: FP8 2026-10-03 이탈리아 북부·TCP, "
              "NVFP4 2026-10-04). C128 행의 두 처리량에는 같은 시드로 인한 캐시 재사용이 섞였다. "
              f"C32 행의 NVFP4 처리량은 2026-10-21 재측정보다 {drift[1024]:.1%} 높았다.",
              "",
              "Low priority 열은 뺐다. Azure ML 저우선 클러스터는 2026-03-31부터 Spot으로 할당·과금되므로 이전 저우선 요율"
              f"(노드·시간당 ${cost['prices_usd_per_node_hour']['low_priority']:.2f})은 적용되지 않는다."]
    with (out_dir / "throughput-tables.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


BENCHMARKS = ROOT / "evidence" / "a100-benchmarks.json"


VERIFICATION = ROOT / "evidence" / "a100-verification-and-doc-speed.json"


def _verification_niah(path):
    """NIAH from the earlier verification run of the serving layout (native DSA, TP8xPP2), if recorded."""
    path = Path(path)
    niah = (_read_json(path).get("per_phase", {}).get("native-pp2", {}).get("niah") if path.is_file() else None)
    if not niah or not niah.get("total"):
        return None
    return {"score": niah["hits"] / niah["total"], "hits": niah["hits"], "total": niah["total"],
            "lengths": sorted(int(k) for k in (niah.get("by_length") or {})), "source": "verification"}


def load_benchmarks(path=BENCHMARKS, niah_fallback=None):
    """Rows of the model card's Thinking Mode table, each with our A100 score when that suite was run.

    The evidence file is the eval run's final summary as posted by the job ({"run", "plan", "summary", ...});
    scores there are fractions, the card's are percent. A run stopped early has n_items < items_expected,
    which the table shows. When the run has no NIAH, niah_fallback (the verification evidence) supplies it.
    """
    published = _read_json(REPORT_DIR / "skt_published.json")["benchmarks"]
    ours = _read_json(Path(path)) if Path(path).is_file() else {}
    summary = ours.get("summary") or {}
    rows = []
    for row in published["rows"]:
        entry = summary.get(row["suite"]) if row["suite"] else None
        score = entry.get("score") if entry else None
        half = entry.get("ci95_half") if entry else None
        rows.append({**row, "a100": None if score is None else round(100 * score, 1),
                     "a100_ci95": None if half is None else round(100 * half, 1),
                     "n_items": entry.get("n_items") if entry else None,
                     "items_expected": entry.get("items_expected") if entry else None,
                     "n_generations": entry.get("n_generations") if entry else None,
                     "generations_expected": entry.get("generations_expected") if entry else None,
                     "repeats": entry.get("repeats") if entry else None})
    niah = summary.get("niah") or (_verification_niah(niah_fallback) if niah_fallback else None)
    niah_row = {"benchmark": "NIAH", "skt": published["niah"]["skt"], "note": published["niah"]["note"],
                "a100": None if not niah or niah.get("score") is None else round(100 * niah["score"], 1),
                "grid": niah.get("grid") if niah else None,
                "hits": niah.get("hits") if niah else None, "total": niah.get("total") if niah else None,
                "lengths": niah.get("lengths") if niah else None, "source": niah.get("source") if niah else None}
    return {"run": ours.get("run"), "rows": rows, "niah": niah_row, "reasons": published["reasons"]}


def render_benchmarks(bench, out_dir):
    measured = [r for r in bench["rows"] if r["a100"] is not None]
    if bench["niah"]["a100"] is not None:
        lengths = bench["niah"].get("lengths") or [8192, 262144]
        measured.append({"benchmark": f"NIAH {lengths[0] // 1024}K-{lengths[-1] // 1024}K", "skt": bench["niah"]["skt"], "a100": bench["niah"]["a100"],
                         "a100_ci95": None})
    if not measured:
        return False
    plt, ko = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.subplots_adjust(bottom=0.24)
    xs = list(range(len(measured)))
    width = 0.36
    skt = ax.bar([x - width / 2 for x in xs], [r["skt"] for r in measured], width, label="SKT 공개값 (모델 카드)" if ko else "SKT published (model card)", color="#1f77b4")
    ours = ax.bar([x + width / 2 for x in xs], [r["a100"] for r in measured], width,
                  yerr=[r.get("a100_ci95") or 0 for r in measured], capsize=4,
                  label="A100 x16 FP8 native DSA (이번 측정)" if ko else "A100 x16 FP8 native DSA (ours)", color="#d62728")
    ax.bar_label(skt, labels=[f"{r['skt']:.1f}" for r in measured], fontsize=8, padding=2)
    ax.bar_label(ours, labels=[f"{r['a100']:.1f}" for r in measured], fontsize=8, padding=2)
    ax.set_xticks(xs, [r["benchmark"] + _sample_note(r).replace(" (", "\n(") for r in measured])
    ax.set_ylim(0, 110)
    ax.set_ylabel("점수 (%)" if ko else "Score (%)")
    ax.set_title("모델 카드 벤치마크: SKT 공개값 vs A100 16장" if ko else "Model card benchmarks: SKT published vs A100 x16")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    foot = "Thinking 모드, 공개 데이터·공개 채점. 오차 막대: 95% 신뢰구간. n=완료/계획(조기 중지, 참고치). NIAH: thinking 끔, 그리디." if ko else "Thinking mode, public data and scoring. Error bars: 95% CI. NIAH: thinking off, greedy."
    fig.text(0.01, 0.015, foot, fontsize=8, color="#444")
    _save(fig, out_dir / "benchmarks-vs-published.png")
    return True


def _sample_note(r):
    """'n=109/200' for a suite stopped before all items, or generations for repeated suites."""
    if r.get("generations_expected") and r.get("repeats", 1) and r["repeats"] > 1:
        done, want, unit = r.get("n_generations"), r["generations_expected"], " gen"
    else:
        done, want, unit = r.get("n_items"), r.get("items_expected"), ""
    return f" (n={done}/{want}{unit})" if done is not None and want and done < want else ""


def write_benchmark_table(bench, out_dir):
    """The model card's Thinking Mode table with one extra column, ready to paste into a card or report."""
    lines = ["# 모델 카드 Thinking Mode 표 + A100 측정값", "",
             "| 분야 | 벤치마크 | A.X K2 (SKT 공개값) | A.X K2, A100 16장 (이번 측정) |", "| --- | --- | ---: | ---: |"]
    partial = False
    for r in bench["rows"]:
        if r["a100"] is not None:
            note = _sample_note(r)
            partial = partial or bool(note)
            ours = f"**{r['a100']:.1f}**" + (f" ± {r['a100_ci95']:.1f}" if r["a100_ci95"] else "") + note
        elif r["suite"]:
            ours = "측정 전"
        else:
            ours = f"– ({r['not_run']})"
        lines.append(f"| {r['domain']} | {r['benchmark']} | {r['skt']:g} | {ours} |")
    niah = bench["niah"]
    if niah["a100"] is None:
        ours = "측정 전"
    else:
        ours = f"**{niah['a100']:.1f}**"
        if niah.get("total"):
            span = ", ".join(f"{n // 1024}K" for n in niah.get("lengths") or [])
            ours += f" ({niah['hits']}/{niah['total']}" + (f", {span}" if span else "") + ")"
    lines.append(f"| Long context | NIAH | {niah['skt']:g} | {ours} |")
    lines += ["", "실행하지 않은 이유: " + "; ".join(f"{k} = {v}" for k, v in bench["reasons"].items())]
    if bench["run"]:
        lines.append(f"A100 실행: `{bench['run']}`. ±는 문항 기준 95% 신뢰구간의 절반 폭이다.")
    if partial:
        lines.append("n=완료/계획: 실행을 중간에 멈췄다. 점수는 그때까지 끝난 문항(각 세트에서 무작위로 뽑은 문항)만으로 "
                     "계산했으므로 참고치로만 본다.")
    if niah.get("source") == "verification":
        lines.append("NIAH: 같은 서빙 구성으로 먼저 돌린 검증 실행의 값이다"
                     "(SKT niah_test.py, thinking 끔, 그리디).")
    with (out_dir / "benchmark-table.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def render_all(out_dir=DEFAULT_OUT, only="throughput"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if only not in ("all", "throughput", "eval", "logprob"):
        raise ValueError(f"Unsupported group: {only}")
    data = None
    if only in ("all", "throughput"):
        data = load_throughput()
        render_fig7_overlay(data, out_dir)
        render_fig9_overlay(data, out_dir)
        render_per_gpu(data, out_dir)
        render_cost(data, out_dir)
        write_throughput_json(data, out_dir)
        write_tables(data, out_dir)
    if only in ("all", "eval"):
        bench = load_benchmarks(niah_fallback=VERIFICATION)
        render_benchmarks(bench, out_dir)
        write_benchmark_table(bench, out_dir)
    if only in ("all", "logprob"):
        render_logprob(out_dir)
    return data

def main(argv=None):
    parser = argparse.ArgumentParser(description="Render SKT/A.X K2 report figures from local evidence.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory (default: docs/report/figures)")
    parser.add_argument("--only", default="throughput", choices=["throughput", "eval", "logprob", "all"], help="Figure group to render")
    parser.add_argument(
        "--logprob-sources", nargs=2, type=Path, metavar=("ORACLE_JSON", "NATIVE_JSON"),
        help="Extract whitelisted log-prob evidence from local original reports before rendering",
    )
    args = parser.parse_args(argv)
    if args.logprob_sources:
        if args.only != "logprob":
            parser.error("--logprob-sources requires --only logprob")
        extract_logprob(*args.logprob_sources)
    render_all(Path(args.out), args.only)


if __name__ == "__main__":
    main()
