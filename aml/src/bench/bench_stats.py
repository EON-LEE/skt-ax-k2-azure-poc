"""Statistics of the pre-registered serving protocol (slo_protocol.json, docs/BENCHMARK_PROTOCOL.md).

  python3 bench_stats.py package <job outputs folder> [--out package.json] [--report]
      Inside the workspace (collect-slo-outputs.yml): one compact, sanitized package per job from the raw
      outputs: per point the gates, the steady-state summary and per-request data, per-engine counter deltas,
      GPU telemetry and InfiniBand counters of every node over the point; parity against A of the same round;
      the plans, phase results, effective configs, NCCL and clean-state records, load-balancer stats.
  python3 bench_stats.py analyze <package.json>... --out evidence.json --md summary.md
      Operator side: round validity, lambda* with the bracket gate, cost per 1M output tokens, paired log-ratio
      t-CIs with A/A2 drift correction where A2 measured the metric, decision labels, the A/A noise floor and
      delta_min, the round-extension rule, P99 gating, parity acceptance, soak, D x 2 scaling, diagnostics.

Standard library only; every step is a function that the tests feed with synthetic data.
"""
import argparse
import base64
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "slo_protocol.json"
LN_105 = math.log(1.05)
SANITIZE = ((re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"), (re.compile(r"/mnt/azureml/cr/j/[^/\s\"']+"), "<job>"))
POINT_STATS = ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_p50", "tpot_p90", "tpot_p99", "e2e_p50", "e2e_p90",
               "itl_p50", "itl_p99", "output_tok_s")
CANDIDATES = (("slo-C2", "slo-A"), ("slo-D", "slo-A"), ("slo-C2s", "slo-A"), ("slo-C2s", "slo-C2"))


def load_protocol(path=None):
    return json.loads(Path(path or PROTOCOL).read_text(encoding="utf-8"))


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sanitize(value, hosts=()):
    """No addresses, job folders or host names in anything that leaves the job."""
    text = value if isinstance(value, str) else json.dumps(value)
    for pattern, replacement in SANITIZE:
        text = pattern.sub(replacement, text)
    for index, host in enumerate(h for h in hosts if h):
        text = text.replace(host, f"<host{index}>")
    return text if isinstance(value, str) else json.loads(text)


def percentile(values, q):
    """numpy's default (linear) percentile."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------------------------------------------
# One point: per-request data, steady state, bootstrap

def request_rows(bench):
    """Per-request rows of a --save-detailed `vllm bench serve` result; times in seconds from the first start."""
    starts = bench.get("start_times") or []
    if not starts:
        return []
    first, rows = min(starts), []
    ttfts, itls = bench.get("ttfts") or [], bench.get("itls") or []
    outputs, inputs, errors = bench.get("output_lens") or [], bench.get("input_lens") or [], bench.get("errors") or []
    for index, start in enumerate(starts):
        ttft = ttfts[index] if index < len(ttfts) else 0.0
        itl = (itls[index] if index < len(itls) else None) or []
        n_out = outputs[index] if index < len(outputs) else 0
        ok = not (errors[index] if index < len(errors) else "") and bool(ttft)
        latency = ttft + sum(itl)
        rows.append({"start": start - first, "ttft": ttft if ok else None, "e2e": latency if ok else None,
                     "tpot": (latency - ttft) / (n_out - 1) if ok and n_out > 1 else None, "ok": ok, "n_out": n_out,
                     "prompt_len": inputs[index] if index < len(inputs) else 0, "itl": list(itl)})
    return rows


def token_times(row):
    """Times of a request's output tokens: start + TTFT + cumulative ITL, spread evenly when chunks carried
    several tokens."""
    if not row["ok"] or not row["n_out"]:
        return []
    first = row["start"] + row["ttft"]
    if len(row["itl"]) == row["n_out"] - 1:
        times, now = [first], first
        for gap in row["itl"]:
            now += gap
            times.append(now)
        return times
    last = row["start"] + row["e2e"]
    count = row["n_out"]
    return [first + (last - first) * i / (count - 1) for i in range(count)] if count > 1 else [first]


def meets(row, slo):
    return row["ok"] and row["ttft"] * 1000 <= slo["ttft_ms"] and (row["tpot"] is None or row["tpot"] * 1000 <= slo["tpot_ms"])


def steady_state(rows, window=None, slos=None, duration=None, p99_min=1000):
    """Summary of the requests that started in the window (every request without a window). Output tok/s
    counts the token timestamps inside the window; without one it is output tokens / duration."""
    if window:
        low, high = window
        chosen = [r for r in rows if low <= r["start"] < high]
        span = high - low
        tokens = sum(1 for r in rows for t in token_times(r) if low <= t < high)
        prompt = sum(r["prompt_len"] for r in chosen)
    else:
        chosen, span = rows, duration or max([r["start"] + (r["e2e"] or 0) for r in rows] or [0])
        tokens, prompt = sum(r["n_out"] for r in rows if r["ok"]), sum(r["prompt_len"] for r in rows if r["ok"])
    ok = [r for r in chosen if r["ok"]]
    ms = lambda values, q: None if percentile(values, q) is None else round(percentile(values, q) * 1000, 3)
    ttft, tpot, e2e = [r["ttft"] for r in ok], [r["tpot"] for r in ok if r["tpot"] is not None], [r["e2e"] for r in ok]
    itl = [gap for r in ok for gap in r["itl"]]
    summary = {"requests": len(chosen), "failed": len(chosen) - len(ok), "window": window,
               "ttft_p50": ms(ttft, 50), "ttft_p90": ms(ttft, 90), "ttft_p99": ms(ttft, 99),
               "tpot_p50": ms(tpot, 50), "tpot_p90": ms(tpot, 90), "tpot_p99": ms(tpot, 99),
               "e2e_p50": ms(e2e, 50), "e2e_p90": ms(e2e, 90), "itl_p50": ms(itl, 50), "itl_p99": ms(itl, 99),
               "output_tok_s": round(tokens / span, 3) if span else None,
               "input_tok_s": round(prompt / span, 3) if span else None, "p99_reportable": len(chosen) >= p99_min}
    summary["goodput"] = {tier: round(sum(meets(r, slo) for r in chosen) / len(chosen), 4) if chosen else None
                          for tier, slo in (slos or {}).items()}
    return summary


def compact_rows(rows, window=None):
    """The per-request data the analysis needs (bootstrap, soak windows), rounded."""
    out = []
    for r in rows:
        in_window = sum(1 for t in token_times(r) if window[0] <= t < window[1]) if window else r["n_out"]
        out.append([round(r["start"], 4), None if r["ttft"] is None else round(r["ttft"], 5),
                    None if r["tpot"] is None else round(r["tpot"], 6), None if r["e2e"] is None else round(r["e2e"], 4),
                    int(r["ok"]), r["n_out"], in_window])
    return out


ROW_FIELDS = ("start", "ttft", "tpot", "e2e", "ok", "n_out", "win_tokens")


def expand_rows(compact):
    return [dict(zip(ROW_FIELDS, row)) for row in compact]


def block_bootstrap(rows, statistic, block_seconds=30, resamples=2000, seed=20261109, level=0.95):
    """Moving-block bootstrap over requests in arrival order: blocks of consecutive requests spanning about
    `block_seconds` of arrivals. (low, high) of the statistic, or None with fewer than 10 requests."""
    rows = sorted(rows, key=lambda r: r["start"])
    count = len(rows)
    if count < 10:
        return None
    span = rows[-1]["start"] - rows[0]["start"]
    block = min(count, max(1, round(count * block_seconds / span))) if span > 0 else count
    rng, values = random.Random(seed), []
    for _ in range(resamples):
        sample = []
        while len(sample) < count:
            start = rng.randrange(0, count - block + 1)
            sample.extend(rows[start:start + block])
        value = statistic(sample[:count])
        if value is not None:
            values.append(value)
    tail = (1 - level) / 2 * 100
    return (percentile(values, tail), percentile(values, 100 - tail)) if values else None


def bootstrap_point(point, protocol, resamples=None):
    """Bootstrap CIs (within one launch) of P90 TTFT, P90 TPOT, standard goodput and output tok/s."""
    rows = expand_rows(point.get("requests") or [])
    settings = protocol["statistics"]["bootstrap"]
    window = point.get("window")
    slo = protocol["slo"]["tiers"]["standard"].get(point.get("slo_workload") or "", None)
    span = (window[1] - window[0]) if window else None
    chosen = [r for r in rows if not window or window[0] <= r["start"] < window[1]]
    stats = {"ttft_p90": lambda s: percentile([r["ttft"] for r in s if r["ok"]], 90),
             "tpot_p90": lambda s: percentile([r["tpot"] for r in s if r["ok"] and r["tpot"] is not None], 90)}
    if slo:
        stats["goodput.standard"] = lambda s: sum(meets(r, slo) for r in s) / len(s)
    cis = {name: block_bootstrap(chosen, fn, settings["block_seconds"], resamples or settings["resamples"],
                                 settings["seed"]) for name, fn in stats.items()}
    if span:
        # The same population as the point estimate: every request with output tokens inside the window,
        # including requests that started before it.
        contributing = [r for r in rows if r.get("win_tokens")]
        cis["output_tok_s"] = block_bootstrap(contributing, lambda s: sum(r["win_tokens"] for r in s) / span,
                                              settings["block_seconds"], resamples or settings["resamples"],
                                              settings["seed"])
    return cis


# ---------------------------------------------------------------------------------------------------------------
# Telemetry over a point

def read_smi(path):
    """[(unix time, gpu, sm MHz, power W, temperature C, throttle/event reasons)] of the 1-s nvidia-smi CSV
    written by protocol.sh (timestamps in UTC)."""
    rows = []
    try:
        handle = open(path, newline="", errors="replace")
    except OSError:
        return rows
    with handle:
        for line in csv.reader(handle):
            if len(line) < 10 or not re.match(r"\s*\d{4}/", line[0]):
                continue
            try:
                when = datetime.strptime(line[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                number = lambda text: float(re.search(r"[-\d.]+", text).group())
                rows.append((when.timestamp(), int(line[1]), number(line[2]), number(line[4]), number(line[5]),
                             int(line[9].strip(), 16)))
            except (ValueError, AttributeError):
                continue
    return rows


def throttle_summary(rows, start, end, invalid_mask, report_mask):
    chosen = [r for r in rows if start <= r[0] <= end]
    if not chosen:
        return {"samples": 0}
    return {"samples": len(chosen),
            "invalid_fraction": round(sum(1 for r in chosen if r[5] & invalid_mask) / len(chosen), 4),
            "power_cap_fraction": round(sum(1 for r in chosen if r[5] & report_mask) / len(chosen), 4),
            "sm_clock_mhz_mean": round(mean([r[2] for r in chosen]), 1),
            "power_w_mean": round(mean([r[3] for r in chosen]), 1), "temperature_c_max": max(r[4] for r in chosen)}


def read_ib(path):
    samples = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return samples
    for line in lines:
        try:
            samples.append(json.loads(line))
        except ValueError:
            continue
    return samples


def ib_window(samples, start, end):
    """Bytes over InfiniBand during a point: the last sample before it and the first after (4-byte words)."""
    before = [s for s in samples if s["t"] <= start]
    after = [s for s in samples if s["t"] >= end]
    if not before or not after:
        return None
    first, last = before[-1], after[0]
    ports = [p for p in last if p != "t" and p in first]
    return {"ports": len(ports), "sent_gb": round(sum(last[p][0] - first[p][0] for p in ports) * 4 / 1e9, 3),
            "received_gb": round(sum(last[p][1] - first[p][1] for p in ports) * 4 / 1e9, 3)}


# ---------------------------------------------------------------------------------------------------------------
# Parity

def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def parity_compare(candidate, reference):
    """Teacher-forced (P1) and greedy (P2) agreement of a launch with the reference launch."""
    diffs, agreement, documents = [], [], {}
    for name, ref in reference.get("p1", {}).items():
        cand = candidate.get("p1", {}).get(name)
        if not cand:
            continue
        diff = [abs(p - q) for p, q in zip(cand["chosen"], ref["chosen"]) if p is not None and q is not None]
        same = [p == q for p, q in zip(cand["top1"], ref["top1"]) if p is not None and q is not None]
        diffs += diff
        agreement += same
        documents[name] = {"positions": len(diff), "mean_abs_logprob_diff": round(mean(diff) or 0, 6),
                           "top1_agreement": round(sum(same) / len(same), 5) if same else None}
    first = {name: first_difference(candidate["p2"][name]["generated"], ref["generated"])
             for name, ref in reference.get("p2", {}).items() if name in candidate.get("p2", {})}
    return {"positions": len(diffs), "mean_abs_logprob_diff": round(mean(diffs), 6) if diffs else None,
            "p99_abs_logprob_diff": round(percentile(diffs, 99), 6) if diffs else None,
            "top1_agreement": round(sum(agreement) / len(agreement), 5) if agreement else None,
            "documents": documents, "greedy_prompts": len(first),
            "greedy_identical": sum(v is None for v in first.values()),
            "greedy_diverged_before_16": sum(v is not None and v < 16 for v in first.values()),
            "greedy_first_difference": first}


def parity_acceptance(candidate, floor):
    """FP8 acceptance against the A2-vs-A floor of the same round (slo_protocol.json parity.acceptance_fp8)."""
    if not floor or floor.get("mean_abs_logprob_diff") is None or candidate.get("mean_abs_logprob_diff") is None:
        return {"status": "unknown", "reason": "no A2 or candidate parity in this round"}
    checks = {"mean_abs_logprob_diff": candidate["mean_abs_logprob_diff"] <= 2 * floor["mean_abs_logprob_diff"] + 0.005,
              "top1_agreement": candidate["top1_agreement"] >= floor["top1_agreement"] - 0.01,
              "early_divergence": candidate["greedy_diverged_before_16"] <= 2}
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


# ---------------------------------------------------------------------------------------------------------------
# Package (inside the job)

def compact_point(record, verify, protocol, telemetry, ib):
    """Everything the analysis needs from one measured point, without addresses."""
    bench = load_json(Path(verify) / Path(record["result_json"]).name) if record.get("result_json") else None
    rows = request_rows(bench) if bench else []
    slo_workload = record.get("slo_workload")
    slos = {tier: values[slo_workload] for tier, values in protocol["slo"]["tiers"].items() if slo_workload in values}
    window = record.get("window") if record.get("kind") == "open" else None
    steady = steady_state(rows, window, slos, (bench or {}).get("duration"),
                          protocol["statistics"]["p99_min_requests"]) if rows else None
    if steady and not window:
        steady["output_tok_s"] = (bench or {}).get("output_throughput")
    masks = protocol["gates"]["throttle"]
    invalid, report = int(masks["invalid_mask"], 16), int(masks["report_mask"], 16)
    start, end = record.get("t_start_unix") or 0, record.get("t_end_unix") or 0
    gpus = {node: throttle_summary(samples, start, end, invalid, report) for node, samples in telemetry.items()}
    reasons = list((record.get("gates") or {}).get("failed", []))
    reasons += [f"{node}: GPU slowdown (throttle mask {masks['invalid_mask']})" for node, s in gpus.items()
                if s.get("invalid_fraction")]
    if record.get("returncode") == 0 and not rows:
        reasons.append("no per-request data (--save-detailed result missing)")
    flags = list(record.get("flags", []))
    flags += [f"{node}: no GPU telemetry samples during the point (throttle gate not applied)"
              for node, s in gpus.items() if not s.get("samples")]
    if not gpus:
        flags.append("no GPU telemetry file (throttle gate not applied)")
    return {"id": f"{record['workload']}/{record['point_key']}", "name": record["name"], "phase": record["phase"],
            "layout": record["layout"], "round": record["round"], "workload": record["workload"],
            "slo_workload": slo_workload, "point": record["point_key"], "index": record.get("point_index"),
            "kind": record.get("kind"), "rate": record.get("request_rate"), "concurrency": record.get("concurrency"),
            "seed": record.get("seed"), "attempt": record.get("attempt", 1), "capacity_point": record.get("capacity_point"),
            "valid": not reasons, "invalid_reasons": reasons, "flags": flags, "window": window,
            "t_start_unix": start, "t_end_unix": end, "steady": steady,
            "summary": {k: record.get(k) for k in ("completed", "failed", "duration", "output_throughput",
                                                   "request_goodput", "median_ttft_ms", "median_tpot_ms",
                                                   "median_itl_ms", "max_concurrent_requests")},
            "engine_peaks": record.get("per_engine_peaks"), "counters": (record.get("deltas") or {}).get("totals"),
            "gpu": gpus, "ib": {node: ib_window(samples, start, end) for node, samples in ib.items()},
            "requests": compact_rows(rows, window)}


def soak_series(path, start):
    """(seconds, max KV usage over engines, waiting requests summed over engines) every 10 s of a soak."""
    series = []
    for sample in read_ib(path):
        if not series or sample["t"] - start >= series[-1][0] + 10:
            engines = sample.get("engines", {}).values()
            series.append([round(sample["t"] - start, 1), max([e.get("vllm:kv_cache_usage_perc", 0) for e in engines] or [0]),
                           sum(e.get("vllm:num_requests_waiting", 0) for e in engines)])
    return series


def package(root, protocol=None):
    root, protocol = Path(root), protocol or load_protocol()
    nodes = sorted(path.name for path in root.glob("node*") if path.is_dir())
    hosts = [(load_json(root / node / "env.json") or {}).get("hostname") for node in nodes]
    verify = root / "node0" / "verify"
    records = [r for r in (load_json(p) for p in sorted(verify.glob("*.json"))
                           if not p.name.startswith(("bench-", "parity-"))) if isinstance(r, dict)]
    telemetry = {node: read_smi(root / node / "gpu-telemetry.csv") for node in nodes}
    ib = {node: read_ib(root / node / "ib-counters.jsonl") for node in nodes}
    points = [compact_point(r, verify, protocol, telemetry, ib) for r in records if r.get("record_type") == "bench_point"]
    for point, record in zip(points, [r for r in records if r.get("record_type") == "bench_point"]):
        if point["workload"] == "soak":
            point["series"] = soak_series(verify / record["metrics_jsonl"], point["t_start_unix"])
    plans = {}
    for path in sorted(verify.glob("*.json")):
        found = re.match(r"(.+)\.(partr|slo|soak|realtext)\.json$", path.name)
        data = load_json(path) if found else None
        if isinstance(data, dict):
            plans.setdefault(found.group(1), {})[found.group(2)] = {
                "plan": data.get("plan", []), "aborted": data.get("aborted"), "skipped": data.get("skipped"),
                "missing": data.get("missing", False)}
    raw = {}
    for path in sorted(verify.glob("parity-*.json")):
        data = load_json(path)
        if isinstance(data, dict) and data.get("record_type") == "parity":
            raw[(data["phase"], bool(data.get("long")))] = data
    parity = {}
    for (phase, long), data in raw.items():
        reference = "diag-A" if phase.startswith("diag-") else "slo-A"
        if phase != reference and (reference, long) in raw:
            parity[f"{phase}{'-long' if long else ''}"] = parity_compare(data, raw[(reference, long)])
    phases = [json.loads(line) for line in (root / "node0" / "phases.jsonl").read_text().splitlines()
              if line.strip()] if (root / "node0" / "phases.jsonl").exists() else []
    extra = {kind: {path.stem[len(kind) + 1:]: load_json(path) for path in sorted((root / "node0").glob(f"{kind}-*.json"))}
             for kind in ("effective", "nccl", "lb-stats", "clean")}
    rounds = sorted({p["round"] for p in points if p.get("round") is not None})
    return sanitize({"schema": "axk2-slo-package/1", "rounds": rounds, "phases": phases, "plans": plans,
                     "points": points, "parity": parity,
                     "hosts_sha256": {node: hashlib.sha256((h or "").encode()).hexdigest()[:12] for node, h in zip(nodes, hosts)},
                     **{kind.replace("-", "_"): value for kind, value in extra.items()}}, hosts)


# ---------------------------------------------------------------------------------------------------------------
# Analysis (across rounds)

def lambda_star(points, threshold=0.9):
    """points: [(rate, goodput fraction, output tok/s, total tok/s)]. lambda* interpolates linearly between the
    highest rate that meets the threshold and the next rate; it is never extrapolated beyond the grid."""
    points = sorted(points)
    if len(points) < 2:
        return {"value": None, "bracketed": False, "reason": "fewer than two valid rates"}
    good = [i for i, p in enumerate(points) if p[1] is not None and p[1] >= threshold]
    if not good:
        return {"value": None, "bracketed": False, "reason": "no rate meets the threshold: the grid starts too high"}
    i = good[-1]
    if i == len(points) - 1:
        return {"value": None, "bracketed": False, "reason": "the highest rate still meets the threshold: the grid ends too low"}
    (r0, g0, o0, t0), (r1, g1, o1, t1) = points[i], points[i + 1]
    share = (g0 - threshold) / (g0 - g1)
    blend = lambda a, b: None if a is None or b is None else a + share * (b - a)
    return {"value": r0 + share * (r1 - r0), "bracketed": True, "between": [r0, r1],
            "output_tok_s": blend(o0, o1), "total_tok_s": blend(t0, t1),
            "monotonic": all(points[j][1] >= points[j + 1][1] for j in range(len(points) - 1))}


def cost_per_million(price, nodes, tok_s, utilization=1.0):
    """USD per 1M tokens: price per node-hour x nodes / (tok/s x utilization x 3600) x 1e6."""
    return price * nodes / (tok_s * utilization * 3600) * 1e6 if tok_s and tok_s > 0 else None


def t975(df, table):
    return table.get(str(df)) or table[max(table, key=int)]


def paired_log_ratio(x, a, a2=None, table=None):
    """x, a, a2: {round: value}. r_k = ln(x_k / base_k), base_k = sqrt(a_k x a2_k) for rounds where A2 measured
    the metric, else a_k. Mean, 95% t-CI over rounds, min and max."""
    table = table or load_protocol()["statistics"]["t_975"]
    rounds = sorted(k for k in x if k in a and (x[k] or 0) > 0 and (a[k] or 0) > 0)
    logs, corrected = [], []
    for k in rounds:
        base = a[k]
        if a2 and (a2.get(k) or 0) > 0:
            base = math.sqrt(a[k] * a2[k])
            corrected.append(k)
        logs.append(math.log(x[k] / base))
    if not logs:
        return {"n": 0}
    n, center = len(logs), sum(logs) / len(logs)
    sd = statistics.stdev(logs) if n > 1 else None
    half = t975(n - 1, table) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "rounds": rounds, "log_ratios": [round(v, 6) for v in logs], "mean_log": center,
            "ratio": math.exp(center), "sd_log": sd, "ci_half_width_log": half,
            "ci": [math.exp(center - half), math.exp(center + half)] if half is not None else None,
            "min_ratio": math.exp(min(logs)), "max_ratio": math.exp(max(logs)), "drift_corrected_rounds": corrected}


def delta_min_log(a, a2):
    """max(ln 1.05, 2 x max_k |ln(A2_k / A_k)|), or None when A2 never measured the metric."""
    logs = [abs(math.log(a2[k] / a[k])) for k in a if (a.get(k) or 0) > 0 and (a2.get(k) or 0) > 0]
    return max(LN_105, 2 * max(logs)) if logs else None


def decide(stats, higher_is_better, delta_min, band=(0.95, 1.05)):
    """better / worse / equivalent / inconclusive (slo_protocol.json decision_rules)."""
    if stats.get("n", 0) < 2 or not stats.get("ci"):
        return "inconclusive"
    low, high = stats["ci"]
    if band[0] <= low and high <= band[1]:
        return "equivalent"
    if abs(stats["mean_log"]) >= delta_min and (low > 1 or high < 1):
        favours_x = [(r > 0) == higher_is_better and r != 0 for r in stats["log_ratios"]]
        favours_a = [(r < 0) == higher_is_better and r != 0 for r in stats["log_ratios"]]
        if (stats["mean_log"] > 0) == higher_is_better and all(favours_x):
            return "better"
        if (stats["mean_log"] > 0) != higher_is_better and all(favours_a):
            return "worse"
    return "inconclusive"


def best_attempts(points):
    """{(round, phase, point id): the last valid attempt, else the last attempt}."""
    chosen = {}
    for point in sorted(points, key=lambda p: p.get("attempt", 1)):
        key = (point["round"], point["phase"], point["id"])
        if point["valid"] or key not in chosen or not chosen[key]["valid"]:
            chosen[key] = point
    return chosen


def launch_metrics(points, protocol, nodes, full_grid=True):
    """Metrics of one launch (round, phase): point statistics, and with a full rate grid lambda* per workload
    and tier and the cost at lambda* (A2 and C2s run two W1 rates only, so they have no lambda*)."""
    metrics, flags, slo = {}, {}, protocol["slo"]
    for point in points:
        if not point["valid"] or not point.get("steady"):
            continue
        steady = point["steady"]
        for stat in POINT_STATS:
            if stat.endswith("p99") and not steady.get("p99_reportable"):
                continue
            if steady.get(stat) is not None:
                metrics[f"{point['id']}.{stat}"] = steady[stat]
        for tier, value in (steady.get("goodput") or {}).items():
            metrics[f"{point['id']}.goodput.{tier}"] = value
        if {"kv-bound", "capacity-bound"} & set(point.get("flags") or []) or point.get("capacity_point"):
            flags[point["id"]] = sorted({"kv-bound", "capacity-bound"} & set(point.get("flags") or [])) or ["capacity point"]
    lambdas = {}
    for workload in ("W1", "W2", "W3"):
        grid = [p for p in points if p["workload"] == workload and p["kind"] == "open"]
        if not grid or not full_grid:
            continue
        for tier in slo["tiers"]:
            if workload not in slo["tiers"][tier]:
                continue
            if not all(p["valid"] and p.get("steady") for p in grid):
                lambdas[f"{workload}.{tier}"] = {"value": None, "bracketed": False, "reason": "a grid point is invalid"}
                continue
            curve = [(p["rate"], p["steady"]["goodput"].get(tier), p["steady"]["output_tok_s"],
                      (p["steady"]["output_tok_s"] or 0) + (p["steady"].get("input_tok_s") or 0)) for p in grid]
            lambdas[f"{workload}.{tier}"] = result = lambda_star(curve, slo["goodput_threshold"])
            if result["value"] is not None:
                metrics[f"lambda_star.{workload}.{tier}"] = result["value"]
                metrics[f"output_tok_s_at_lambda.{workload}.{tier}"] = result["output_tok_s"]
                for meter in ("on_demand", "spot"):
                    cost = cost_per_million(protocol["prices_usd_per_node_hour"][meter], nodes, result["output_tok_s"])
                    if cost:
                        metrics[f"cost_per_m_output.{workload}.{tier}.{meter}"] = cost
    return metrics, flags, lambdas


def phase_nodes(protocol, phase):
    entry = protocol["phases"].get(phase, {})
    layout = protocol["layouts"].get(entry.get("layout"), {})
    while "same_as" in layout and "nodes" not in layout:
        layout = {**protocol["layouts"][layout["same_as"]], **{k: v for k, v in layout.items() if k != "same_as"}}
    return layout.get("nodes", 1) * entry.get("replicas", 1)


def expected_phases(protocol, round_):
    rounds = protocol["rounds"]
    return ["slo-A", *(f"slo-{n}" for n in rounds["middle_order"].get(str(round_), [])), "slo-A2"]


def full_grid(protocol, phase):
    options = protocol["phases"].get(phase, {}).get("options", {}).get("slo", {})
    return not any(name.endswith("_points") for name in options)


def package_round(pkg):
    return (pkg.get("rounds") or [None])[0]


def round_validity(round_, package_list, chosen, protocol):
    """A round counts only if every measuring suite of every comparison phase reported, none aborted, and every
    planned point has a valid attempt (in the soak round the soak included)."""
    plans = {}
    for pkg in package_list:
        if package_round(pkg) == round_:
            plans.update(pkg.get("plans", {}))
    problems = []
    for phase in expected_phases(protocol, round_):
        suites = plans.get(phase, {})
        required = [s for s in protocol["phases"][phase]["suites"] if s in ("partr", "slo", "realtext", "soak")
                    and (s != "soak" or round_ == protocol["rounds"]["soak_round"])]
        for suite in required:
            plan = suites.get(suite)
            if plan is None:
                problems.append(f"{phase}: {suite} reported no results")
                continue
            if plan.get("missing") or plan.get("skipped"):
                problems.append(f"{phase}: {suite} did not run ({plan.get('skipped')})")
            if plan.get("aborted"):
                problems.append(f"{phase}: {suite} aborted ({plan['aborted']})")
            for name in plan.get("plan", []):
                hits = [p for (r, ph, _), p in chosen.items()
                        if r == round_ and ph == phase and p["name"] in (name, f"{name}-retry")]
                if not hits or not hits[0]["valid"]:
                    problems.append(f"{phase}: {name} has no valid attempt")
    return {"valid": not problems, "problems": problems}


def soak_check(point, rules):
    """Soak pass rules: no failure or preemption, P90 TTFT/TPOT of every 5-min window within the tolerance of
    the first window, no upward KV or waiting trend, no GPU slowdown."""
    rows = expand_rows(point.get("requests") or [])
    low, high = point["window"] or (0, max([r["start"] for r in rows] or [0]))
    width, problems, windows = rules["window_seconds"], [], []
    edge = low
    while edge + width <= high:
        chunk = [r for r in rows if edge <= r["start"] < edge + width and r["ok"]]
        windows.append({"start": edge, "ttft_p90": percentile([r["ttft"] for r in chunk], 90),
                        "tpot_p90": percentile([r["tpot"] for r in chunk if r["tpot"] is not None], 90)})
        edge += width
    for metric in ("ttft_p90", "tpot_p90"):
        first = windows[0][metric] if windows else None
        for window in windows[1:]:
            if first and window[metric] is not None and abs(window[metric] / first - 1) > rules["p90_tolerance"]:
                problems.append(f"{metric} of the window at {window['start']} s is {window[metric] / first:.2f} x the first")
    failed = sum(1 for r in rows if not r["ok"])
    if failed > rules["failed_max"]:
        problems.append(f"{failed} failed requests")
    preemptions = (point.get("counters") or {}).get("vllm:num_preemptions_total", 0) or 0
    if preemptions > rules["preemptions_max"]:
        problems.append(f"{preemptions:.0f} preemptions")
    series = point.get("series") or []
    if len(series) > 2:
        times = [s[0] for s in series]
        center = mean(times)
        spread = sum((t - center) ** 2 for t in times) or 1
        slope = sum((t - center) * (s[1] - mean([v[1] for v in series])) for t, s in zip(times, series)) / spread
        if slope * (times[-1] - times[0]) > rules["kv_rise_max"]:
            problems.append(f"KV usage rose by {slope * (times[-1] - times[0]):.3f}")
        tenth = max(1, len(series) // 10)
        rise = mean([s[2] for s in series[-tenth:]]) - mean([s[2] for s in series[:tenth]])
        if rise > rules["waiting_rise_max"]:
            problems.append(f"waiting requests rose by {rise:.1f}")
    if not point["valid"]:
        problems += point["invalid_reasons"]
    return {"passed": not problems, "problems": problems, "windows": windows}


def suggest_grid(lambdas, factors=(0.6, 1.4), count=4):
    """Four geometric rates from 0.6 x to 1.4 x lambda*: one shared grid when it brackets every layout."""
    values = [v for v in lambdas.values() if v]
    if not values:
        return None
    low, high = factors[0] * min(values), factors[1] * max(values)
    if high / low > (factors[1] / factors[0]) * 1.5:
        return {name: [round(v * factors[0] * (factors[1] / factors[0]) ** (i / (count - 1)), 3) for i in range(count)]
                for name, v in lambdas.items() if v}
    return {"shared": [round(low * (high / low) ** (i / (count - 1)), 3) for i in range(count)]}


def analyze(packages, protocol=None, resamples=None):
    protocol = protocol or load_protocol()
    rules = protocol["decision_rules"]
    points = [p for pkg in packages for p in pkg.get("points", [])]
    chosen = best_attempts(points)
    rounds = sorted({p["round"] for p in points if isinstance(p.get("round"), int) and p["round"] > 0})
    launches = {}
    for (round_, phase, _), point in chosen.items():
        launches.setdefault((round_, phase), []).append(point)
    metrics = {key: launch_metrics(group, protocol, phase_nodes(protocol, key[1]), full_grid(protocol, key[1]))
               for key, group in launches.items()}
    validity = {r: round_validity(r, packages, chosen, protocol) for r in rounds}
    valid_rounds = [r for r in rounds if validity[r]["valid"]]
    series = lambda phase, metric: {r: metrics[(r, phase)][0].get(metric) for r in valid_rounds
                                    if (r, phase) in metrics and metrics[(r, phase)][0].get(metric) is not None}
    catalogue = protocol["metrics"]["primary"] + protocol["metrics"]["secondary"]
    noise = {}
    for item in catalogue:
        a, a2 = series("slo-A", item["id"]), series("slo-A2", item["id"])
        logs = [math.log(a2[k] / a[k]) for k in a if k in a2 and a[k] > 0 and a2[k] > 0]
        if logs:
            noise[item["id"]] = {"log_a2_over_a": [round(v, 6) for v in logs], "max_abs": max(abs(v) for v in logs),
                                 "sd": statistics.stdev(logs) if len(logs) > 1 else None,
                                 "delta_min_log": delta_min_log(a, a2)}
    pooled = max([LN_105] + [n["delta_min_log"] for n in noise.values() if n["delta_min_log"]])
    comparisons = {}
    for candidate, baseline in CANDIDATES:
        rows = {}
        for item in catalogue:
            x, a = series(candidate, item["id"]), series(baseline, item["id"])
            a2 = series("slo-A2", item["id"]) if baseline == "slo-A" else {}
            if not x or not a:
                continue
            stats = paired_log_ratio(x, a, a2 or None, protocol["statistics"]["t_975"])
            point_id = item["id"].split(".")[0]
            bound = sorted({f for r in valid_rounds for ph in (candidate, baseline)
                            for f in metrics.get((r, ph), ({}, {}, {}))[1].get(point_id, [])})
            delta = noise.get(item["id"], {}).get("delta_min_log") or pooled
            label = "excluded (" + ", ".join(bound) + ")" if bound else decide(
                stats, item["better"] == "higher", delta, tuple(rules["equivalent_band"]))
            rows[item["id"]] = {**stats, "better": item["better"], "delta_min_log": delta,
                                "delta_min_source": "A2 of this metric" if item["id"] in noise else "pooled A2 noise",
                                "primary": item in protocol["metrics"]["primary"], "decision": label}
        comparisons[f"{candidate} vs {baseline}"] = rows
    extend = rules["extend"]
    reasons = [f"SD of ln(A2/A) of {m} is {n['sd']:.3f}" for m, n in noise.items()
               if n["sd"] is not None and n["sd"] > extend["sd_ln_a2_over_a_max"]
               and any(m == p["id"] for p in protocol["metrics"]["primary"])]
    for name, rows in comparisons.items():
        if name.endswith("slo-C2"):
            continue
        reasons += [f"{name}: 95% CI half-width of {m} is {math.exp(r['ci_half_width_log']) - 1:.1%}"
                    for m, r in rows.items() if r.get("primary") and r.get("ci_half_width_log")
                    and r["ci_half_width_log"] > math.log(1 + extend["primary_ci_half_width_max"])]
    extension = {"rounds_valid": len(valid_rounds),
                 "recommended": len(valid_rounds) >= extend["after_rounds"] and bool(reasons)
                 and len(valid_rounds) < extend["max_rounds"],
                 "evaluated": len(valid_rounds) >= extend["after_rounds"], "reasons": reasons}
    parity, acceptance = {}, {}
    for pkg in packages:
        for name, result in pkg.get("parity", {}).items():
            parity[f"round {package_round(pkg)} {name}"] = result
        floor = pkg.get("parity", {}).get("slo-A2")
        for phase in ("slo-C2", "slo-C2s"):
            if phase in pkg.get("parity", {}):
                acceptance[f"round {package_round(pkg)} {phase}"] = parity_acceptance(pkg["parity"][phase], floor)
    soak = {f"round {p['round']} {p['phase']}": soak_check(p, protocol["workloads"]["soak"]["pass"])
            for p in chosen.values() if p["workload"] == "soak"}
    lambda_table = {f"round {r} {ph}": lam for (r, ph), (_, _, lam) in sorted(metrics.items()) if lam}
    grid = {}
    for workload in ("W1", "W2"):
        per_phase = {}
        for phase in ("slo-A", "slo-C2", "slo-D"):
            values = [metrics[(r, phase)][0].get(f"lambda_star.{workload}.standard") for r in valid_rounds
                      if (r, phase) in metrics]
            per_phase[phase] = mean(values) if values else None
        grid[workload] = suggest_grid(per_phase)
    dx = {}
    for key in {k for k in metrics if k[1] in ("dx1-D", "dx2-D")}:
        dx.setdefault(key[0], {})[key[1]] = metrics[key][0]
    scaling = {}
    for round_, launch in dx.items():
        for workload in ("W1", "W2"):
            one, two = (launch.get(p, {}).get(f"lambda_star.{workload}.standard") for p in ("dx1-D", "dx2-D"))
            scaling[f"round {round_} {workload}"] = {
                "lambda_dx1": one, "lambda_dx2": two, "ratio": two / one if one and two else None,
                "passed": bool(one and two and two >= protocol["dx2"]["pass_ratio"] * one)}
    diagnostics = []
    for pair in protocol["diagnostics"]["pairs"]:
        x = metrics.get((0, pair["x"]), ({},))[0]
        base = metrics.get((0, pair["base"]), ({},))[0]
        diagnostics.append({"name": pair["name"], "x": pair["x"], "base": pair["base"],
                            "ratios": {m: x[m] / base[m] for m in pair["metrics"] if x.get(m) and base.get(m)},
                            "note": "one launch each: an indication, not a statistically established difference"})
    bootstrap = {}
    if resamples != 0:
        for (round_, phase, point_id), point in sorted(chosen.items()):
            if point["kind"] == "open" and point["valid"] and round_ in valid_rounds:
                bootstrap[f"round {round_} {phase} {point_id}"] = bootstrap_point(point, protocol, resamples)
    invalid = [{"round": p["round"], "phase": p["phase"], "point": p["name"], "attempt": p["attempt"],
                "reasons": p["invalid_reasons"]} for p in points if not p["valid"]]
    return {"schema": "axk2-slo-evidence/1", "protocol_status": protocol["status"], "rounds": rounds,
            "round_validity": validity, "valid_rounds": valid_rounds, "a2_noise": noise,
            "pooled_delta_min_log": pooled, "comparisons": comparisons, "extension": extension,
            "lambda_star": lambda_table, "next_grid": grid, "parity": parity, "parity_acceptance": acceptance,
            "soak": soak, "dx2_scaling": scaling, "diagnostics": diagnostics, "bootstrap_within_launch": bootstrap,
            "invalid_attempts": invalid,
            "launch_metrics": {f"round {r} {ph}": m for (r, ph), (m, _, _) in sorted(metrics.items())},
            "prices_usd_per_node_hour": protocol["prices_usd_per_node_hour"],
            "slo_assumption": protocol["slo"]["assumption"]}


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(evidence):
    lines = ["# Serving protocol results", "", f"- Protocol: {evidence['protocol_status']}",
             f"- SLO: {evidence['slo_assumption']}",
             f"- Rounds: {evidence['rounds']}; valid: {evidence['valid_rounds']}", ""]
    for round_, result in evidence["round_validity"].items():
        if not result["valid"]:
            lines.append(f"- Round {round_} is invalid: " + "; ".join(result["problems"][:10]))
    for name, rows in evidence["comparisons"].items():
        lines += ["", f"## {name}", "", "| metric | ratio | 95% CI | min-max over rounds | rounds | decision |",
                  "|---|---|---|---|---|---|"]
        for metric, row in rows.items():
            ci = f"{row['ci'][0]:.3f}-{row['ci'][1]:.3f}" if row.get("ci") else "-"
            lines.append(f"| {metric}{' (primary)' if row['primary'] else ''} | {fmt(row.get('ratio'))} | {ci} | "
                         f"{fmt(row.get('min_ratio'))}-{fmt(row.get('max_ratio'))} | {row.get('n')} | {row['decision']} |")
    lines += ["", "## lambda* (req/s, linear interpolation inside the grid)", "", "| launch | workload.tier | lambda* | note |",
              "|---|---|---|---|"]
    for launch, table in evidence["lambda_star"].items():
        for key, result in table.items():
            lines.append(f"| {launch} | {key} | {fmt(result.get('value'))} | {result.get('reason', '')} |")
    extension = evidence["extension"]
    lines += ["", f"Round extension: {'recommended' if extension['recommended'] else 'not recommended'}"
              f"{'' if extension['evaluated'] else ' (fewer than 3 valid rounds)'}. " + "; ".join(extension["reasons"])]
    if evidence["dx2_scaling"]:
        lines += ["", "## D x 2 scaling", ""] + [f"- {k}: ratio {fmt(v['ratio'])}, passed {v['passed']}"
                                                 for k, v in evidence["dx2_scaling"].items()]
    if evidence["parity_acceptance"]:
        lines += ["", "## Parity (FP8 acceptance against the A/A2 floor)", ""] + [
            f"- {k}: {v['status']}" for k, v in evidence["parity_acceptance"].items()]
    if evidence["soak"]:
        lines += ["", "## Soak", ""] + [f"- {k}: {'passed' if v['passed'] else 'failed: ' + '; '.join(v['problems'])}"
                                        for k, v in evidence["soak"].items()]
    lines += ["", f"Invalid attempts: {len(evidence['invalid_attempts'])} (all listed in the evidence JSON)", ""]
    return "\n".join(lines)


def report_package(data, key="slo.package"):
    """Sends the package through the job record (report.py), compressed like fetch_results.py expects."""
    text = "z:" + base64.b64encode(zlib.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()
    path = Path("/tmp/axk2-slo-package.txt")
    path.write_text(text)
    subprocess.run([sys.executable, str(HERE.parent / "report.py"), key, "--file", str(path)], check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench_stats.py")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package", help="compact package of one job's outputs (run inside the workspace)")
    pack.add_argument("root")
    pack.add_argument("--out")
    pack.add_argument("--report", action="store_true", help="also send it through the job record")
    study = commands.add_parser("analyze", help="statistics over the packages of every round")
    study.add_argument("packages", nargs="+")
    study.add_argument("--out", required=True)
    study.add_argument("--md")
    study.add_argument("--resamples", type=int, help="bootstrap resamples (0 skips the bootstrap)")
    for sub in (pack, study):
        sub.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "package":
        data = package(args.root, protocol)
        if args.out:
            Path(args.out).write_text(json.dumps(data))
        if args.report:
            report_package(data)
        print(json.dumps({"points": len(data["points"]), "rounds": data["rounds"], "parity": sorted(data["parity"])}))
        return 0
    packages = []
    for path in args.packages:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            packages += data
            continue
        data = data.get("reports", data)  # `aml/fetch_results.py show <collect job>` nests the tags here
        packages.append(data.get("slo.package", data))
    evidence = analyze(packages, protocol, args.resamples)
    Path(args.out).write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(evidence), encoding="utf-8")
    print(json.dumps({"valid_rounds": evidence["valid_rounds"], "extension": evidence["extension"]["recommended"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
