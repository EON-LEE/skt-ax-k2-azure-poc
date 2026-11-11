"""Statistics of the pre-registered serving protocol (slo_protocol.json, docs/BENCHMARK_PROTOCOL.md).

  python3 bench_stats.py package <job outputs folder> [--out package.json] [--report [--state reported.json]]
      Inside the job (protocol.sh, after every phase and on exit) or the workspace (collect-slo-outputs.yml):
      one compact, sanitized package per job from the raw outputs: per point the gates, the steady-state
      summary and per-request data, per-engine counter deltas, GPU telemetry and InfiniBand counters of every
      node over the point; parity against the round job's reference (A or D; screening: the job's first launch)
      with the P3 long-context check; the accuracy results; the plans, phase results, start records, effective
      configs, NCCL and clean-state records, load-balancer stats.
      --report sends it through report.py as chunked MLflow tags: slo.package (everything but the points, with
      the list of parts) and slo.package.<launch> (one launch's points), each under 150 chunks of 4,000 chars.
  python3 bench_stats.py analyze <package.json>... --out evidence.json --md summary.md
      Operator side: validity per round job (FP8, NVFP4), lambda* with the bracket gate, cost per 1M output
      tokens, paired log-ratio t-CIs with drift correction by each job's relaunch (A2, D2) where it measured the
      metric, decision labels, the relaunch noise floor and delta_min, cross-job comparisons of the NVFP4 layouts
      with A, 'added after screening' labels, the round-extension rule per job, P99 gating, parity acceptance
      and the P3 long-context check, soak and the round-3 SOAK_RATES, the paired accuracy comparison (McNemar,
      non-inferiority), D x 2 scaling, diagnostics.
  python3 bench_stats.py select <screening package.json>... --out selection.json --md selection.md
      The pre-registered selection rule (slo_protocol.json screen.selection): the round candidates
      (ROUND_LAYOUTS), the D x 2 decision and every screened launch's facts, parity and metrics.

Package arguments may be raw packages or `aml/fetch_results.py show` output (whole or split). Standard library
only; every step is a function that the tests feed with synthetic data.
"""
import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "slo_protocol.json"
REPORT = HERE.parent / "report.py"
REPORT_KEY = "slo.package"
CHUNK_CHARS = 4000  # report.py
MAX_CHUNKS = 150  # per reported key (slo_protocol.json reporting)
REPORT_FAILURES = ("AXK2_REPORT_WARN mlflow unavailable", "set_tags attempt 4 failed")  # report.py output
TAIL_CHARS = 16384  # node1 server-log tail kept per two-server phase
LN_105 = math.log(1.05)
SANITIZE = ((re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"), (re.compile(r"/mnt/azureml/cr/j/[^/\s\"']+"), "<job>"))
ACCURACY_KEYS = ("phase", "layout", "failed", "summary", "unscored", "pins", "settings", "item_fields", "items", "seconds")
POINT_STATS = ("ttft_p50", "ttft_p90", "ttft_p99", "tpot_p50", "tpot_p90", "tpot_p99", "e2e_p50", "e2e_p90",
               "itl_p50", "itl_p99", "output_tok_s")


def load_protocol(path=None):
    return json.loads(Path(path or PROTOCOL).read_text(encoding="utf-8"))


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve_layout(protocol, name):
    """A layout with its `same_as` chain applied: overrides replace, extra_args append; `root` names the base."""
    layout = dict(protocol["layouts"][name])
    if "same_as" not in layout:
        return {**layout, "root": name}
    base = resolve_layout(protocol, layout.pop("same_as"))
    extra = layout.pop("extra_args", [])
    base.update(layout)
    base["args"] = [*base["args"], *extra]
    return base


def phase_entry(protocol, phase):
    """The protocol entry of a PHASE_PLAN token: an explicit phase, or <kind>-<layout> built from
    phase_templates for a layout of the table (slo-sensitivity for a sensitivity layout; none for an
    experimental layout)."""
    if phase in protocol["phases"]:
        return protocol["phases"][phase]
    kind, _, name = phase.partition("-")
    templates = protocol.get("phase_templates", {})
    if isinstance(templates.get(kind), dict) and name in protocol["layouts"]:
        layout = resolve_layout(protocol, name)
        if not layout.get("experimental"):
            entry = json.loads(json.dumps(templates["slo-sensitivity" if kind == "slo" and layout.get("sensitivity") else kind]))
            entry["layout"] = name
            if "label" in entry:
                entry["label"] = entry["label"].format(layout=name)
            return entry
    raise ValueError(f"{phase} is not a phase of slo_protocol.json")


def phase_suites(protocol, phase, round_=None):
    """The client suites a phase runs: soak only in the soak round, realtext only for the checkpoints (or layouts)
    of W1real, accuracy only in the accuracy rounds."""
    entry = phase_entry(protocol, phase)
    checkpoint = resolve_layout(protocol, entry["layout"])["checkpoint"]
    real = protocol["workloads"]["W1real"]
    return [s for s in entry["suites"] if (s != "soak" or round_ == protocol["rounds"]["soak_round"])
            and (s != "realtext" or checkpoint in real["checkpoints"] or entry["layout"] in real.get("layouts", []))
            and (s != "accuracy" or round_ in protocol.get("accuracy", {}).get("rounds", []))]


def ordered(layouts, round_):
    """The round's order of the candidates: as listed in odd rounds, reversed in even rounds."""
    return list(layouts) if int(round_) % 2 else list(layouts)[::-1]


def sensitivity_of(protocol, name):
    """<name>s when the layout table defines it as a sensitivity layout, else None."""
    return f"{name}s" if protocol["layouts"].get(f"{name}s", {}).get("sensitivity") else None


def rates_key(protocol, name):
    """The name a layout's rate grid (SLO_RATES_<key>) and soak rate (SOAK_RATES <key>=...) are looked up under:
    its own, except for A2 and the sensitivity layouts (same_rates), which use their base layout's."""
    layout = protocol["layouts"][name]
    return layout["same_as"] if layout.get("same_rates") else name


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
    """Teacher-forced (P1) and greedy (P2, or P3 for the long documents) agreement of a launch with the reference
    launch: first differing token, position-wise token agreement and |delta logprob| of the chosen tokens before the
    first difference."""
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
    first, same_tokens, prefix = {}, {}, []
    for name, ref in reference.get("p2", {}).items():
        cand = candidate.get("p2", {}).get(name)
        if not cand:
            continue
        first[name] = first_difference(cand["generated"], ref["generated"])
        same_tokens[name] = round(sum(a == b for a, b in zip(cand["generated"], ref["generated"]))
                                  / max(1, len(cand["generated"]), len(ref["generated"])), 4)
        upto = first[name] if first[name] is not None else len(ref["generated"])
        prefix += [abs(a - b) for a, b in zip((cand.get("logprobs") or [])[:upto], (ref.get("logprobs") or [])[:upto])
                   if a is not None and b is not None]
    return {"positions": len(diffs), "mean_abs_logprob_diff": round(mean(diffs), 6) if diffs else None,
            "p99_abs_logprob_diff": round(percentile(diffs, 99), 6) if diffs else None,
            "top1_agreement": round(sum(agreement) / len(agreement), 5) if agreement else None,
            "documents": documents, "greedy_prompts": len(first),
            "greedy_identical": sum(v is None for v in first.values()),
            "greedy_diverged_before_16": sum(v is not None and v < 16 for v in first.values()),
            "greedy_first_difference": first, "greedy_token_agreement": same_tokens,
            "greedy_prefix_logprob_diff": round(mean(prefix), 6) if prefix else None}


def long_check(comparison, rule):
    """P3 verdict of a launch against its job's reference (slo_protocol.json parity.p3): suspect when the greedy
    continuation of every long document leaves the reference's before token `suspect_before`."""
    firsts = comparison.get("greedy_first_difference") or {}
    if not firsts:
        return {"status": "no data"}
    early = [value is not None and value < rule["suspect_before"] for value in firsts.values()]
    return {"status": "suspect" if all(early) else "ok", "first_difference": firsts,
            "token_agreement": comparison.get("greedy_token_agreement"),
            "prefix_logprob_diff": comparison.get("greedy_prefix_logprob_diff")}


def parity_acceptance(candidate, floor):
    """Acceptance of a candidate against its job's control-vs-reference floor of the same round (A2 vs A, D2 vs D;
    slo_protocol.json parity.acceptance)."""
    if not floor or floor.get("mean_abs_logprob_diff") is None or candidate.get("mean_abs_logprob_diff") is None:
        return {"status": "unknown", "reason": "no control or candidate parity in this round"}
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
            "peaks": {"kv_cache_usage": record.get("peak_kv_cache_usage"),
                      "running": record.get("peak_running_requests_sum_over_engines")},
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
    plans, screens = {}, {}
    for path in sorted(verify.glob("*.json")):
        found = re.match(r"(.+)\.(partr|slo|soak|realtext|screen)\.json$", path.name)
        data = load_json(path) if found else None
        if isinstance(data, dict):
            plans.setdefault(found.group(1), {})[found.group(2)] = {
                "plan": data.get("plan", []), "aborted": data.get("aborted"), "skipped": data.get("skipped"),
                "missing": data.get("missing", False), **({"rates_override": data["rates_override"]}
                                                          if data.get("rates_override") else {})}
            if found.group(2) == "screen":
                screens[found.group(1)] = {k: data.get(k) for k in ("facts", "steps", "aborted", "suite_seconds")}
    accuracy = {path.name[:-len(".accuracy.json")]: {k: data.get(k) for k in ACCURACY_KEYS}
                for path in sorted(verify.glob("*.accuracy.json")) for data in [load_json(path)] if isinstance(data, dict)}
    plan_record = load_json(root / "node0" / "protocol-plan.json")
    round_reference = f"slo-{(plan_record if isinstance(plan_record, dict) else {}).get('reference') or 'A'}"
    raw = {}
    for path in sorted(verify.glob("parity-*.json")):
        data = load_json(path)
        if isinstance(data, dict) and data.get("record_type") == "parity":
            raw[(data["phase"], bool(data.get("long")))] = data
    # Screening compares every launch with the job's first launch (scr-A, scr-D or scr-Dx1).
    screen_reference = next((ref for ref in protocol.get("screen", {}).get("references", []) if (ref, False) in raw), None)
    parity = {}
    for (phase, long), data in raw.items():
        if phase.startswith("scr-"):
            reference = screen_reference
        else:
            reference = "diag-A" if phase.startswith("diag-") else round_reference  # the round job's reference
        if reference and phase != reference and (reference, long) in raw:
            result = parity_compare(data, raw[(reference, long)])
            if long and "p3" in protocol.get("parity", {}):
                result["long_check"] = long_check(result, protocol["parity"]["p3"])
            parity[f"{phase}{'-long' if long else ''}"] = result
    phases = [json.loads(line) for line in (root / "node0" / "phases.jsonl").read_text().splitlines()
              if line.strip()] if (root / "node0" / "phases.jsonl").exists() else []
    extra = {kind: {path.stem[len(kind) + 1:]: load_json(path) for path in sorted((root / "node0").glob(f"{kind}-*.json"))}
             for kind in ("effective", "nccl", "lb-stats", "clean", "start", "reclaim", "peer", "preflight", "descs")}
    # Tails of the second server's log (node1), pulled by node_tools.py pull for two-server phases.
    tails = {path.name[len("tail-vllm-"):-len("-replica1.log")]: path.read_text(errors="replace")[-TAIL_CHARS:]
             for node in nodes for path in sorted((root / node).glob("tail-vllm-*-replica1.log"))}
    rounds = sorted({p["round"] for p in points if p.get("round") is not None})
    return sanitize({"schema": "axk2-slo-package/1", "rounds": rounds, "phases": phases, "plans": plans,
                     "points": points, "parity": parity, "screen_reference": screen_reference, "screens": screens,
                     "protocol_plan": plan_record, "replica_tails": tails, "accuracy": accuracy,
                     "hosts_sha256": {node: hashlib.sha256((h or "").encode()).hexdigest()[:12] for node, h in zip(nodes, hosts)},
                     **{kind.replace("-", "_"): value for kind, value in extra.items()}}, hosts)


# ---------------------------------------------------------------------------------------------------------------
# Reporting through the job record (report.py chunks, fetch_results.py show reads them back)

def encode_report(data):
    """The text report.py sends for one key: "z:" + base64(zlib(JSON)), what fetch_results.py decodes."""
    return "z:" + base64.b64encode(zlib.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()


def chunks(text):
    return max(1, math.ceil(len(text) / CHUNK_CHARS))


def split_package(data, max_chars=MAX_CHUNKS * CHUNK_CHARS):
    """{report key: payload}. slo.package holds everything but the points and maps each part to its launch;
    each slo.package.<launch> holds one launch's points, halved (<launch>-p1, -p2, ...) while its encoded text
    is longer than max_chars. When even the index is too long, the effective configs lose their log lines."""
    parts, launches = {}, {}
    for point in data.get("points", []):
        launches.setdefault(point["phase"], []).append(point)

    def add(name, launch, points):
        part = {"schema": "axk2-slo-package-part/1", "launch": launch, "points": points}
        if len(points) > 1 and len(encode_report(part)) > max_chars:
            half = len(points) // 2
            add(f"{name}-p1", launch, points[:half])
            add(f"{name}-p2", launch, points[half:])
        else:
            parts[name] = part

    for launch, points in launches.items():
        add(launch, launch, points)
    index = {**{k: v for k, v in data.items() if k != "points"},
             "parts": {name: part["launch"] for name, part in sorted(parts.items())}}
    if len(encode_report(index)) > max_chars:
        index["effective"] = {phase: {k: v for k, v in (value or {}).items() if k != "lines"}
                              for phase, value in (index.get("effective") or {}).items()}
        index["trimmed"] = ["effective.*.lines"]
    return {REPORT_KEY: index, **{f"{REPORT_KEY}.{name}": part for name, part in sorted(parts.items())}}


def merge_package(reports, key=REPORT_KEY):
    """One package from what `fetch_results.py show` returns ("reports"): the whole package under key, or the
    index under key plus the parts it names. Missing parts are listed with the launches they held (only those
    launches lose their points), never silently ignored."""
    index = reports.get(key) if isinstance(reports, dict) else None
    if not isinstance(index, dict):
        return None
    if "parts" not in index:
        return index
    names = index["parts"] if isinstance(index["parts"], dict) else {name: name for name in index["parts"]}
    points, missing, lost = [], [], set()
    for name, launch in names.items():
        part = reports.get(f"{key}.{name}")
        if isinstance(part, dict):
            points += part.get("points", [])
        else:
            missing.append(name)
            lost.add(launch)
    return {**{k: v for k, v in index.items() if k != "parts"}, "points": points, "missing_parts": missing,
            "missing_launches": sorted(lost)}


def report_package(data, state=None):
    """Sends the split package through report.py (chunked MLflow tags): the parts first, the index last, so an
    index never names a part that has not been sent. With a state file, only keys whose text changed since the
    last call go out again. report.py always exits 0, so its output decides whether a key went out: no mlflow, a
    failed fourth set_tags attempt or a reported exception means not sent. AXK2_REPORT_PKGS (the job's mlflow
    packages) becomes the child's PYTHONPATH, as in verify_suites.emit."""
    sent = (load_json(state) if state else None) or {}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if os.environ.get("AXK2_REPORT_PKGS"):
        env["PYTHONPATH"] = os.environ["AXK2_REPORT_PKGS"]
    folder = Path(state).parent if state else Path(tempfile.gettempdir())
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, payload in sorted(split_package(data).items(), key=lambda item: item[0] == REPORT_KEY):
        text = encode_report(payload)
        digest = hashlib.sha256(text.encode()).hexdigest()
        summary[key] = {"chunks": chunks(text), "sent": False}
        if sent.get(key) == digest:
            continue
        path = folder / "slo-package-report.txt"
        path.write_text(text)
        try:
            run = subprocess.run([sys.executable, str(REPORT), key, "--file", str(path)], env=env, check=False,
                                 timeout=300, capture_output=True, text=True)
            output = f"{run.stdout or ''}\n{run.stderr or ''}"
            warnings = [line for line in output.splitlines() if line.startswith("AXK2_REPORT_WARN")]
            done = run.returncode == 0 and not any(marker in output for marker in REPORT_FAILURES) and \
                not any(line.startswith(f"AXK2_REPORT_WARN {key}:") for line in warnings)
        except subprocess.TimeoutExpired:
            done, warnings = False, [f"AXK2_REPORT_WARN {key}: report.py timed out after 300 s"]
        for line in warnings:
            print(line[:300], flush=True)
        summary[key]["sent"] = done
        if done and state:
            sent[key] = digest
            Path(state).write_text(json.dumps(sent))
    return summary


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
        return {"value": None, "bracketed": False, "high": points[0][0], "bound": "below the grid",
                "reason": f"below the grid (lambda* < {points[0][0]:g}): no rate meets the threshold, the grid starts too high"}
    i = good[-1]
    if i == len(points) - 1:
        return {"value": None, "bracketed": False, "low": points[-1][0], "bound": "above the grid",
                "reason": f"above the grid (lambda* > {points[-1][0]:g}): the highest rate still meets the threshold, "
                          "the grid ends too low"}
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
    """Nodes a phase occupies (cost): the layout's nodes x the phase's servers."""
    try:
        entry = phase_entry(protocol, phase)
    except ValueError:
        return 1
    return resolve_layout(protocol, entry["layout"]).get("nodes", 1) * entry.get("replicas", 1)


def package_round(pkg):
    return (pkg.get("rounds") or [None])[0]


def round_jobs(protocol, packages):
    """{(round, job): plan} from the round packages' protocol-plan.json records: the job's reference, control and
    candidates, the packages that carry it and whether ROUND_LAYOUTS deviated. A record without a job (the earlier
    single-job plan) counts as the FP8 job with its recorded candidates."""
    jobs = {}
    for pkg in packages:
        plan, round_ = pkg.get("protocol_plan"), package_round(pkg)
        if not isinstance(plan, dict) or not isinstance(round_, int) or round_ < 1:
            continue
        job = plan.get("job") or ("fp8" if plan.get("layouts") else None)
        if not job:
            continue
        if (round_, job) in jobs:
            jobs[(round_, job)]["sources"].append(pkg.get("source"))
            continue
        spec = protocol["rounds"]["jobs"].get(job, {})
        jobs[(round_, job)] = {"job": job, "reference": plan.get("reference") or spec.get("reference", "A"),
                               "control": plan.get("control") if plan.get("job") else spec.get("control", "A2"),
                               "layouts": list(plan.get("layouts") or []), "deviation": bool(plan.get("deviation")),
                               "sources": [pkg.get("source")]}
    return jobs


def job_phases(plan, round_):
    """A round job's comparison phases in the round's order: reference, candidates, control."""
    return [f"slo-{plan['reference']}", *(f"slo-{n}" for n in ordered(plan["layouts"], round_)),
            *([f"slo-{plan['control']}"] if plan.get("control") else [])]


def phase_jobs(protocol, plans):
    """{phase: job} of every round phase: reference, candidates, their sensitivity runs and the control."""
    owner = {}
    for job, plan in plans.items():
        names = [plan["reference"], *plan["layouts"], *filter(None, (sensitivity_of(protocol, n) for n in plan["layouts"])),
                 *([plan["control"]] if plan.get("control") else [])]
        for name in names:
            owner.setdefault(f"slo-{name}", job)
    return owner


def comparison_pairs(protocol, plans):
    """(x, baseline, kind): every candidate against its job's reference, each sensitivity run against the reference
    and its layout, and the reference and candidates of a job without A against A (cross-job)."""
    pairs = []
    for plan in plans.values():
        reference = f"slo-{plan['reference']}"
        for name in plan["layouts"]:
            pairs.append((f"slo-{name}", reference, "within job"))
            if sensitivity_of(protocol, name):
                sensitivity = f"slo-{sensitivity_of(protocol, name)}"
                pairs += [(sensitivity, reference, "sensitivity"), (sensitivity, f"slo-{name}", "sensitivity")]
    if any(plan["reference"] == "A" for plan in plans.values()):
        for plan in plans.values():
            if plan["reference"] != "A":
                pairs += [(f"slo-{name}", "slo-A", "cross-job") for name in (plan["reference"], *plan["layouts"])]
    return pairs


def selection_label(protocol, phase):
    """How a round layout entered the rounds (rounds.selection): pre-registered or added after screening; a
    sensitivity run inherits its layout's label."""
    try:
        name = phase_entry(protocol, phase)["layout"]
    except ValueError:
        return None
    layout = protocol["layouts"].get(name, {})
    base = layout.get("same_as", name) if layout.get("sensitivity") else name
    selection = protocol["rounds"].get("selection", {})
    if base in selection.get("added_after_screening", []):
        return selection.get("label", "added after screening")
    return "pre-registered" if base in selection.get("pre_registered", []) else None


def full_grid(protocol, phase):
    try:
        options = phase_entry(protocol, phase).get("options", {}).get("slo", {})
    except ValueError:
        return True
    return not any(name.endswith("_points") for name in options)


def realtext_not_run(protocol, round_):
    """The recorded reason W1real could not run in this round (workloads.W1real.not_run_rounds), else None."""
    return (protocol["workloads"]["W1real"].get("not_run_rounds") or {}).get(str(round_))


def round_validity(round_, package_list, chosen, protocol, phases):
    """A round job counts only if every measuring suite of its comparison phases reported, none aborted, and every
    planned point has a valid attempt (in the soak round the soak included). A round recorded in
    workloads.W1real.not_run_rounds does not need realtext (a harness defect, slo_protocol.json deviations)."""
    plans = {}
    for pkg in package_list:
        if package_round(pkg) == round_:
            plans.update(pkg.get("plans", {}))
    problems, waived = [], []
    lost = {launch for pkg in package_list if package_round(pkg) == round_ for launch in pkg.get("missing_launches") or []}
    not_run = realtext_not_run(protocol, round_)
    for phase in phases:
        if phase in lost:
            problems.append(f"{phase}: a part of the results package never arrived")
        suites = plans.get(phase, {})
        required = [s for s in phase_suites(protocol, phase, round_) if s in ("partr", "slo", "realtext", "soak")]
        if not_run and "realtext" in required:
            required.remove("realtext")
            waived.append(f"{phase}: realtext not required in round {round_} ({not_run})")
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
    return {"valid": not problems, "problems": problems, **({"waived": waived} if waived else {})}


# ---------------------------------------------------------------------------------------------------------------
# Accuracy: NVFP4 against FP8 on the same items (slo_protocol.json accuracy)

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial test of the b + c discordant items with p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired_accuracy(pairs, margin, z=1.959964):
    """pairs: {item: (FP8 correct, NVFP4 correct)}. d = accuracy(NVFP4) - accuracy(FP8), its two-sided 95% Wald
    interval for paired proportions, the exact McNemar p-value and the non-inferiority decision for the margin."""
    n = len(pairs)
    if not n:
        return {"n": 0, "decision": "no paired items"}
    both = sum(1 for f, v in pairs.values() if f and v)
    b = sum(1 for f, v in pairs.values() if f and not v)
    c = sum(1 for f, v in pairs.values() if v and not f)
    d = (c - b) / n
    half = z * math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    low, high = d - half, d + half
    return {"n": n, "accuracy_fp8": (both + b) / n, "accuracy_nvfp4": (both + c) / n, "both_right": both,
            "fp8_only": b, "nvfp4_only": c, "neither": n - both - b - c, "difference": d, "ci95": [low, high],
            "mcnemar_p": mcnemar_exact(b, c), "margin": margin,
            "decision": "non-inferior" if low > margin else "inferior" if high < margin else "inconclusive"}


def accuracy_items(result):
    """{(dataset, key): (correct, gold)} of one server's accuracy result (correct None: no score)."""
    index = {field: i for i, field in enumerate(result.get("item_fields") or [])}
    if not {"dataset", "key", "correct", "gold"} <= set(index):
        return {}
    return {(row[index["dataset"]], row[index["key"]]): (row[index["correct"]], row[index["gold"]])
            for row in result.get("items") or []}


def accuracy_comparison(packages, protocol):
    """The pre-registered accuracy comparison: the FP8 and NVFP4 servers' results (accuracy.pair, accuracy rounds),
    paired by item over both datasets (primary) and per dataset."""
    settings = protocol.get("accuracy")
    if not settings:
        return None
    found = {}
    for pkg in packages:
        if package_round(pkg) not in settings["rounds"]:
            continue
        for phase, result in (pkg.get("accuracy") or {}).items():
            for side, wanted in settings["pair"].items():
                if phase == wanted and side not in found and isinstance(result, dict):
                    found[side] = {"phase": phase, "round": package_round(pkg), "source": pkg.get("source"), "result": result}
    rule = settings["comparison"]
    out = {"margin": rule["margin"], "rule": rule["decision"], "primary": rule["primary"],
           "servers": {side: {**{k: f[k] for k in ("phase", "round", "source")},
                              **{k: f["result"].get(k) for k in ("summary", "failed", "unscored", "pins")}}
                       for side, f in found.items()}}
    missing = [settings["pair"][side] for side in ("fp8", "nvfp4") if side not in found]
    failed = [f"{side}: {f['result']['failed']}" for side, f in found.items() if f["result"].get("failed")]
    if missing or failed:
        return {**out, "decision": "missing" if missing else "failed",
                "problems": [f"no accuracy result of {phase}" for phase in missing] + failed}
    fp8, nvfp4 = accuracy_items(found["fp8"]["result"]), accuracy_items(found["nvfp4"]["result"])
    mismatched = sorted(k for k in fp8 if k in nvfp4 and fp8[k][1] != nvfp4[k][1])
    pairs = {k: (fp8[k][0], nvfp4[k][0]) for k in fp8 if k in nvfp4 and k not in mismatched
             and fp8[k][0] is not None and nvfp4[k][0] is not None}
    expected = sum(spec["items"] for spec in settings["datasets"].values())
    tables = {"pooled": paired_accuracy(pairs, rule["margin"]),
              **{name: paired_accuracy({k: v for k, v in pairs.items() if k[0] == name}, rule["margin"])
                 for name in settings["datasets"]}}
    complete = len(pairs) >= math.ceil(rule["min_paired_fraction"] * expected - 1e-9)
    partial = not complete and len(pairs) >= rule.get("min_partial_pairs", math.inf)
    coverage = "complete" if complete else "partial" if partial else "incomplete"
    problems = [] if complete else [f"{len(pairs)} of {expected} items scored on both servers ("
                                    + ("decision on those items: partial coverage" if partial else
                                       f"at least {rule['min_paired_fraction']:.0%} for a complete and "
                                       f"{rule.get('min_partial_pairs')} for a partial decision needed") + ")"]
    return {**out, "expected": expected, "paired": len(pairs), "gold_mismatches": len(mismatched),
            "cross_job": found["fp8"]["source"] != found["nvfp4"]["source"], "tables": tables, "coverage": coverage,
            "decision": tables["pooled"]["decision"] if complete or partial else "incomplete", "problems": problems}


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
    point_rates = {key: {p["id"]: p.get("rate") for p in group} for key, group in launches.items()}
    jobs, order = round_jobs(protocol, packages), list(protocol["rounds"]["jobs"])
    plans = {}
    for (round_, job), plan in sorted(jobs.items(), key=lambda kv: (order.index(kv[0][1]) if kv[0][1] in order else len(order),
                                                                  kv[0][0])):
        merged = plans.setdefault(job, {"job": job, "reference": plan["reference"], "control": plan["control"], "layouts": []})
        merged["layouts"] += [n for n in plan["layouts"] if n not in merged["layouts"]]
    by_job = {}
    for (round_, job), plan in sorted(jobs.items()):
        result = round_validity(round_, packages, chosen, protocol, job_phases(plan, round_))
        if len(plan["sources"]) > 1:
            result = {"valid": False, "problems": [f"round {round_} has {len(plan['sources'])} packages of the {job} job "
                                                   "(pass one package per round and job)", *result["problems"]]}
        by_job.setdefault(round_, {})[job] = result
    validity = {r: {"valid": bool(by_job.get(r)) and all(v["valid"] for v in by_job[r].values()),
                    "problems": [p for v in by_job.get(r, {}).values() for p in v["problems"]]
                    or ([] if by_job.get(r) else ["no round job reported a plan (protocol-plan.json) for this round"]),
                    "jobs": by_job.get(r, {})} for r in rounds}
    valid_rounds = [r for r in rounds if validity[r]["valid"]]
    job_valid = {job: [r for r in rounds if by_job.get(r, {}).get(job, {}).get("valid")] for job in plans}
    owner = phase_jobs(protocol, plans)
    rounds_of = lambda *phases: [r for r in rounds if all(r in job_valid.get(owner.get(ph), []) for ph in phases)]
    series = lambda phase, metric, use: {r: metrics[(r, phase)][0].get(metric) for r in use
                                         if (r, phase) in metrics and metrics[(r, phase)][0].get(metric) is not None}
    catalogue = protocol["metrics"]["primary"] + protocol["metrics"]["secondary"]
    primary = {item["id"] for item in protocol["metrics"]["primary"]}
    noise = {}
    for job, plan in plans.items():
        if not plan.get("control"):
            continue
        for item in catalogue:
            use = job_valid.get(job, [])
            a, a2 = series(f"slo-{plan['reference']}", item["id"], use), series(f"slo-{plan['control']}", item["id"], use)
            logs = [math.log(a2[k] / a[k]) for k in a if k in a2 and a[k] > 0 and a2[k] > 0]
            if logs:
                noise.setdefault(job, {})[item["id"]] = {
                    "log_a2_over_a": [round(v, 6) for v in logs], "max_abs": max(abs(v) for v in logs),
                    "sd": statistics.stdev(logs) if len(logs) > 1 else None, "delta_min_log": delta_min_log(a, a2)}
    pooled = {job: max([LN_105] + [n["delta_min_log"] for n in values.values() if n["delta_min_log"]])
              for job, values in noise.items()}
    fp8_job = next((job for job, plan in plans.items() if plan["reference"] == "A"), None)

    def delta_for(base_job, metric):
        """delta_min of a comparison: the baseline job's control noise, else the FP8 job's A/A2 noise (borrowed)."""
        source = base_job if base_job in noise else fp8_job if fp8_job in noise else None
        if source is None:
            return LN_105, "ln 1.05 (no relaunch control measured)"
        note = "" if source == base_job else f" (borrowed from the {source} job)"
        if noise[source].get(metric, {}).get("delta_min_log"):
            return noise[source][metric]["delta_min_log"], f"{plans[source]['control']} of this metric{note}"
        return pooled[source], f"pooled {plans[source]['control']} noise{note}"

    comparisons, info = {}, {}
    for x, base, kind in comparison_pairs(protocol, plans):
        use, base_job = rounds_of(x, base), owner.get(base)
        base_plan = plans.get(base_job, {})
        control = base_plan.get("control") if base == f"slo-{base_plan.get('reference')}" else None
        rows = {}
        for item in catalogue:
            point_id = item["id"].split(".")[0]
            same = [r for r in use if point_rates.get((r, x), {}).get(point_id) == point_rates.get((r, base), {}).get(point_id)]
            xs, a = series(x, item["id"], same), series(base, item["id"], same)
            a2 = series(f"slo-{control}", item["id"], same) if control else {}
            if not xs or not a:
                continue
            stats = paired_log_ratio(xs, a, a2 or None, protocol["statistics"]["t_975"])
            bound = sorted({f for r in same for ph in (x, base)
                            for f in metrics.get((r, ph), ({}, {}, {}))[1].get(point_id, [])})
            delta, source = delta_for(base_job, item["id"])
            label = "excluded (" + ", ".join(bound) + ")" if bound else decide(
                stats, item["better"] == "higher", delta, tuple(rules["equivalent_band"]))
            rows[item["id"]] = {**stats, "better": item["better"], "delta_min_log": delta, "delta_min_source": source,
                                "primary": item["id"] in primary, "decision": label}
        name = f"{x} vs {base}"
        comparisons[name] = rows
        info[name] = {"kind": kind, "x_job": owner.get(x), "base_job": base_job, "rounds": use,
                      "selection": selection_label(protocol, x)}
    extend = rules["extend"]
    reasons = {job: [] for job in plans}
    for job, values in noise.items():
        plan = plans[job]
        reasons[job] += [f"SD of ln({plan['control']}/{plan['reference']}) of {m} is {n['sd']:.3f}" for m, n in values.items()
                         if n["sd"] is not None and n["sd"] > extend["sd_ln_a2_over_a_max"] and m in primary]
    references = {f"slo-{plan['reference']}" for plan in plans.values()}
    for name, rows in comparisons.items():
        if name.split(" vs ")[1] not in references:
            continue
        found = [f"{name}: 95% CI half-width of {m} is {math.exp(r['ci_half_width_log']) - 1:.1%}"
                 for m, r in rows.items() if r.get("primary") and r.get("ci_half_width_log")
                 and r["ci_half_width_log"] > math.log(1 + extend["primary_ci_half_width_max"])]
        for job in sorted({info[name]["x_job"], info[name]["base_job"]} - {None}):
            reasons[job] += found
    per_job = {job: {"rounds_valid": len(job_valid[job]), "evaluated": len(job_valid[job]) >= extend["after_rounds"],
                     "recommended": len(job_valid[job]) >= extend["after_rounds"] and bool(reasons[job])
                     and len(job_valid[job]) < extend["max_rounds"], "reasons": reasons[job]} for job in plans}
    extension = {"rounds_valid": len(valid_rounds), "recommended": any(v["recommended"] for v in per_job.values()),
                 "evaluated": any(v["evaluated"] for v in per_job.values()), "jobs": per_job,
                 "reasons": list(dict.fromkeys(r for job in plans for r in reasons[job]))}
    parity, acceptance, long_context = {}, {}, {}
    for pkg in packages:
        for name, result in pkg.get("parity", {}).items():
            parity[f"round {package_round(pkg)} {name}"] = result
            if result.get("long_check"):
                long_context[f"round {package_round(pkg)} {name[:-len('-long')]}"] = result["long_check"]
        for plan in plans.values():
            floor = pkg.get("parity", {}).get(f"slo-{plan['control']}") if plan.get("control") else None
            for phase in (f"slo-{n}" for c in plan["layouts"] for n in (c, sensitivity_of(protocol, c)) if n):
                if phase in pkg.get("parity", {}) and plan.get("control"):
                    acceptance[f"round {package_round(pkg)} {phase}"] = parity_acceptance(pkg["parity"][phase], floor)
    soak = {f"round {p['round']} {p['phase']}": soak_check(p, protocol["workloads"]["soak"]["pass"])
            for p in chosen.values() if p["workload"] == "soak"}
    lambda_table = {f"round {r} {ph}": lam for (r, ph), (_, _, lam) in sorted(metrics.items()) if lam}
    measured = [f"slo-{n}" for plan in plans.values() for n in (plan["reference"], *plan["layouts"])]
    grid = {}
    for workload in ("W1", "W2"):
        per_phase = {}
        for phase in dict.fromkeys(measured or ["slo-A"]):
            values = [metrics[(r, phase)][0].get(f"lambda_star.{workload}.standard") for r in valid_rounds
                      if (r, phase) in metrics]
            values = [v for v in values if v]
            per_phase[phase] = mean(values) if values else None
        grid[workload] = suggest_grid(per_phase)
    fraction, soak_rates = protocol["workloads"]["soak"]["rate_fraction_of_lambda_star"], {}
    for job, plan in plans.items():
        early = [r for r in job_valid.get(job, []) if r < protocol["rounds"]["soak_round"]]
        for name in (plan["reference"], *plan["layouts"]):
            phase = f"slo-{name}"
            values = [metrics[(r, phase)][0].get("lambda_star.W1.standard") for r in early if (r, phase) in metrics]
            values = [v for v in values if v]
            bounds = [b.get("low") or b.get("high") for r in early if (r, phase) in metrics
                      for b in [metrics[(r, phase)][2].get("W1.standard") or {}] if b.get("low") or b.get("high")]
            if values or bounds:
                soak_rates[rates_key(protocol, name)] = {
                    "rate": round(fraction * (mean(values) if values else min(bounds)), 3),
                    "from": "lambda*" if values else "bound", "rounds": early}
    dx = {}
    for (round_, phase), (values, _, _) in metrics.items():
        kind, _, layout = phase.partition("-")
        if kind in ("dx1", "dx2"):
            dx.setdefault((round_, layout), {})[kind] = values
    scaling = {}
    for (round_, layout), launch in sorted(dx.items()):
        for workload in ("W1", "W2"):
            one, two = (launch.get(kind, {}).get(f"lambda_star.{workload}.standard") for kind in ("dx1", "dx2"))
            name = f"round {round_} {workload}" if layout == "D" else f"round {round_} {layout} {workload}"
            scaling[name] = {
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
            if point["kind"] == "open" and point["valid"] and round_ in job_valid.get(owner.get(phase), valid_rounds):
                bootstrap[f"round {round_} {phase} {point_id}"] = bootstrap_point(point, protocol, resamples)
    invalid = [{"round": p["round"], "phase": p["phase"], "point": p["name"], "attempt": p["attempt"],
                "reasons": p["invalid_reasons"]} for p in points if not p["valid"]]
    deviations = [*protocol.get("deviations", []),
                  *({"round": r, "job": job, "what": f"ROUND_LAYOUTS={','.join(plan['layouts'])} differs from the frozen list"}
                    for (r, job), plan in sorted(jobs.items()) if plan["deviation"]),
                  *({"round": package_round(pkg), "phase": phase, "what": f"SLO_RATES override {suites['slo']['rates_override']}"}
                    for pkg in packages for phase, suites in (pkg.get("plans") or {}).items()
                    if (suites.get("slo") or {}).get("rates_override"))]
    return {"schema": "axk2-slo-evidence/2", "protocol_status": protocol["status"], "rounds": rounds,
            "round_jobs": {str(r): {job: {k: plan[k] for k in ("reference", "control", "layouts", "deviation", "sources")}
                                    for (rr, job), plan in sorted(jobs.items()) if rr == r} for r in rounds},
            "round_layouts": {str(r): {job: plan["layouts"] for (rr, job), plan in sorted(jobs.items()) if rr == r}
                              for r in rounds},
            "selection": protocol["rounds"].get("selection"), "deviations": deviations,
            "round_validity": validity, "valid_rounds": valid_rounds, "valid_rounds_by_job": job_valid,
            "noise": noise, "a2_noise": noise.get(fp8_job, {}), "pooled_delta_min_log": pooled.get(fp8_job, LN_105),
            "comparisons": comparisons, "comparison_info": info, "extension": extension,
            "lambda_star": lambda_table, "next_grid": grid, "parity": parity, "parity_acceptance": acceptance,
            "long_context": long_context,
            "soak": soak, "soak_rates": {"rule": protocol["rounds"].get("soak_rates"), "rates": soak_rates,
                                         "env": "SOAK_RATES=" + ",".join(f"{k}={v['rate']:g}" for k, v in soak_rates.items())},
            "accuracy": accuracy_comparison(packages, protocol), "dx2_scaling": scaling, "diagnostics": diagnostics,
            "bootstrap_within_launch": bootstrap, "invalid_attempts": invalid,
            "launch_metrics": {f"round {r} {ph}": m for (r, ph), (m, _, _) in sorted(metrics.items())},
            "prices_usd_per_node_hour": protocol["prices_usd_per_node_hour"],
            "slo_assumption": protocol["slo"]["assumption"]}


# ---------------------------------------------------------------------------------------------------------------
# Screening: per-launch facts and the pre-registered selection of the round candidates

def point_peaks(point):
    """Peak KV usage (largest engine) and peak of running requests (summed over engines) during a point."""
    peaks, engines = point.get("peaks") or {}, (point.get("engine_peaks") or {}).values()
    kv = peaks.get("kv_cache_usage")
    running = peaks.get("running")
    if kv is None and engines:
        kv = max(e.get("kv_cache_usage_perc", 0) for e in engines)
    if running is None and engines:
        running = sum(e.get("num_requests_running", 0) for e in engines)
    return {"peak_kv_cache_usage": kv, "peak_running": running}


def peer_engines(protocol, phase):
    """Does node1 run engines of this phase that log only to node1's own file? A second server (replicas: Dx2, PD)
    or an mp-multinode layout's headless DP engines do; Ray's workers on node1 log through node0's server."""
    entry = phase_entry(protocol, phase)
    return entry.get("replicas", 1) > 1 or resolve_layout(protocol, entry["layout"]).get("launcher") == "mp-multinode"


def kv_from_text(text):
    """KV cache sizes logged after the last bench_replica start mark of (the tail of) a node1 log."""
    text = (text or "").rsplit("=== axk2 replica start", 1)[-1]
    return [int(value.replace(",", "")) for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)]


def screen_launches(packages, protocol):
    """Per screening launch (scr-*): phase result, start record, effective config, screen facts and steps,
    parity against the job's reference, metrics of its valid points (with flags, peaks and lambda*).
    max_model_len is the served value of the start record (/v1/models after a -1 relaunch), not the engine config
    line's pre-fit value. KV tokens add node1's engines (its facts, else its pulled log tail) for phases whose node1
    engines log only there; without them kv_scope says "node0 only". A phase found in several packages (scr-A of
    both FP8 jobs) is kept once per package: the first under its name, the others as <phase>@<source>."""
    launches = {}
    for number, pkg in enumerate(packages, start=1):
        source = pkg.get("source") or f"package {number}"
        points = [p for p in pkg.get("points", []) if str(p.get("phase", "")).startswith("scr-")]
        groups = {}
        for (_, phase, _), point in best_attempts(points).items():
            groups.setdefault(phase, []).append(point)
        results = {p["phase"]: p for p in pkg.get("phases", []) if str(p.get("phase", "")).startswith("scr-")}
        for phase in dict.fromkeys([*results, *groups]):
            key = phase if phase not in launches else f"{phase}@{source}"
            metrics, flags, lambdas = launch_metrics(groups.get(phase, []), protocol, phase_nodes(protocol, phase))
            effective, result = (pkg.get("effective") or {}).get(phase) or {}, results.get(phase) or {}
            start, two = (pkg.get("start") or {}).get(phase), peer_engines(protocol, phase)
            node1 = (((pkg.get("peer") or {}).get(phase) or {}).get("kv_tokens")
                     or kv_from_text((pkg.get("replica_tails") or {}).get(phase))) if two else []
            kv = [*(effective.get("kv_tokens") or []), *node1]
            launches[key] = {
                "phase": phase, "source": source, "key": key,
                "status": result.get("status"), "note": result.get("note"), "seconds": result.get("seconds"),
                "start": start, "max_model_len": start.get("max_model_len") if start else effective.get("max_model_len"),
                "kv_tokens": kv, "kv_tokens_total": sum(kv) if kv else None,
                "kv_scope": ("node0 and node1" if node1 else "node0 only") if two else "all engines",
                "problems": effective.get("problems"),
                "screen": (pkg.get("screens") or {}).get(phase), "parity": (pkg.get("parity") or {}).get(phase),
                "reference": pkg.get("screen_reference"), "metrics": metrics, "flags": flags, "lambda": lambdas,
                "peaks": {p["id"]: point_peaks(p) for p in groups.get(phase, []) if p["valid"]},
                "invalid_points": sorted({p["id"] for p in groups.get(phase, []) if not p["valid"]}),
                "reclaim": (pkg.get("reclaim") or {}).get(phase),
                "preflight": {k: v for k, v in ((pkg.get("preflight") or {}).get(phase) or {}).items()
                              if k in ("choice", "choice_env", "job_env", "reason")} or None,
                "missing_part": phase in (pkg.get("missing_launches") or [])}
    return launches


def same_job(launches, launch, phase):
    """The launch of `phase` from the package (job) of `launch`; the first one under that name otherwise."""
    for other in launches.values() if launch else ():
        if other["phase"] == phase and other["source"] == launch["source"]:
            return other
    return launches.get(phase)


def screen_parity_problems(parity, limits):
    if not parity or parity.get("mean_abs_logprob_diff") is None or parity.get("top1_agreement") is None:
        return ["no short parity against the job's reference"]
    problems = []
    if parity["mean_abs_logprob_diff"] > limits["mean_abs_logprob_diff_max"]:
        problems.append(f"mean |d logprob| {parity['mean_abs_logprob_diff']:.4f} > {limits['mean_abs_logprob_diff_max']}")
    if parity["top1_agreement"] < limits["top1_agreement_min"]:
        problems.append(f"top-1 agreement {parity['top1_agreement']:.4f} < {limits['top1_agreement_min']}")
    if (parity.get("greedy_diverged_before_16") or 0) > limits["greedy_diverged_before_16_max"]:
        problems.append(f"{parity['greedy_diverged_before_16']} greedy prompts diverged before token 16")
    return problems


def screen_eligibility(phase, launch, reference, rule, protocol):
    """Reasons a screened launch cannot join the rounds (slo_protocol.json screen.selection.eligible)."""
    if launch is None:
        return ["not screened"]
    problems = []
    if launch["status"] != "passed":
        problems.append(f"phase {launch['status'] or 'not finished'}" + (f": {launch['note']}" if launch.get("note") else ""))
    fallback = (launch.get("start") or {}).get("length_fallback")
    if fallback:
        problems.append(f"could not hold --max-model-len {fallback.get('requested')} (relaunched with -1)")
    if launch.get("missing_part"):
        problems.append("a part of the results package never arrived")
    if phase != rule["reference"]:
        problems += [f"parity: {p}" for p in screen_parity_problems(launch.get("parity"), protocol["screen"]["parity"]["pass"])]
    if rule["rank_by"] not in launch["metrics"]:
        problems.append(f"{rule['rank_by']} missing (point not run or invalid)")
    for metric in rule["tpot"]:
        value, base = launch["metrics"].get(metric), ((reference or {}).get("metrics") or {}).get(metric)
        if value is None or base is None:
            problems.append(f"{metric} missing ({'reference' if value is not None else 'this launch'})")
        elif value > rule["tpot_max_ratio"] * base:
            problems.append(f"{metric} is {value / base:.2f} x the reference's (limit {rule['tpot_max_ratio']})")
    return problems


def capacity_bounds(result):
    """(low, high) of lambda*: equal when bracketed, one bound when the ladder missed it, (None, None) if unknown."""
    if not result:
        return None, None
    if result.get("value") is not None:
        return result["value"], result["value"]
    return result.get("low"), result.get("high")


def dx2_decision(x, base, ratio):
    """pass when lambda*(x) >= ratio x lambda*(base) for every value the bounds allow, fail when it cannot be,
    otherwise undecided."""
    x_low, x_high = capacity_bounds(x)
    b_low, b_high = capacity_bounds(base)
    if x_low is not None and b_high is not None and x_low >= ratio * b_high:
        return "pass"
    if x_high is not None and b_low is not None and x_high < ratio * b_low:
        return "fail"
    return "undecided"


def select(packages, protocol=None):
    """The pre-registered selection of the round candidates from the screening packages."""
    protocol = protocol or load_protocol()
    rules = protocol["screen"]["selection"]
    launches = screen_launches(packages, protocol)
    out = {"schema": "axk2-screen-selection/1", "protocol_status": protocol["status"], "rule": rules["rule"],
           "always": rules["always"], "families": {}}
    chosen = []
    for family in ("fp8", "nvfp4"):
        rule = rules[family]
        candidates = {}
        for phase in rule["phases"]:
            for key in [k for k, launch in launches.items() if launch["phase"] == phase] or [phase]:
                launch, point = launches.get(key), rule["rank_by"].rsplit(".", 1)[0]
                reference = same_job(launches, launch, rule["reference"])
                problems = screen_eligibility(phase, launch, reference, rule, protocol)
                if reference is None or reference["status"] != "passed":
                    problems.insert(0, f"the reference {rule['reference']} did not pass")
                launch = launch or {}
                candidates[key] = {"layout": phase_entry(protocol, phase)["layout"], "phase": phase,
                                   "reference": (reference or {}).get("key"), "eligible": not problems,
                                   "value": launch.get("metrics", {}).get(rule["rank_by"]), "problems": problems,
                                   "ranking_point": {"flags": (launch.get("flags") or {}).get(point, []),
                                                     **(launch.get("peaks") or {}).get(point, {})}}
        ranking = sorted((k for k, c in candidates.items() if c["eligible"]),
                         key=lambda k: (-candidates[k]["value"], rule["phases"].index(candidates[k]["phase"])))
        choice = candidates[ranking[0]]["layout"] if ranking else None
        out["families"][family] = {"reference": rule["reference"], "rank_by": rule["rank_by"], "ranking": ranking,
                                   "choice": choice, "candidates": candidates}
        if choice:
            chosen.append(choice)
    dx = rules["dx2"]
    key = dx["metric"].split(".", 1)[1]
    x, base = ((launch or {}).get("lambda", {}).get(key) for launch in
               (launches.get(dx["x"]), same_job(launches, launches.get(dx["x"]), dx["base"])))
    decision = dx2_decision(x, base, dx["ratio_min"])
    out["dx2"] = {"decision": decision, "lambda_dx2": x, "lambda_dx1": base, "ratio_min": dx["ratio_min"],
                  "ratio": x["value"] / base["value"] if x and base and x.get("value") and base.get("value") else None,
                  "run_dx2_job": decision != "fail", "note": dx["unbracketed"]}
    out["pd"] = {key: {"status": pd["status"], "note": pd["note"], "source": pd["source"], "preflight": pd["preflight"],
                       "lambda_w1_standard": pd["lambda"].get("W1.standard"), "parity": pd["parity"],
                       "invalid_points": pd["invalid_points"], "selected": False, "rule": rules["pd"]}
                 for key, pd in launches.items() if pd["phase"] == "scr-PD"} or None
    sensitivity = sensitivity_of(protocol, chosen[0]) if chosen and resolve_layout(protocol, chosen[0])["checkpoint"] == "fp8" else None
    keys = [rates_key(protocol, name) for name in ["A", *chosen]]
    out.update(round_layouts=",".join(chosen), sensitivity=sensitivity,
               submit=f"--set environment_variables.ROUND_LAYOUTS={','.join(chosen)}" if chosen else None,
               rates={"slo_rates_env": [f"SLO_RATES_{key}" for key in keys], "soak_rates_keys": keys,
                      "soak_rates": "SOAK_RATES=" + ",".join(f"{key}=<req/s>" for key in keys),
                      "note": "the round-3 soak looks these keys up (A2 and the sensitivity runs use their base layout's)"},
               pairs=[pair_ratios(launches, pair, key) for pair in protocol["screen"]["pairs"]
                      for key in [k for k, launch in launches.items() if launch["phase"] == pair["x"]] or [pair["x"]]],
               launches=launches)
    return out


def pair_ratios(launches, pair, key):
    """x / base of a screening pair's metrics for the launch `key` of its x: the base of the same job when that job
    has one, else the first one of another job (cross_job: PD's rerun against the earlier two-node job's D)."""
    x = launches.get(key)
    base = same_job(launches, x, pair["base"])
    ratios = {m: x["metrics"][m] / base["metrics"][m] for m in pair["metrics"]
              if x and base and x["metrics"].get(m) is not None and base["metrics"].get(m)}
    return {"name": pair["name"], "x": key, "base": (base or {}).get("key", pair["base"]),
            "cross_job": bool(x and base and x["source"] != base["source"]), "ratios": ratios,
            "note": "one launch each: an indication, not a statistically established difference"}


def selection_markdown(selection):
    rows = ["# Screening and selection", "", f"- Rule: {selection['rule']}",
            f"- Round candidates: A + {selection['round_layouts'] or '(none eligible)'}"
            + (f"; round 1 adds {selection['sensitivity']}" if selection.get("sensitivity") else ""),
            f"- Submit the rounds with: `{selection['submit']}`" if selection.get("submit") else "- No candidate is eligible.",
            f"- D x 2: {selection['dx2']['decision']} (ratio {fmt(selection['dx2']['ratio'])}, minimum "
            f"{selection['dx2']['ratio_min']}); run slo-nvfp4-2replica-nd96.yml: {selection['dx2']['run_dx2_job']}",
            f"- Rate keys of the rounds: {', '.join(selection['rates']['slo_rates_env'])}; {selection['rates']['soak_rates']}", "",
            "| launch | status | max_model_len | KV tokens per engine (engines, total) | s to healthy | parity mean abs / top-1 | "
            "c1 TPOT 1K / 8K ms | c32 8K tok/s | W1 lambda* | invalid points |", "|---|---|---|---|---|---|---|---|---|---|"]
    for phase, launch in selection["launches"].items():
        start, parity, metrics = launch.get("start") or {}, launch.get("parity") or {}, launch["metrics"]
        kv = launch.get("kv_tokens") or []
        rows.append(f"| {phase} | {launch['status']} | {fmt(launch.get('max_model_len'))}"
                    f"{' (fallback)' if start.get('length_fallback') else ''} | "
                    f"{(f'{min(kv):,}-{max(kv):,} ({len(kv)}, {sum(kv):,})' if kv else '-')}"
                    f"{' node0 only' if launch.get('kv_scope') == 'node0 only' else ''} | {fmt(start.get('seconds_to_healthy'))} | "
                    f"{fmt(parity.get('mean_abs_logprob_diff'), 4)} / {fmt(parity.get('top1_agreement'), 4)} | "
                    f"{fmt(metrics.get('R/isl1024-c1.tpot_p50'), 1)} / {fmt(metrics.get('R/isl8192-c1.tpot_p50'), 1)} | "
                    f"{fmt(metrics.get('R/isl8192-c32.output_tok_s'), 1)} | {fmt(metrics.get('lambda_star.W1.standard'))} | "
                    f"{', '.join(launch['invalid_points']) or '-'} |")
    for family, result in selection["families"].items():
        rows += ["", f"## {family}: {result['choice'] or 'no eligible layout'} (by {result['rank_by']})", ""]
        rows += [f"- {phase}: {'eligible' if c['eligible'] else 'excluded'}, {fmt(c['value'], 1)}"
                 + (f" [{', '.join(c['ranking_point']['flags'])}; peak KV {fmt(c['ranking_point'].get('peak_kv_cache_usage'))}, "
                    f"peak running {fmt(c['ranking_point'].get('peak_running'))}]" if c["ranking_point"]["flags"] else "")
                 + ("" if c["eligible"] else " - " + "; ".join(c["problems"])) for phase, c in result["candidates"].items()]
    rows += ["", "## Indications (one launch each)", ""]
    rows += [f"- {pair['name']} ({pair['x']} / {pair['base']}{', cross-job' if pair.get('cross_job') else ''}): "
             + (", ".join(f"{m} x{v:.3f}" for m, v in pair["ratios"].items()) or "-") for pair in selection["pairs"]]
    for key, pd in (selection.get("pd") or {}).items():
        preflight = pd.get("preflight") or {}
        rows += ["", f"PD (experimental, never selected) {key}: {pd['status']}"
                 + (f"; NIXL preflight: {preflight.get('choice') or 'failed'}" if preflight else "")
                 + (f" - {pd['note']}" if pd.get("note") else "")]
    return "\n".join(rows) + "\n"


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(evidence):
    lines = ["# Serving protocol results", "", f"- Protocol: {evidence['protocol_status']}",
             f"- SLO: {evidence['slo_assumption']}",
             f"- Rounds: {evidence['rounds']}; valid in every job: {evidence['valid_rounds']}; valid per job: "
             + (", ".join(f"{job} {rounds}" for job, rounds in evidence["valid_rounds_by_job"].items()) or "-"), ""]
    selection = evidence.get("selection") or {}
    label = selection.get("label") or "added after screening"
    if selection.get("added_after_screening"):
        lines.append(f"- {label[0].upper() + label[1:]} ({selection.get('date')}): {', '.join(selection['added_after_screening'])}; "
                     f"pre-registered: {', '.join(selection.get('pre_registered', []))}")
    for deviation in evidence.get("deviations") or []:
        lines.append(f"- Deviation: {deviation.get('what')}" + (f" ({deviation['why']})" if deviation.get("why") else ""))
    for round_, result in evidence["round_validity"].items():
        if not result["valid"]:
            lines.append(f"- Round {round_} is invalid: " + "; ".join(result["problems"][:10]))
        waived = [w for job in (result.get("jobs") or {}).values() for w in job.get("waived", [])]
        if waived:
            lines.append(f"- Round {round_} waived: " + "; ".join(waived[:10]))
    for name, rows in evidence["comparisons"].items():
        meta = (evidence.get("comparison_info") or {}).get(name, {})
        tags = [tag for tag in ("cross-job" if meta.get("kind") == "cross-job" else None,
                                label if meta.get("selection") == label else None) if tag]
        lines += ["", f"## {name}" + (f" ({', '.join(tags)})" if tags else ""), "",
                  "| metric | ratio | 95% CI | min-max over rounds | rounds | decision |", "|---|---|---|---|---|---|"]
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
              f"{'' if extension['evaluated'] else ' (fewer than 3 valid rounds)'}"
              + "".join(f"; {job}: {'extend' if v['recommended'] else 'no'}" for job, v in (extension.get("jobs") or {}).items())
              + ". " + "; ".join(extension["reasons"])]
    if (evidence.get("soak_rates") or {}).get("rates"):
        lines += ["", f"Round-3 soak rates: `{evidence['soak_rates']['env']}`"]
    accuracy = evidence.get("accuracy")
    if accuracy:
        coverage = " (partial coverage)" if accuracy.get("coverage") == "partial" else ""
        lines += ["", f"## Accuracy, NVFP4 against FP8 (margin {accuracy['margin']:+.2f}): {accuracy['decision']}{coverage}", ""]
        lines += [f"- {problem}" for problem in accuracy.get("problems") or []]
        if accuracy.get("tables"):
            lines += ["| items | n | FP8 | NVFP4 | difference | 95% CI | McNemar p | decision |", "|---|---|---|---|---|---|---|---|"]
            for name, row in accuracy["tables"].items():
                ci = f"{row['ci95'][0]:+.3f} to {row['ci95'][1]:+.3f}" if row.get("ci95") else "-"
                lines.append(f"| {name} | {row['n']} | {fmt(row.get('accuracy_fp8'))} | {fmt(row.get('accuracy_nvfp4'))} | "
                             f"{fmt(row.get('difference'))} | {ci} | {fmt(row.get('mcnemar_p'))} | {row['decision']} |")
    if evidence["dx2_scaling"]:
        lines += ["", "## D x 2 scaling", ""] + [f"- {k}: ratio {fmt(v['ratio'])}, passed {v['passed']}"
                                                 for k, v in evidence["dx2_scaling"].items()]
    if evidence["parity_acceptance"]:
        lines += ["", "## Parity (acceptance against the control-vs-reference floor: A2/A, D2/D)", ""] + [
            f"- {k}: {v['status']}" for k, v in evidence["parity_acceptance"].items()]
    if evidence.get("long_context"):
        lines += ["", "## Long-context greedy check (P3, 32K documents, against the job's reference)", ""] + [
            f"- {k}: {v['status']}; first difference {v.get('first_difference')}, token agreement {v.get('token_agreement')}"
            for k, v in evidence["long_context"].items()]
    if evidence["soak"]:
        lines += ["", "## Soak", ""] + [f"- {k}: {'passed' if v['passed'] else 'failed: ' + '; '.join(v['problems'])}"
                                        for k, v in evidence["soak"].items()]
    lines += ["", f"Invalid attempts: {len(evidence['invalid_attempts'])} (all listed in the evidence JSON)", ""]
    return "\n".join(lines)


def read_packages(paths):
    """Packages from files: raw packages, lists of them, or `aml/fetch_results.py show` output whose "reports"
    hold the package whole (slo.package) or split (slo.package + slo.package.<part>)."""
    packages = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            packages += [{"source": f"{Path(path).stem}[{i}]", **pkg} for i, pkg in enumerate(data)]
            continue
        source = data.get("job") or Path(path).stem
        data = data.get("reports", data)
        packages.append({"source": source, **(merge_package(data) or data)})
    return packages


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench_stats.py")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package", help="compact package of one job's outputs (in the job or the workspace)")
    pack.add_argument("root")
    pack.add_argument("--out")
    pack.add_argument("--report", action="store_true", help="also send it through the job record (report.py)")
    pack.add_argument("--state", help="with --report: send only the keys that changed since the last call")
    study = commands.add_parser("analyze", help="statistics over the packages of every round")
    study.add_argument("packages", nargs="+")
    study.add_argument("--out", required=True)
    study.add_argument("--md")
    study.add_argument("--resamples", type=int, help="bootstrap resamples (0 skips the bootstrap)")
    choose = commands.add_parser("select", help="round candidates from the screening packages")
    choose.add_argument("packages", nargs="+")
    choose.add_argument("--out", required=True)
    choose.add_argument("--md")
    for sub in (pack, study, choose):
        sub.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "package":
        data = package(args.root, protocol)
        if args.out:
            Path(args.out).write_text(json.dumps(data))
        keys = report_package(data, args.state) if args.report else {
            key: {"chunks": chunks(encode_report(value))} for key, value in split_package(data).items()}
        print(json.dumps({"points": len(data["points"]), "rounds": data["rounds"], "parity": sorted(data["parity"]),
                          "keys": keys}))
        return 0
    packages = read_packages(args.packages)
    if args.command == "select":
        selection = select(packages, protocol)
        Path(args.out).write_text(json.dumps(selection, indent=1), encoding="utf-8")
        if args.md:
            Path(args.md).write_text(selection_markdown(selection), encoding="utf-8")
        print(json.dumps({"round_layouts": selection["round_layouts"], "sensitivity": selection["sensitivity"],
                          "dx2": selection["dx2"]["decision"], "submit": selection["submit"],
                          "slo_rates_env": selection["rates"]["slo_rates_env"],
                          "soak_rates": selection["rates"]["soak_rates"]}))
        return 0
    evidence = analyze(packages, protocol, args.resamples)
    Path(args.out).write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(evidence), encoding="utf-8")
    print(json.dumps({"valid_rounds": evidence["valid_rounds"], "valid_rounds_by_job": evidence["valid_rounds_by_job"],
                      "extension": evidence["extension"]["recommended"], "soak_rates": evidence["soak_rates"]["env"],
                      "accuracy": (evidence["accuracy"] or {}).get("decision")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
