"""Client suites of the pre-registered serving protocol (slo_protocol.json, docs/BENCHMARK_PROTOCOL.md).

Benchmark-only: entry.sh reaches this module through bench/protocol.sh for the slo-*, diag-*, cap-*, dx1-*,
dx2-* and scr-* phases; the demo and the older phases never import it.

  python3 slo_suites.py plan --round N [--layouts C4,D24] [--record F] <PHASE_PLAN tokens>
                                                            expand `slo-round` into the round's phases
  python3 slo_suites.py phase-spec <phase> --round N [--json] server flags, environment and suites (bash)
  python3 slo_suites.py run --tag T --suites a,b --round N ...  run client suites against a server

Suites:
parity       P1 teacher-forced prompt log-probabilities of 16 fixed 2,048-token documents and P2 greedy
             continuations of 32 fixed prompts (parity_corpus.json); raw per-token data for bench_stats.py.
paritylong   P1 for the two 32,768-token documents (run last in a phase).
warmup       Discarded 32-request bursts per input length (seeds 9xxxxxx), then a prefix-cache reset.
partr        Part R: the published 1K/8K/32K x concurrency 1/32 matrix with round seeds.
slo          Closed-loop W1 (S1) and open-loop Poisson W1/W2 rate grids with --goodput.
realtext     W1 lengths with real-text windows (FP8 layouts).
soak         W1 at 0.8 x lambda* for 45 minutes (soak round only).
screen       Screening (ROUND 0): start facts, short parity, warm-up, Part R and one Poisson W1 point (or a W1
             ladder); every step fails softly.
Other names (functional, ...) run from verify_suites.py.

Every measured point waits for an idle server, resets the prefix cache (POST /reset_prefix_cache needs
VLLM_SERVER_DEV_MODE=1), snapshots /metrics before and after, samples /metrics every second per engine
(JSONL, engines never added together), applies the validity gates and is rerun once if invalid.
"""
import argparse
import concurrent.futures as futures
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PARENT = str(HERE.parent)
_ADDED = _PARENT not in sys.path
if _ADDED:
    sys.path.insert(0, _PARENT)
import verify_suites as vs  # noqa: E402  (aml/src: emit, call, server_healthy and the older suites)
if _ADDED:
    sys.path.remove(_PARENT)
from bench_stats import ordered, phase_entry, phase_suites, rates_key, resolve_layout, sensitivity_of  # noqa: E402,F401

PROTOCOL = HERE / "slo_protocol.json"
CORPUS = HERE / "parity_corpus.json"
MODEL = vs.MODEL

SAMPLE_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
LABEL_PAIR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")
COUNTERS = ("vllm:num_preemptions_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_cached_total",
            "vllm:request_success_total")
HISTOGRAMS = ("vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
              "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
              "vllm:request_decode_time_seconds", "vllm:iteration_tokens_total")
TRACKED = GAUGES + COUNTERS + tuple(f"{name}_{part}" for name in HISTOGRAMS for part in ("sum", "count"))
BENCH_KEYS = ["completed", "failed", "duration", "total_input_tokens", "total_output_tokens", "request_throughput",
              "request_goodput", "output_throughput", "total_token_throughput", "max_output_tokens_per_s",
              "max_concurrent_requests", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p95_ttft_ms",
              "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
              "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms", "mean_e2el_ms",
              "median_e2el_ms", "p90_e2el_ms", "p95_e2el_ms", "p99_e2el_ms"]
SUMMARY_KEYS = ("name", "attempt", "valid", "flags", "returncode", "completed", "failed", "concurrency",
                "request_rate", "output_throughput", "request_goodput", "median_ttft_ms", "p90_ttft_ms",
                "median_tpot_ms", "p90_tpot_ms", "median_itl_ms", "peak_kv_cache_usage",
                "peak_running_requests_sum_over_engines")


# ---------------------------------------------------------------------------------------------------------------
# Plans, seeds and phase settings

def load_protocol(path=None):
    return json.loads(Path(path or PROTOCOL).read_text(encoding="utf-8"))


def protocol_seed(round_, workload, point, rep=0):
    """SEED = 1000000*ROUND + 10000*WORKLOAD + 100*POINT + REP: identical for every layout of a round."""
    return 1000000 * int(round_) + 10000 * int(workload) + 100 * int(point) + int(rep)


def warmup_seed(round_, point):
    return 9000000 + 10000 * int(round_) + 100 * int(point)


def rate_order(round_, workload, count):
    """The pre-generated random order of a rate grid's indices; the same for every layout of a round."""
    order = list(range(count))
    random.Random(1000000 * int(round_) + 10000 * int(workload) + 99).shuffle(order)
    return order


def without_flags(args, flags):
    """args without the given flags ({flag: number of values it takes})."""
    out, skip = [], 0
    for arg in args:
        if skip:
            skip -= 1
        elif arg in flags:
            skip = flags[arg]
        else:
            out.append(arg)
    return out


def phase_spec(protocol, phase, round_=None):
    """Server flags, environment and client suites of a protocol phase (a PHASE_PLAN token). Every phase gets
    the same launcher and common flags, --max-num-seqs, --gpu-memory-utilization and --api-server-count;
    performance phases share one --max-model-len, capacity phases fit it to the KV cache (-1). A phase with
    two servers runs `peer_args` on node1 behind lb_proxy.py: the same server (proxy lb) or, for a layout with
    prefill/decode roles, the decode server (proxy pd). An mp-multinode layout runs `peer_args` on node1 as
    headless DP engines of node0's server: node0's flags without the API-server flags, plus --headless and the
    first DP rank of node1 (server.launchers)."""
    entry = phase_entry(protocol, phase)
    layout = resolve_layout(protocol, entry["layout"])
    server = protocol["server"]
    length = server["max_model_len"][entry.get("max_model_len", "performance")]
    seqs = layout.get("max_num_seqs", server["max_num_seqs"])
    batched = layout.get("max_num_batched_tokens", server["max_num_batched_tokens"])
    options = entry.get("options", {})
    replicas = entry.get("replicas", 1)
    roles = layout.get("roles") or {}
    launcher = layout.get("launcher", "")
    multinode = server["launchers"][launcher] if launcher else {}

    def flags(extra):
        return [*server["launcher_flags"], "--max-model-len", str(length), "--gpu-memory-utilization",
                str(server["gpu_memory_utilization"]), "--max-num-seqs", str(seqs), *layout["args"], *extra,
                "--max-num-batched-tokens", str(batched), *server["common_flags"]]

    if multinode:
        args = flags(multinode["engine_args"])
        local = args[args.index("--data-parallel-size-local") + 1]
        peer = [*without_flags(args, multinode["api_only_flags"]),
                *(local if arg == "@dp_local" else arg for arg in multinode["peer_args"])]
    else:
        args = flags(roles.get("prefill", []))
        peer = (flags(roles.get("decode", [])) if roles else args) if replicas > 1 else []
    suites = phase_suites(protocol, phase, round_)
    return {"phase": phase, "label": entry.get("label", entry["layout"]), "layout": entry["layout"],
            "rates_key": "Dx2" if options.get("slo", {}).get("rate_set") == "dx2" else rates_key(protocol, entry["layout"]),
            "checkpoint": layout["checkpoint"], "nodes": layout["nodes"], "replicas": replicas,
            "max_model_len": length, "max_num_seqs": seqs, "gpu_memory_utilization": server["gpu_memory_utilization"],
            "max_num_batched_tokens": batched, "args": args, "peer_args": peer, "launcher": launcher,
            "proxy": entry.get("proxy") or ("lb" if replicas > 1 else ""),
            "env": {**server["env"], **server["benchmark_env"], **layout.get("env", {})},
            "node_env": [*layout.get("node_env", []), *multinode.get("node_env", [])],
            "experimental": bool(layout.get("experimental")),
            "length_fallback": "screen" in suites and length != -1, "suites": suites, "options": options}


def shell_spec(spec):
    """The spec as bash assignments for `eval` in protocol.sh (every value shell-quoted)."""
    scalars = {"SPEC_LABEL": spec["label"], "SPEC_LAYOUT": spec["layout"], "SPEC_RATES_KEY": spec["rates_key"],
               "SPEC_CHECKPOINT": spec["checkpoint"], "SPEC_NODES": spec["nodes"], "SPEC_REPLICAS": spec["replicas"],
               "SPEC_LEN": spec["max_model_len"], "SPEC_SEQS": spec["max_num_seqs"],
               "SPEC_MEM": spec["gpu_memory_utilization"], "SPEC_BATCHED": spec["max_num_batched_tokens"],
               "SPEC_PROXY": spec["proxy"], "SPEC_LAUNCHER": spec["launcher"], "SPEC_EXPERIMENTAL": int(spec["experimental"]),
               "SPEC_FALLBACK": int(spec["length_fallback"]),
               "SPEC_SUITES": ",".join(spec["suites"]), "SPEC_OPTIONS": json.dumps(spec["options"])}
    lines = [f"{name}={shlex.quote(str(value))}" for name, value in scalars.items()]
    lines.append("SPEC_ARGS=(" + " ".join(shlex.quote(arg) for arg in spec["args"]) + ")")
    lines.append("SPEC_PEER_ARGS=(" + " ".join(shlex.quote(arg) for arg in spec["peer_args"]) + ")")
    lines.append("SPEC_ENV=(" + " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(spec["env"].items())) + ")")
    lines.append("SPEC_NODE_ENV=(" + " ".join(shlex.quote(name) for name in spec["node_env"]) + ")")
    return "\n".join(lines)


def round_layouts(protocol, text=None):
    """The candidates compared with A: ROUND_LAYOUTS ("C4,D24", printed by `bench_stats.py select`) or the
    protocol's list. Each must be a layout of the table that is neither A, a sensitivity nor experimental."""
    names = [n.strip() for n in (text or "").split(",") if n.strip()] or list(protocol["rounds"]["layouts"])
    for name in names:
        layout = resolve_layout(protocol, name) if name in protocol["layouts"] else None
        if layout is None or layout["root"] == "A" or layout.get("sensitivity") or layout.get("experimental"):
            raise ValueError(f"{name} cannot be a round candidate (unknown, A, sensitivity or experimental layout)")
    if len(set(names)) != len(names):
        raise ValueError(f"ROUND_LAYOUTS repeats a layout: {','.join(names)}")
    return names


def round_plan(protocol, round_, layouts=None):
    """A -> the candidates in the round's order -> sensitivity runs (round 1) -> A2 -> capacity phases (round 1)."""
    rounds, round_ = protocol["rounds"], int(round_)
    if not 1 <= round_ <= rounds["maximum"]:
        raise ValueError(f"round {round_} is outside the pre-registered rounds 1..{rounds['maximum']}")
    names = round_layouts(protocol, layouts) if layouts is None or isinstance(layouts, str) else list(layouts)
    plan = ["slo-A", *(f"slo-{name}" for name in ordered(names, round_))]
    if round_ in rounds["sensitivity_rounds"]:
        plan += [f"slo-{sensitivity_of(protocol, name)}" for name in names if sensitivity_of(protocol, name)]
    plan.append("slo-A2")
    if round_ in rounds["capacity_rounds"]:
        plan += ["cap-A", *(f"cap-{name}" for name in names)]
    return plan


def expand_plan(protocol, tokens, round_, layouts=None):
    """PHASE_PLAN with every `slo-round` token replaced by the round's phases; unknown tokens are refused."""
    expanded = []
    for token in tokens:
        if token == "slo-round":
            expanded += round_plan(protocol, round_, layouts)
        else:
            phase_entry(protocol, token)
            expanded.append(token)
    return expanded


def parse_rates(text):
    """'W1=0.25,0.6,1,1.5;W2=0.05,0.12' -> {'W1': [...], 'W2': [...]} (rate-grid overrides, req/s)."""
    rates = {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        name, values = part.split("=", 1)
        rates[name.strip()] = sorted(float(v) for v in values.split(",") if v.strip())
    return rates


def parse_soak_rates(text):
    """'A=1.1,C2=1.3,D=0.9' -> {'A': 1.1, ...} (0.8 x lambda* of W1 from the earlier rounds, req/s)."""
    return {k.strip(): float(v) for k, v in (p.split("=", 1) for p in (text or "").split(",") if "=" in p)}


# ---------------------------------------------------------------------------------------------------------------
# /metrics per engine, benchmark command and validity gates

def parse_prometheus(text, prefix="vllm:"):
    """[(name, labels, value)] of every sample line of a Prometheus text page whose name starts with prefix."""
    samples = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        found = SAMPLE_LINE.match(line.strip())
        if not found:
            continue
        name, labels, value = found.groups()
        try:
            samples.append((name, dict(LABEL_PAIR.findall(labels or "")), float(value)))
        except ValueError:
            continue
    return samples


def engine_key(labels):
    """One key per engine: vLLM labels every series with `engine` (the DP rank); lb_proxy.py adds `replica`."""
    engine = labels.get("engine", "0")
    return f"{labels['replica']}:{engine}" if "replica" in labels else engine


def engine_values(samples):
    """{engine: {metric: value}} for the tracked metrics. Engines are never added together; label sets inside
    one engine (finished_reason, ...) are."""
    values = {}
    for name, labels, value in samples:
        if name in TRACKED:
            slot = values.setdefault(engine_key(labels), {})
            slot[name] = slot.get(name, 0.0) + value
    return values


def fetch_metrics(base, timeout=10):
    return urllib.request.urlopen(base + "/metrics", timeout=timeout).read().decode()


class MetricsSampler(threading.Thread):
    """Samples /metrics every `interval` seconds, per engine. Each sample becomes one JSONL line
    ({"t": unix time, "engines": {engine: {metric: value}}}) when a path is given; peaks are kept per engine."""

    def __init__(self, base, interval=1.0, path=None):
        super().__init__(daemon=True)
        self.base, self.interval, self.path, self.done = base, interval, path, threading.Event()
        self.peak, self.peak_total = {}, {name: 0.0 for name in GAUGES}
        self.samples = self.errors = 0

    def run(self):
        handle = open(self.path, "a", encoding="utf-8") if self.path else None
        try:
            while not self.done.is_set():
                began = time.time()
                try:
                    self.add(engine_values(parse_prometheus(fetch_metrics(self.base, timeout=5))), began, handle)
                except Exception:
                    self.errors += 1
                self.done.wait(max(0.0, self.interval - (time.time() - began)))
        finally:
            if handle:
                handle.close()

    def add(self, values, when, handle=None):
        for engine, metrics in values.items():
            slot = self.peak.setdefault(engine, {name: 0.0 for name in GAUGES})
            for name in GAUGES:
                slot[name] = max(slot[name], metrics.get(name, 0.0))
        for name in GAUGES:
            self.peak_total[name] = max(self.peak_total[name], sum(m.get(name, 0.0) for m in values.values()))
        if handle:
            handle.write(json.dumps({"t": round(when, 3), "engines": values}) + "\n")
            handle.flush()
        self.samples += 1

    def summary(self):
        """Peaks are the largest per-engine value (never a sum over engines); the sum over engines is reported
        separately and labelled as such."""
        def peak(name):
            return max((slot[name] for slot in self.peak.values()), default=0.0)
        return {"peak_running_requests": peak("vllm:num_requests_running"),
                "peak_waiting_requests": peak("vllm:num_requests_waiting"),
                "peak_kv_cache_usage": round(peak("vllm:kv_cache_usage_perc"), 4),
                "peak_running_requests_sum_over_engines": self.peak_total["vllm:num_requests_running"],
                "engines": len(self.peak), "metrics_samples": self.samples, "metrics_errors": self.errors,
                "per_engine_peaks": {engine: {name.split(":", 1)[1]: round(value, 4) for name, value in slot.items()}
                                     for engine, slot in sorted(self.peak.items())}}


def bench_command(args, filename, concurrency, prompts, input_len, output_len, seed, request_rate=None,
                  burstiness=None, goodput=None, sampling=None, range_ratio=None, metadata=None, dataset=None):
    """`vllm bench serve` with per-request data always saved (--save-detailed) and never --num-warmups, which
    resends prompt 0 and lets it hit the prefix cache. A request rate makes the run open-loop (Poisson for
    burstiness 1) without a concurrency cap; otherwise the concurrency is a closed loop."""
    command = ["vllm", "bench", "serve", "--backend", "openai", "--base-url", args.base,
               "--endpoint", "/v1/completions", "--model", MODEL, "--tokenizer", args.tokenizer]
    if dataset:
        command += ["--dataset-name", "custom", "--dataset-path", str(dataset), "--custom-output-len", str(output_len),
                    "--skip-chat-template", "--disable-shuffle"]
    else:
        command += ["--dataset-name", "random", "--random-input-len", str(input_len),
                    "--random-output-len", str(output_len)]
        if range_ratio is not None:
            command += ["--random-range-ratio", str(range_ratio)]
    command += ["--num-prompts", str(prompts)]
    if request_rate is None:
        command += ["--max-concurrency", str(concurrency)]
    else:
        command += ["--request-rate", str(request_rate), "--burstiness", str(burstiness or 1.0)]
    command += ["--ignore-eos", "--seed", str(seed), "--percentile-metrics", "ttft,tpot,itl,e2el",
                "--metric-percentiles", "50,90,95,99", "--save-result", "--save-detailed", "--disable-tqdm",
                "--result-dir", args.out_dir, "--result-filename", filename]
    for key in ("temperature", "top_p"):
        if (sampling or {}).get(key) is not None:
            command += [f"--{key.replace('_', '-')}", str(sampling[key])]
    if goodput:
        command += ["--goodput", *(f"{key}:{value}" for key, value in goodput.items())]
    if metadata:
        command += ["--metadata", *(f"{key}={value}" for key, value in metadata.items())]
    return command


def bench(args, name, concurrency, prompts, input_len, output_len, timeout, seed, **options):
    """One benchmark point with the 1-s per-engine /metrics time series. options: request_rate, burstiness,
    goodput ({"ttft": ms, "tpot": ms}), sampling ({"temperature", "top_p"}), range_ratio, metadata, dataset."""
    filename = f"bench-{args.tag}-{name}.json"
    command = bench_command(args, filename, concurrency, prompts, input_len, output_len, seed, **options)
    metrics_path = Path(args.out_dir) / f"metrics-{args.tag}-{name}.jsonl"
    sampler = MetricsSampler(args.base, interval=1.0, path=metrics_path)
    sampler.start()
    began = time.time()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
        returncode, stderr = run.returncode, run.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, stderr = -1, f"timeout after {timeout}s: {exc}"
    ended = time.time()
    sampler.done.set()
    sampler.join(10)
    rate = options.get("request_rate")
    record = {"name": name, "concurrency": None if rate is not None else concurrency, "request_rate": rate,
              "num_prompts": prompts, "input_len": input_len, "output_len": output_len, "seed": seed,
              "returncode": returncode, "wall_seconds": round(ended - began, 1), "t_start_unix": round(began, 3),
              "t_end_unix": round(ended, 3), "metrics_jsonl": metrics_path.name, "command": command[3:],
              **sampler.summary()}
    path = Path(args.out_dir) / filename
    if returncode == 0 and path.exists():
        data = json.loads(path.read_text())
        record.update({k: data.get(k) for k in BENCH_KEYS})
        record["result_json"] = str(path)
    else:
        record["stderr_tail"] = stderr[-1500:]
    print(json.dumps({k: v for k, v in record.items() if k not in ("command", "per_engine_peaks")}), flush=True)
    return record


def reset_prefix_cache(base):
    """POST /reset_prefix_cache (served only with VLLM_SERVER_DEV_MODE=1); True when the server accepted it."""
    request = urllib.request.Request(base + "/reset_prefix_cache", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def snapshot(base, path=None):
    """Per-engine values of the tracked metrics; the vllm:* lines of the page are kept at `path`."""
    try:
        text = fetch_metrics(base, timeout=30)
    except Exception:
        return None
    if path:
        Path(path).write_text("\n".join(l for l in text.splitlines() if l.startswith("vllm:")) + "\n")
    return engine_values(parse_prometheus(text))


def counter_deltas(before, after):
    """Increase of every counter and histogram sum/count between two snapshots, per engine and in total."""
    per_engine = {}
    for engine, metrics in after.items():
        prior = before.get(engine, {})
        per_engine[engine] = {name: round(value - prior.get(name, 0.0), 6) for name, value in metrics.items()
                              if name not in GAUGES}
    totals = {}
    for metrics in per_engine.values():
        for name, value in metrics.items():
            totals[name] = round(totals.get(name, 0.0) + value, 6)
    return {"per_engine": per_engine, "totals": totals}


def wait_idle(base, timeout=900, interval=2.0):
    """(idle, seconds): True once two consecutive samples show no running or waiting request on any engine."""
    began, quiet = time.time(), 0
    while True:
        try:
            values = engine_values(parse_prometheus(fetch_metrics(base)))
            busy = sum(m.get("vllm:num_requests_running", 0.0) + m.get("vllm:num_requests_waiting", 0.0)
                       for m in values.values())
            quiet = quiet + 1 if values and busy == 0 else 0
        except Exception:
            quiet = 0
        if quiet >= 2:
            return True, round(time.time() - began, 1)
        if time.time() - began > timeout:
            return False, round(time.time() - began, 1)
        time.sleep(interval)


def ib_counters(root="/sys/class/infiniband"):
    """{device/port: [bytes sent, bytes received]} of this node (the sysfs data counters count 4-byte words)."""
    counters = {}
    for path in sorted(Path(root).glob("*/ports/*/counters")):
        try:
            counters[f"{path.parts[-4]}/{path.parts[-2]}"] = [
                4 * int((path / name).read_text().strip()) for name in ("port_xmit_data", "port_rcv_data")]
        except (OSError, ValueError):
            continue
    return counters


def ib_delta(before, after):
    sent = sum(after[p][0] - before[p][0] for p in after if p in before)
    received = sum(after[p][1] - before[p][1] for p in after if p in before)
    return {"ports": len(after), "sent_gb": round(sent / 1e9, 3), "received_gb": round(received / 1e9, 3)}


def evaluate_gates(record, kind, gates):
    """Validity gates of one point (slo_protocol.json "gates"). Failures invalidate the point; flags
    (capacity-bound, kv-bound, failed requests of an open-loop point) only mark it."""
    failed, flags = [], []
    totals = (record.get("deltas") or {}).get("totals", {})
    if record.get("deltas") is None:
        failed.append("no /metrics snapshot before or after the point")
    if not record.get("idle_before"):
        failed.append("the server was not idle before the point")
    if not record.get("prefix_cache_reset"):
        failed.append("POST /reset_prefix_cache failed")
    if record.get("returncode") != 0:
        failed.append(f"vllm bench serve exited with {record.get('returncode')}")
    errors = record.get("failed") or 0
    if kind == "closed" and errors > gates["closed_loop_failed_max"]:
        failed.append(f"{errors} failed requests in a closed-loop point")
    elif errors:
        flags.append(f"{errors} failed requests (SLO misses)")
    client, server = record.get("total_input_tokens"), totals.get("vllm:prompt_tokens_total")
    per_engine = (record.get("deltas") or {}).get("per_engine", {})
    if any(engine.startswith("decode:") for engine in per_engine):
        # Prefill/decode disaggregation: the decode server receives every prompt again and reads its KV cache
        # from the prefill server, so only the prefill server's count must match the client's.
        server = sum(values.get("vllm:prompt_tokens_total", 0.0) for engine, values in per_engine.items()
                     if not engine.startswith("decode:"))
    if not errors and client and server is not None and abs(server - client) > gates["prompt_token_tolerance"] * client:
        failed.append(f"the server counted {server:.0f} prompt tokens, the client {client}")
    queries, hits = totals.get("vllm:prefix_cache_queries_total") or 0, totals.get("vllm:prefix_cache_hits_total") or 0
    if queries and hits / queries > gates["prefix_cache_hit_fraction_max"]:
        failed.append(f"{hits / queries:.2%} of the queried prompt tokens hit the prefix cache")
    if (totals.get("vllm:num_preemptions_total") or 0) > 0:
        flags.append("capacity-bound")
    concurrency = record.get("concurrency")
    if (record.get("peak_kv_cache_usage") or 0) >= gates["kv_bound_usage"] or (
            kind == "closed" and concurrency and record.get("peak_running_requests_sum_over_engines", 0) < concurrency):
        flags.append("kv-bound")
    return {"valid": not failed, "failed": failed, "flags": flags}


# ---------------------------------------------------------------------------------------------------------------
# Measured points

def protocol_context(args):
    if not isinstance(getattr(args, "round", None), int):
        raise ValueError("protocol suites need --round")
    return load_protocol(args.protocol)


def bench_options(args, protocol, workload, key, rate=None, slo_workload=None, dataset=None):
    """vllm bench serve options of a protocol point: explicit sampling, goodput at the SLO tier, metadata."""
    sampling = protocol["sampling"]
    options = {"sampling": {"temperature": sampling["temperature"], "top_p": sampling["top_p"]},
               "metadata": {"layout": args.layout or args.tag, "round": args.round, "phase": args.phase or args.tag,
                            "workload": workload, "point": key, "commit": args.commit or "unknown"}}
    if dataset is None:
        options["range_ratio"] = sampling["random_range_ratio"]
    else:
        options["dataset"] = dataset
    if rate is not None:
        options.update(request_rate=rate, burstiness=1.0)
    if slo_workload:
        tier = protocol["slo"]["tiers"][args.slo or protocol["slo"]["bench_goodput_tier"]][slo_workload]
        options["goodput"] = {"ttft": tier["ttft_ms"], "tpot": tier["tpot_ms"]}
    return options


def run_point(args, suite, point, protocol):
    """Idle server -> POST /reset_prefix_cache -> /metrics snapshot -> bench (1-s per-engine sampler) ->
    snapshot -> gates. An invalid point is rerun once with the same seed; every attempt is reported."""
    gates, attempts = protocol["gates"], []
    for attempt in (1, 2):
        name = point["name"] if attempt == 1 else f"{point['name']}-retry"
        idle, idle_seconds = wait_idle(args.base, gates["idle_timeout_seconds"])
        reset = reset_prefix_cache(args.base)
        prefix = Path(args.out_dir) / f"metrics-{args.tag}-{name}"
        before, ib_before = snapshot(args.base, f"{prefix}-before.prom"), ib_counters()
        record = bench(args, name, point.get("concurrency"), point["prompts"], point["input_len"], point["output_len"],
                       point["timeout"], point["seed"],
                       **bench_options(args, protocol, point["workload"], point["key"], point.get("rate"),
                                       point.get("slo_workload"), point.get("dataset")))
        after, ib_after = snapshot(args.base, f"{prefix}-after.prom"), ib_counters()
        record.update({"record_type": "bench_point", "suite": suite, "phase": args.phase or args.tag,
                       "layout": args.layout or args.tag, "round": args.round, "workload": point["workload"],
                       "point_key": point["key"], "point_index": point["index"], "kind": point["kind"],
                       "window": point.get("window"), "slo_workload": point.get("slo_workload"),
                       "capacity_point": point.get("capacity_point", False), "attempt": attempt,
                       "idle_before": idle, "idle_wait_seconds": idle_seconds, "prefix_cache_reset": reset,
                       "deltas": counter_deltas(before, after) if before is not None and after is not None else None,
                       "ib_node0": ib_delta(ib_before, ib_after), "dataset": point.get("dataset_source")})
        record["gates"] = evaluate_gates(record, point["kind"], gates)
        record.update(valid=record["gates"]["valid"], flags=record["gates"]["flags"])
        vs.emit(args.out_dir, args.tag, f"{suite}-{name}", record)
        attempts.append(record)
        if record["valid"] or not gates["rerun_invalid_once"] or not vs.server_healthy(args.base):
            break
    return attempts


def run_points(args, suite, points, protocol, method):
    results, plan = [], [p["name"] for p in points]
    for point in points:
        if point.get("prepare"):
            point.update(point.pop("prepare")())
        attempts = run_point(args, suite, point, protocol)
        results += attempts
        if attempts[-1]["returncode"] != 0 and not vs.server_healthy(args.base):
            return {"method": method, "plan": plan, "points": results,
                    "aborted": f"the server stopped answering /health after {attempts[-1]['name']}"}
    return {"method": method, "plan": plan, "points": results}


def partr_points(protocol, round_, subset=None, all_rounds=False):
    """Part R: the published matrix with round seeds; 32K x 32 only in the rounds the protocol lists (every
    point with all_rounds, as the screening runs it)."""
    workload = protocol["workloads"]["R"]
    wanted = set(workload["subsets"][subset]) if subset else None
    points = []
    for index, item in enumerate(workload["points"]):
        if (item.get("rounds") and round_ not in item["rounds"] and not all_rounds) or \
                (wanted is not None and item["name"] not in wanted):
            continue
        points.append({"name": f"R-{item['name']}", "key": item["name"], "index": index, "workload": "R",
                       "kind": "closed", "concurrency": item["concurrency"], "prompts": item["prompts"],
                       "input_len": item["input_len"], "output_len": workload["output_len"],
                       "timeout": item["timeout"], "seed": protocol_seed(round_, workload["id"], index),
                       "capacity_point": bool(item.get("capacity_point"))})
    return points


def open_point(protocol, round_, name, index, rate):
    workload = protocol["workloads"][name]
    return {"name": f"{name}-p{index}-r{rate:g}", "key": f"p{index}", "index": index, "workload": name,
            "kind": "open", "rate": rate, "input_len": workload["input_len"], "output_len": workload["output_len"],
            "prompts": max(workload["min_prompts"], math.ceil(rate * workload["arrival_seconds"])),
            "window": workload["window"], "timeout": workload["arrival_seconds"] + workload["drain_timeout"],
            "seed": protocol_seed(round_, workload["id"], index), "slo_workload": workload.get("slo_from", name)}


def slo_points(protocol, round_, workloads, rates=None, point_filter=None, rate_set=None):
    """Closed-loop S1 points and open-loop rate grids in their pre-generated random order."""
    points = []
    for name in workloads:
        workload = protocol["workloads"][name]
        if workload["kind"] == "closed":
            for index, concurrency in enumerate(workload["concurrency"]):
                points.append({"name": f"{name}-c{concurrency}", "key": f"c{concurrency}", "index": index,
                               "workload": name, "kind": "closed", "concurrency": concurrency,
                               "prompts": workload["prompts_per_concurrency"] * concurrency,
                               "input_len": workload["input_len"], "output_len": workload["output_len"],
                               "timeout": workload["timeout"], "seed": protocol_seed(round_, workload["id"], index),
                               "slo_workload": workload["slo_from"]})
            continue
        grid = (rates or {}).get(name) or (protocol["rate_sets"][rate_set][name] if rate_set else workload["rates"])
        allowed = (point_filter or {}).get(name)
        for index in rate_order(round_, workload["id"], len(grid)):
            if allowed is None or index in allowed:
                points.append(open_point(protocol, round_, name, index, grid[index]))
    return points


def suite_partr(args):
    protocol = protocol_context(args)
    points = partr_points(protocol, args.round, args.options.get("partr"))
    return run_points(args, "partr", points, protocol, "Part R: " + protocol["workloads"]["R"]["description"])


def suite_warmup(args):
    """Discarded bursts that warm the kernels of each input length, then a prefix-cache reset."""
    protocol = protocol_context(args)
    workload, runs = protocol["workloads"]["warmup"], []
    for index, isl in enumerate(workload["input_lens"]):
        record = bench(args, f"warmup-isl{isl}", workload["concurrency"], workload["prompts"], isl,
                       workload["output_len"], workload["timeout"], warmup_seed(args.round, index),
                       **bench_options(args, protocol, "warmup", f"isl{isl}"))
        runs.append({k: record.get(k) for k in ("name", "seed", "returncode", "completed", "failed", "wall_seconds")})
    return {"method": workload["description"], "warmups": runs, "prefix_cache_reset": reset_prefix_cache(args.base)}


def suite_slo(args):
    protocol = protocol_context(args)
    options = args.options.get("slo", {})
    filters = {name[:-len("_points")]: set(value) for name, value in options.items() if name.endswith("_points")}
    points = slo_points(protocol, args.round, options.get("workloads", ["S1", "W1", "W2"]), args.rates, filters,
                        options.get("rate_set"))
    tier = args.slo or protocol["slo"]["bench_goodput_tier"]
    return run_points(args, "slo", points, protocol,
                      f"closed-loop S1 and open-loop Poisson grids, --goodput at the {tier} SLO tier")


def suite_soak(args):
    protocol = protocol_context(args)
    if args.round != protocol["rounds"]["soak_round"]:
        return {"skipped": f"the soak runs in round {protocol['rounds']['soak_round']} only"}
    rate = args.soak_rates.get(args.rates_key or args.layout)
    if not rate:
        return {"skipped": "no soak rate for this layout (SOAK_RATES=<layout>=<req/s>,...; 0.8 x lambda* of W1)",
                "missing": True}
    point = open_point(protocol, args.round, "soak", 0, rate)
    point["name"] = f"soak-r{rate:g}"
    return run_points(args, "soak", [point], protocol, protocol["workloads"]["soak"]["description"])


# ---------------------------------------------------------------------------------------------------------------
# Real text and parity

def tokenize(base, text, special):
    return vs.call(base, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": special}, 600)[0]["tokens"]


def detokenize(base, ids):
    return vs.call(base, "/detokenize", {"model": MODEL, "tokens": ids}, 600)[0]["prompt"]


def realtext_windows(base, corpus, length, stride):
    """(document id, first token, token ids) of every `length`-token window, `stride` tokens apart. No two
    windows start alike, so no two prompts share a cached prefix."""
    windows = []
    for doc in corpus["documents"]:
        ids = tokenize(base, doc["text"], False)
        windows += [(doc["id"], start, ids[start:start + length]) for start in range(0, len(ids) - length + 1, stride)]
    return windows


def realtext_dataset(args, source, count, seed, output_len, path):
    """A custom-dataset JSONL of `count` distinct prompts drawn with the point's seed: corpus windows
    (detokenized by the server) or lines of a customer JSONL (--realtext-path)."""
    rng = random.Random(seed)
    if isinstance(source, list):
        if count > len(source):
            raise ValueError(f"{count} prompts requested but the corpus has {len(source)} windows")
        chosen = rng.sample(source, count)
        with futures.ThreadPoolExecutor(16) as pool:
            texts = list(pool.map(lambda window: detokenize(args.base, window[2]), chosen))
        labels, kind = [f"{doc}@{start}" for doc, start, _ in chosen], "parity_corpus.json windows"
    else:
        lines = [json.loads(l)["prompt"] for l in Path(source).read_text(encoding="utf-8").splitlines() if l.strip()]
        if count > len(lines):
            raise ValueError(f"{count} prompts requested but {source} has {len(lines)} lines")
        picks = rng.sample(range(len(lines)), count)
        texts, labels, kind = [lines[i] for i in picks], [f"line{i}" for i in picks], "customer JSONL"
    with open(path, "w", encoding="utf-8") as handle:
        for text, label in zip(texts, labels):
            handle.write(json.dumps({"prompt": text, "output_tokens": output_len, "source": label},
                                    ensure_ascii=False) + "\n")
    return {"dataset": str(path), "dataset_source": kind}


def suite_realtext(args):
    protocol = protocol_context(args)
    workload = protocol["workloads"]["W1real"]
    if args.realtext_path:
        source = args.realtext_path
    else:
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        source = realtext_windows(args.base, corpus, workload["input_len"], workload["stride_tokens"])
    grid, points = args.rates.get("W1real") or workload["rates"], []
    for index in rate_order(args.round, workload["id"], len(grid)):
        point = open_point(protocol, args.round, "W1real", index, grid[index])
        path = Path(args.out_dir) / f"realtext-{args.tag}-{point['name']}.jsonl"
        point["prepare"] = (lambda p=point, file=path: realtext_dataset(args, source, p["prompts"], p["seed"],
                                                                         p["output_len"], file))
        points.append(point)
    return run_points(args, "realtext", points, protocol, workload["description"])


def teacher_forced(base, name, ids):
    """P1: prompt log-probabilities of a fixed token sequence (one generated token, greedy)."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 20,
            "return_tokens_as_token_ids": True, "cache_salt": f"parity-p1-{name}"}
    payload, seconds = vs.call(base, "/v1/completions", body, 3600)
    entries = payload["choices"][0].get("prompt_logprobs") or []
    chosen, rank, top1, top1_logprob = [], [], [], []
    for position in range(1, len(ids)):
        entry = (entries[position] if position < len(entries) else None) or {}
        token = entry.get(str(ids[position]))
        best = min(entry.items(), key=lambda kv: kv[1].get("rank", 1 << 30)) if entry else None
        chosen.append(round(token["logprob"], 5) if token else None)
        rank.append(token.get("rank") if token else None)
        top1.append(int(best[0]) if best else None)
        top1_logprob.append(round(best[1]["logprob"], 5) if best else None)
    return {"tokens": len(ids), "chosen": chosen, "rank": rank, "top1": top1, "top1_logprob": top1_logprob,
            "seconds": round(seconds, 2)}


def greedy(base, name, ids, max_tokens):
    """P2: greedy continuation of a fixed prompt, as token ids."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 1,
            "return_tokens_as_token_ids": True, "cache_salt": f"parity-p2-{name}"}
    payload, seconds = vs.call(base, "/v1/completions", body, 3600)
    choice = payload["choices"][0]
    return {"prompt_tokens": len(ids), "generated": [int(t.split(":", 1)[1]) for t in choice["logprobs"]["tokens"]],
            "finish_reason": choice.get("finish_reason"), "seconds": round(seconds, 2)}


def parity_documents(base, corpus, long):
    """(name, token ids) of the teacher-forced inputs: every document cut to tokens_per_document, or the two
    long documents of the corpus rule (join in order / in reverse order, cycle until long enough, cut)."""
    if not long:
        size = corpus["tokens_per_document"]
        return [(doc["id"], tokenize(base, doc["text"], True)[:size]) for doc in corpus["documents"]]
    rule, documents, result = corpus["long_documents"], corpus["documents"], []
    for name, order in zip(rule["names"], (documents, documents[::-1])):
        block = rule["separator"].join(doc["text"] for doc in order)
        text, ids = block, tokenize(base, block, True)
        while len(ids) < rule["tokens"]:
            text += rule["separator"] + block
            ids = tokenize(base, text, True)
        result.append((name, ids[:rule["tokens"]]))
    return result


def parity_summary(raw, path):
    p1 = {name: {"tokens": doc["tokens"],
                 "mean_chosen_logprob": round(sum(v for v in doc["chosen"] if v is not None)
                                              / max(1, sum(v is not None for v in doc["chosen"])), 5)}
          for name, doc in raw["p1"].items()}
    return {"method": "teacher-forced prompt_logprobs (P1) and greedy continuations (P2) of parity_corpus.json; "
                      "bench_stats.py compares them with slo-A of the same round", "raw": path.name, "p1": p1,
            "p2_prompts": len(raw.get("p2", {}))}


def parity_raw(args, long):
    return {"record_type": "parity", "phase": args.phase or args.tag, "layout": args.layout or args.tag,
            "round": args.round, "long": long, "p1": {}, "p2": {}}


def suite_parity(args):
    protocol = protocol_context(args)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    settings, raw = protocol["parity"]["p2"], parity_raw(args, False)
    reset_prefix_cache(args.base)
    for name, ids in parity_documents(args.base, corpus, long=False):
        raw["p1"][name] = teacher_forced(args.base, name, ids)
    reset_prefix_cache(args.base)
    for doc in corpus["documents"]:
        ids = tokenize(args.base, doc["text"], True)
        for start in (0, 1024):
            name = f"{doc['id']}@{start}"
            raw["p2"][name] = greedy(args.base, name, ids[start:start + settings["prompt_tokens"]], settings["max_tokens"])
    reset_prefix_cache(args.base)
    path = Path(args.out_dir) / f"parity-{args.tag}.json"
    path.write_text(json.dumps(raw))
    return parity_summary(raw, path)


def suite_paritylong(args):
    protocol_context(args)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    raw = parity_raw(args, True)
    reset_prefix_cache(args.base)
    for name, ids in parity_documents(args.base, corpus, long=True):
        raw["p1"][name] = teacher_forced(args.base, name, ids)
    reset_prefix_cache(args.base)
    path = Path(args.out_dir) / f"parity-{args.tag}-long.json"
    path.write_text(json.dumps(raw))
    return parity_summary(raw, path)


# ---------------------------------------------------------------------------------------------------------------
# Screening

def screen_facts(base):
    """What the server reports about itself: the served max_model_len and, per engine, the KV cache size in
    tokens (vllm:cache_config_info: num_gpu_blocks x block_size)."""
    models, _ = vs.call(base, "/v1/models", None, 60)
    kv = {}
    for name, labels, _ in parse_prometheus(fetch_metrics(base, timeout=30)):
        if name == "vllm:cache_config_info" and str(labels.get("num_gpu_blocks", "")).isdigit():
            kv[engine_key(labels)] = int(labels["num_gpu_blocks"]) * int(labels.get("block_size") or 0)
    return {"max_model_len": models["data"][0].get("max_model_len"), "kv_tokens_per_engine": kv}


def screen_parity(args, protocol):
    """P1 for the screening documents cut to 2,048 tokens and P2 for their first 512 tokens (128 greedy tokens);
    bench_stats.py compares the raw data with the job's first launch."""
    settings = protocol["screen"]["parity"]
    documents = {doc["id"]: doc for doc in json.loads(CORPUS.read_text(encoding="utf-8"))["documents"]}
    raw = {**parity_raw(args, False), "screen": True}
    ids = {name: tokenize(args.base, documents[name]["text"], True) for name in settings["documents"]}
    reset_prefix_cache(args.base)
    for name in settings["documents"]:
        raw["p1"][name] = teacher_forced(args.base, name, ids[name][:settings["tokens"]])
    reset_prefix_cache(args.base)
    for name in settings["documents"]:
        raw["p2"][f"{name}@0"] = greedy(args.base, f"{name}@0", ids[name][:settings["greedy_prompt_tokens"]],
                                        settings["greedy_tokens"])
    reset_prefix_cache(args.base)
    path = Path(args.out_dir) / f"parity-{args.tag}.json"
    path.write_text(json.dumps(raw))
    return {**parity_summary(raw, path), "method": "screening parity: P1 and short P2 of four documents; "
                                                   "bench_stats.py compares them with the job's first launch"}


def screen_points(protocol, round_, ladder=None):
    """Every Part R point (the 32K x 32 capacity point with few output tokens), then the W1 point of the
    screening or a whole W1 ladder (W1 grid or a rate set) in the round's rate order."""
    screen, workload = protocol["screen"], protocol["workloads"]["W1"]
    points = partr_points(protocol, round_, all_rounds=True)
    for point in points:
        if point["key"] == screen["partr"]["capacity_point"]:
            point.update(output_len=screen["partr"]["capacity_output_len"], capacity_point=True)
    if ladder:
        grid = workload["rates"] if ladder == "W1" else protocol["rate_sets"][ladder]["W1"]
        return points + [open_point(protocol, round_, "W1", i, grid[i]) for i in rate_order(round_, workload["id"], len(grid))]
    return points + [open_point(protocol, round_, "W1", screen["w1"]["point"], screen["w1"]["rate"])]


def suite_screen(args):
    """One screening launch: start facts, short parity, warm-up, Part R and Poisson W1 (options.screen.steps runs a
    subset). Each step that fails is recorded and the next one runs while the server answers /health."""
    protocol = protocol_context(args)
    screen = protocol["screen"]
    if args.round != screen["round"]:
        raise ValueError(f"the screening runs with ROUND={screen['round']}, apart from the rounds' seeds")
    steps, result = {}, {"method": screen["description"]}

    def step(name, function):
        began = time.time()
        try:
            value = function()
            steps[name] = {"ok": True}
            return value
        except Exception as exc:
            steps[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
            return None
        finally:
            steps[name]["seconds"] = round(time.time() - began, 1)

    options = args.options.get("screen", {})
    wanted = options.get("steps") or screen["steps"]  # scr-Dref: facts and parity only
    if "facts" in wanted:
        result["facts"] = step("facts", lambda: screen_facts(args.base))
    if "parity" in wanted:
        result["parity"] = step("parity", lambda: screen_parity(args, protocol))
    if "warmup" in wanted:
        step("warmup", lambda: suite_warmup(args))
    points = screen_points(protocol, args.round, options.get("ladder")) if "points" in wanted else []
    data = None
    if points and vs.server_healthy(args.base):
        data = step("points", lambda: run_points(args, "screen", points, protocol, "Part R and Poisson W1"))
    result.update(plan=[p["name"] for p in points], points=(data or {}).get("points", []), steps=steps)
    if points and (data is None or data.get("aborted")):
        result["aborted"] = ((data or {}).get("aborted") or steps.get("points", {}).get("error")
                             or "the server stopped answering /health before the points")
    return result


SUITES = {"parity": suite_parity, "paritylong": suite_paritylong, "warmup": suite_warmup, "partr": suite_partr,
          "slo": suite_slo, "soak": suite_soak, "realtext": suite_realtext, "screen": suite_screen}


# ---------------------------------------------------------------------------------------------------------------
# Command line

def run_suites(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    status = 0
    for name in args.suites.split(","):
        began = time.time()
        try:
            data = (SUITES.get(name) or vs.SUITES[name])(args)
            data["suite_seconds"] = round(time.time() - began, 1)
            if "points" in data:
                data = {**{k: v for k, v in data.items() if k != "points"},
                        "summary": [{k: p.get(k) for k in SUMMARY_KEYS} for p in data["points"]]}
            vs.emit(args.out_dir, args.tag, name, data)
            if data.get("aborted"):
                return 1
        except Exception as exc:
            status = 1
            vs.emit(args.out_dir, args.tag, f"{name}-error", {"error": f"{type(exc).__name__}: {exc}"[:2000],
                                                              "seconds": round(time.time() - began, 1)})
            if not vs.server_healthy(args.base):
                return status
    return status


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="slo_suites.py")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="expand PHASE_PLAN tokens (slo-round) for a round")
    plan.add_argument("names", nargs="*")
    plan.add_argument("--layouts", default="", help="ROUND_LAYOUTS: the candidates compared with A (default: the protocol's)")
    plan.add_argument("--record", help="also write {round, layouts, plan} to this JSON file (for the package)")
    spec = commands.add_parser("phase-spec", help="settings of one protocol phase")
    spec.add_argument("name")
    spec.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="run client suites against a server")
    run.add_argument("--base", default="http://127.0.0.1:8000")
    run.add_argument("--tag", required=True)
    run.add_argument("--suites", required=True, help="comma-separated: " + ",".join(SUITES) + " or verify_suites names")
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--max-model-len", type=int, required=True)
    run.add_argument("--max-num-batched-tokens", type=int, default=8192, help="the server's setting, recorded only")
    run.add_argument("--niah-lengths", default="32768,131072,262144")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--layout", default="", help="layout label recorded with every point (A, A2, C2, ...)")
    run.add_argument("--phase", default="", help="PHASE_PLAN token")
    run.add_argument("--rates-key", default="", help="layout whose rate grid and soak rate this phase uses")
    run.add_argument("--options", type=json.loads, default={}, help="the phase's suite options (JSON)")
    run.add_argument("--workloads", default="", help="comma-separated slo workloads, e.g. S1,W1,W2")
    run.add_argument("--rates", type=parse_rates, default={}, help="rate-grid overrides: W1=0.25,0.6;W2=0.05")
    run.add_argument("--slo", default="", help="SLO tier passed to --goodput (default: the protocol's)")
    run.add_argument("--soak-rates", type=parse_soak_rates, default={}, help="A=1.1,C2=1.3,D=0.9 (req/s)")
    run.add_argument("--realtext-path", default="", help="customer JSONL ({\"prompt\": ...}) for realtext")
    run.add_argument("--commit", default="", help="source commit recorded in every result")
    for sub in (plan, spec, run):
        sub.add_argument("--round", type=int, required=sub is not spec)
        sub.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "plan":
        layouts = round_layouts(protocol, args.layouts)
        expanded = expand_plan(protocol, args.names, args.round, layouts)
        if args.record:
            Path(args.record).write_text(json.dumps({"round": args.round, "tokens": args.names, "plan": expanded,
                                                     "layouts": layouts if "slo-round" in args.names else []}))
        print(" ".join(expanded))
        return 0
    if args.command == "phase-spec":
        settings = phase_spec(protocol, args.name, args.round)
        print(json.dumps(settings) if args.json else shell_spec(settings))
        return 0
    if args.workloads:
        args.options = {**args.options, "slo": {**args.options.get("slo", {}), "workloads": args.workloads.split(",")}}
    return run_suites(args)


if __name__ == "__main__":
    sys.exit(main())
