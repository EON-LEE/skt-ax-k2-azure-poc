import csv
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "aml" / "src" / "bench"), str(ROOT / "aml" / "src")]
import bench_stats as bs  # noqa: E402
del sys.path[:2]

PROTOCOL = bs.load_protocol()
W1 = PROTOCOL["slo"]["tiers"]["standard"]["W1"]
ENGINES = {"A4": 1, "C2": 2, "C4": 4, "C8": 8, "C16": 16, "D8": 8, "D24": 2, "Dx2": 2, "PD": 2}


def synthetic_point(rng, phase, workload, key, index, kind, requests, engines, rate=None, attempt=1):
    """A package point with the fields and the per-request rows compact_point() writes, sized like a real one."""
    rows, now = [], 0.0
    output = 512 if workload == "W1" else 1024
    for _ in range(requests):
        now += rng.expovariate(rate) if rate else rng.random() * 0.2
        ttft, tpot = rng.lognormvariate(-0.7, 0.6), rng.uniform(0.018, 0.09)
        rows.append([round(now, 4), round(ttft, 5), round(tpot, 6), round(ttft + tpot * (output - 1), 4), 1, output,
                     rng.randint(0, output)])
    stats = {name: round(rng.uniform(10, 900), 3) for name in bs.POINT_STATS}
    return {"id": f"{workload}/{key}", "name": f"{workload}-{key}" + (f"-r{rate:g}" if rate else ""), "phase": phase,
            "layout": phase.split("-", 1)[1], "round": 0, "workload": workload, "slo_workload": "W1", "point": key,
            "index": index, "kind": kind, "rate": rate, "concurrency": None if rate else 32, "seed": 10000 + index,
            "attempt": attempt, "capacity_point": False, "valid": True, "invalid_reasons": [], "flags": [],
            "window": [60, 360] if rate else None, "t_start_unix": 1.79e9 + rng.random() * 1e4, "t_end_unix": 1.79e9,
            "steady": {**stats, "requests": requests, "failed": 0, "window": [60, 360] if rate else None,
                       "input_tok_s": round(rng.uniform(100, 9000), 3), "p99_reportable": False,
                       "goodput": {"standard": round(rng.random(), 4), "strict": round(rng.random(), 4)}},
            "summary": {k: round(rng.uniform(0, 5000), 3) for k in ("completed", "failed", "duration", "output_throughput",
                                                                    "request_goodput", "median_ttft_ms", "median_tpot_ms",
                                                                    "median_itl_ms", "max_concurrent_requests")},
            "engine_peaks": {str(e): {"num_requests_running": rng.randint(0, 128), "num_requests_waiting": rng.randint(0, 9),
                                      "kv_cache_usage_perc": round(rng.random(), 4)} for e in range(engines)},
            "counters": {f"vllm:counter_{i}_total": round(rng.uniform(0, 1e6), 1) for i in range(18)},
            "gpu": {node: {"samples": rng.randint(100, 900), "invalid_fraction": 0.0, "power_cap_fraction": round(rng.random(), 4),
                           "sm_clock_mhz_mean": round(rng.uniform(1200, 1410), 1), "power_w_mean": round(rng.uniform(150, 400), 1),
                           "temperature_c_max": rng.randint(40, 80)} for node in ("node0", "node1")},
            "ib": {node: {"ports": 8, "sent_gb": round(rng.uniform(0, 90), 3), "received_gb": round(rng.uniform(0, 90), 3)}
                   for node in ("node0", "node1")},
            "requests": rows}


def synthetic_screening_package(job, seed=7, retries=0.15):
    """A results package shaped like one screening job's (slo_protocol.json screen.jobs[job]) after every launch."""
    rng = random.Random(seed)
    plan = PROTOCOL["screen"]["jobs"][job]["plan"]
    part_r = [(item["name"], index, 32 if item["name"] == "isl32768-c32" else item["prompts"])
              for index, item in enumerate(PROTOCOL["workloads"]["R"]["points"])]
    points, effective, nccl, clean, start, screens, parity, plans = [], {}, {}, {}, {}, {}, {}, {}
    for phase in plan:
        layout = phase.split("-", 1)[1]
        engines = ENGINES.get(layout, 1)
        options = PROTOCOL["phases"][phase].get("options", {}).get("screen", {})
        ladder = options.get("ladder")
        rates = ([PROTOCOL["screen"]["w1"]["rate"]] if not ladder else
                 PROTOCOL["workloads"]["W1"]["rates"] if ladder == "W1" else PROTOCOL["rate_sets"][ladder]["W1"])
        launch = [synthetic_point(rng, phase, "R", key, index, "closed", count, engines) for key, index, count in part_r]
        launch += [synthetic_point(rng, phase, "W1", f"p{i}", i, "open", max(60, math.ceil(r * 360)), engines, rate=r)
                   for i, r in enumerate(rates)]
        retried = [dict(p, attempt=2, name=p["name"] + "-retry") for p in launch if rng.random() < retries]
        points += launch + retried
        effective[phase] = {"max_model_len": 131072, "enforce_eager": False, "kv_tokens": [rng.randint(60000, 700000)] * engines,
                            "graph_capture_by_node": {"node0": 8}, "problems": [],
                            "engine_config": "model='/models/x', " + ", ".join(f"option_{i}={rng.random():.6f}" for i in range(120)),
                            "lines": {k: [f"line {k} {i} " + "x" * rng.randint(60, 380) for i in range(8)]
                                      for k in ("async_scheduling", "executor", "cudagraph", "kv_cache", "loading", "dp_placement",
                                                "expert_placement", "kv_transfer", "errors", "auto_fit")}}
        nccl[phase] = {node: {"files": 40, "channels": {"NET/IB": rng.randint(100, 4000), "P2P": rng.randint(100, 4000)},
                              "communicators": {f"0x{rng.getrandbits(64):016x}": {"nranks": 16, "ranks_here": 8}
                                                for _ in range(8 * engines)}} for node in ("node0", "node1")}
        clean[f"{phase}-before"] = clean[f"{phase}-after"] = {"clean": True, "port_8000_free": True, "placement_groups": 0}
        start[phase] = {"phase": phase, "started": True, "requested_max_model_len": 131072, "length_fallback": None,
                        "seconds_to_healthy": rng.randint(150, 450)}
        screens[phase] = {"facts": {"max_model_len": 131072, "kv_tokens_per_engine": {str(e): 238656 for e in range(engines)}},
                          "steps": {s: {"ok": True, "seconds": 9.5} for s in ("facts", "parity", "warmup", "points")}}
        parity[phase] = {"positions": 8188, "mean_abs_logprob_diff": 0.012, "top1_agreement": 0.986,
                         "documents": {d: {"positions": 2047, "mean_abs_logprob_diff": 0.01, "top1_agreement": 0.99}
                                       for d in PROTOCOL["screen"]["parity"]["documents"]},
                         "greedy_first_difference": {f"{d}@0": rng.choice([None, 40]) for d in PROTOCOL["screen"]["parity"]["documents"]}}
        plans[phase] = {"screen": {"plan": [p["name"] for p in launch]}}
    return {"schema": "axk2-slo-package/1", "rounds": [0], "phases": [{"phase": p, "status": "passed", "seconds": 2200} for p in plan],
            "plans": plans, "points": points, "parity": parity, "screen_reference": plan[0], "screens": screens,
            "protocol_plan": {"round": 0, "plan": plan, "layouts": []}, "hosts_sha256": {"node0": "a" * 12, "node1": "b" * 12},
            "effective": effective, "nccl": nccl, "lb_stats": {}, "clean": clean, "start": start}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detailed(starts, ttfts, itls, outputs, errors=None, inputs=None):
    count = len(starts)
    return {"start_times": [100.0 + s for s in starts], "ttfts": ttfts, "itls": itls, "output_lens": outputs,
            "input_lens": inputs or [1024] * count, "errors": errors or [""] * count, "duration": 400.0,
            "output_throughput": 123.0}


def point(round_, phase, point_id, steady=None, valid=True, flags=(), kind="closed", rate=None, attempt=1, **extra):
    workload, key = point_id.split("/")
    name = {"R": f"R-{key}", "S1": f"S1-{key}"}.get(workload, f"{workload}-{key}-r{rate}")
    return {"id": point_id, "name": name, "phase": phase, "layout": phase.split("-", 1)[1], "round": round_,
            "workload": workload, "point": key, "kind": kind, "rate": rate, "attempt": attempt, "valid": valid,
            "invalid_reasons": [] if valid else ["synthetic"], "flags": list(flags), "steady": steady,
            "window": [60, 360] if kind == "open" else None, "requests": [], "capacity_point": False, **extra}


def open_points(round_, phase, rates, goodputs, tok_s):
    return [point(round_, phase, f"W1/p{i}", kind="open", rate=r,
                  steady={"goodput": {"standard": g, "strict": g / 2}, "output_tok_s": t, "input_tok_s": 0.0,
                          "ttft_p90": 1000.0, "tpot_p90": 50.0, "p99_reportable": False})
            for i, (r, g, t) in enumerate(zip(rates, goodputs, tok_s))]


class SteadyStateTests(unittest.TestCase):
    def test_percentile_is_numpys_linear_default(self):
        self.assertEqual(bs.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(bs.percentile([1, 2, 3, 4], 90), 3.7)
        self.assertIsNone(bs.percentile([], 50))

    def test_rows_window_goodput_and_tokens(self):
        bench = detailed(starts=[10, 70, 200, 300, 380], ttfts=[0.5, 0.5, 4.0, 0.0, 0.5],
                         itls=[[0.01] * 3, [0.01] * 3, [0.2] * 3, [], [0.01] * 3], outputs=[4, 4, 4, 0, 4],
                         errors=["", "", "", "timeout", ""])
        rows = bs.request_rows(bench)
        self.assertEqual([round(r["start"]) for r in rows], [0, 60, 190, 290, 370])
        self.assertFalse(rows[3]["ok"])
        self.assertAlmostEqual(rows[2]["tpot"], 0.2)
        summary = bs.steady_state(rows, [60, 360], {"standard": W1})
        self.assertEqual((summary["requests"], summary["failed"]), (3, 1))
        # request 1 meets both limits, request 2 misses TTFT (4 s > 3 s) and TPOT (200 ms), the failure misses
        self.assertAlmostEqual(summary["goodput"]["standard"], 1 / 3, places=4)
        self.assertAlmostEqual(summary["output_tok_s"], round(8 / 300, 3))
        self.assertFalse(summary["p99_reportable"])
        everything = bs.steady_state(rows, None, {"standard": W1}, duration=400.0)
        self.assertEqual(everything["requests"], 5)

    def test_token_times_spread_bundled_chunks(self):
        row = {"start": 0.0, "ttft": 1.0, "e2e": 3.0, "ok": True, "n_out": 5, "itl": [1.0, 1.0]}
        self.assertEqual(bs.token_times(row), [1.0, 1.5, 2.0, 2.5, 3.0])
        row["itl"] = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(bs.token_times(row), [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_block_bootstrap_is_deterministic_and_covers_the_estimate(self):
        rows = [{"start": i * 0.5, "ttft": 0.1 + (i % 7) * 0.05, "ok": True} for i in range(400)]
        stat = lambda sample: bs.percentile([r["ttft"] for r in sample], 90)
        low, high = bs.block_bootstrap(rows, stat, resamples=200, seed=3)
        self.assertLessEqual(low, stat(rows))
        self.assertGreaterEqual(high, stat(rows))
        self.assertEqual((low, high), bs.block_bootstrap(rows, stat, resamples=200, seed=3))
        self.assertIsNone(bs.block_bootstrap(rows[:5], stat, resamples=10))

    def test_output_rate_bootstrap_uses_the_point_estimates_population(self):
        # Requests every 2 s for 400 s, each producing 1 token/s for 40 s: those starting before the window
        # still put tokens into it. The point estimate counts them, so the interval must as well.
        window, rows = (60, 360), []
        for i in range(200):
            start = i * 2.0
            row = {"start": start, "ttft": 1.0, "tpot": 1.0, "e2e": 40.0, "ok": True, "n_out": 40, "itl": [1.0] * 39,
                   "prompt_len": 1024}
            row["win_tokens"] = sum(1 for t in bs.token_times(row) if window[0] <= t < window[1])
            rows.append(row)
        estimate = bs.steady_state(rows, list(window))["output_tok_s"]
        compact = [[r["start"], r["ttft"], 1.0, r["e2e"], 1, r["n_out"], r["win_tokens"]] for r in rows]
        protocol = json.loads(json.dumps(PROTOCOL))
        protocol["statistics"]["bootstrap"]["resamples"] = 200
        cis = bs.bootstrap_point({"requests": compact, "window": list(window), "slo_workload": "W1"}, protocol)
        low, high = cis["output_tok_s"]
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)


class DecisionTests(unittest.TestCase):
    def test_lambda_star_interpolates_inside_the_grid_only(self):
        result = bs.lambda_star([(0.25, 1.0, 100, 900), (0.6, 0.95, 240, 2000), (1.0, 0.85, 380, 3500),
                                 (1.5, 0.4, 420, 5000)])
        self.assertTrue(result["bracketed"])
        self.assertAlmostEqual(result["value"], 0.6 + 0.5 * 0.4)
        self.assertAlmostEqual(result["output_tok_s"], 310)
        self.assertIn("ends too low", bs.lambda_star([(0.25, 1.0, 1, 1), (1.5, 0.95, 2, 2)])["reason"])
        self.assertIn("starts too high", bs.lambda_star([(0.25, 0.5, 1, 1), (1.5, 0.1, 2, 2)])["reason"])
        wiggle = bs.lambda_star([(0.25, 0.95, 1, 1), (0.6, 0.85, 2, 2), (1.0, 0.92, 3, 3), (1.5, 0.3, 4, 4)])
        self.assertAlmostEqual(wiggle["value"], 1.0 + (0.92 - 0.9) / (0.92 - 0.3) * 0.5)
        self.assertFalse(wiggle["monotonic"])

    def test_paired_log_ratio_with_drift_correction(self):
        x, a, a2 = {1: 1.2, 2: 1.25, 3: 1.18}, {1: 1.0, 2: 1.02, 3: 0.98}, {1: 1.02, 2: 1.0, 3: 1.0}
        stats = bs.paired_log_ratio(x, a, a2, PROTOCOL["statistics"]["t_975"])
        logs = [math.log(x[k] / math.sqrt(a[k] * a2[k])) for k in (1, 2, 3)]
        center = sum(logs) / 3
        sd = math.sqrt(sum((v - center) ** 2 for v in logs) / 2)
        self.assertAlmostEqual(stats["mean_log"], center)
        self.assertAlmostEqual(stats["ci"][0], math.exp(center - 4.303 * sd / math.sqrt(3)))
        self.assertEqual(stats["drift_corrected_rounds"], [1, 2, 3])
        self.assertEqual(bs.paired_log_ratio({1: 2.0}, {1: 1.0}, None, PROTOCOL["statistics"]["t_975"])["ci"], None)

    def test_delta_min(self):
        self.assertAlmostEqual(bs.delta_min_log({1: 1.0, 2: 1.0}, {1: 1.01, 2: 0.99}), math.log(1.05))
        self.assertAlmostEqual(bs.delta_min_log({1: 1.0}, {1: 1.04}), 2 * math.log(1.04))
        self.assertIsNone(bs.delta_min_log({1: 1.0}, {}))

    def test_decision_labels(self):
        table = PROTOCOL["statistics"]["t_975"]
        stats = lambda x: bs.paired_log_ratio(x, {1: 1.0, 2: 1.0, 3: 1.0}, None, table)
        floor = math.log(1.05)
        self.assertEqual(bs.decide(stats({1: 1.30, 2: 1.32, 3: 1.29}), True, floor), "better")
        self.assertEqual(bs.decide(stats({1: 1.30, 2: 1.32, 3: 1.29}), False, floor), "worse")
        self.assertEqual(bs.decide(stats({1: 1.01, 2: 0.995, 3: 1.005}), True, floor), "equivalent")
        self.assertEqual(bs.decide(stats({1: 1.30, 2: 0.90, 3: 1.25}), True, floor), "inconclusive")
        self.assertEqual(bs.decide(stats({1: 1.06, 2: 1.065, 3: 1.062}), True, 2 * math.log(1.05)), "inconclusive",
                         "an effect below delta_min is not a difference")
        self.assertEqual(bs.decide(bs.paired_log_ratio({1: 1.5}, {1: 1.0}, None, table), True, floor), "inconclusive")

    def test_cost_per_million_output_tokens(self):
        self.assertAlmostEqual(bs.cost_per_million(12.89, 2, 500.0), 12.89 * 2 / (500 * 3600) * 1e6)
        self.assertAlmostEqual(bs.cost_per_million(40.96, 1, 500.0, 0.5), 2 * bs.cost_per_million(40.96, 1, 500.0))
        self.assertIsNone(bs.cost_per_million(40.96, 1, 0))


class ParityAndTelemetryTests(unittest.TestCase):
    def raw(self, chosen, top1, generated):
        return {"p1": {"ko-01": {"chosen": chosen, "top1": top1}}, "p2": {"ko-01@0": {"generated": generated}}}

    def test_parity_compare_and_acceptance(self):
        reference = self.raw([-1.0, -2.0, None, -0.5], [5, 6, 7, 8], [1, 2, 3, 4])
        candidate = self.raw([-1.01, -2.02, -3.0, -0.5], [5, 6, 9, 8], [1, 2, 9, 4])
        result = bs.parity_compare(candidate, reference)
        self.assertEqual(result["positions"], 3)
        self.assertAlmostEqual(result["mean_abs_logprob_diff"], 0.01)
        self.assertAlmostEqual(result["top1_agreement"], 0.75)
        self.assertEqual(result["greedy_first_difference"], {"ko-01@0": 2})
        self.assertEqual(result["greedy_diverged_before_16"], 1)
        floor = {"mean_abs_logprob_diff": 0.004, "top1_agreement": 0.76, "greedy_diverged_before_16": 0}
        self.assertEqual(bs.parity_acceptance(result, floor)["status"], "pass")
        self.assertEqual(bs.parity_acceptance(dict(result, mean_abs_logprob_diff=0.02), floor)["status"], "fail")
        self.assertEqual(bs.parity_acceptance(result, None)["status"], "unknown")

    def test_smi_throttle_and_ib_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "gpu-telemetry.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", " index", " clocks.current.sm [MHz]"])
                for second, reasons in ((0, "0x0000000000000004"), (1, "0x0000000000000001"), (2, "0x0000000000000020")):
                    writer.writerow([f"2026/11/09 04:00:0{second}.000", "0", "1410 MHz", "1593 MHz", "250.5 W", "60",
                                     "90 %", "40 %", "70000 MiB", reasons])
            rows = bs.read_smi(path)
        self.assertEqual(len(rows), 3)
        start = rows[0][0]
        clean = bs.throttle_summary(rows, start, start + 1, 0xE8, 0x4)
        self.assertEqual((clean["invalid_fraction"], clean["power_cap_fraction"]), (0, 0.5))
        self.assertAlmostEqual(bs.throttle_summary(rows, start, start + 2, 0xE8, 0x4)["invalid_fraction"], 1 / 3, 3)
        samples = [{"t": 10, "mlx5_ib0/1": [0, 0]}, {"t": 20, "mlx5_ib0/1": [250, 500]}, {"t": 30, "mlx5_ib0/1": [500, 500]}]
        self.assertEqual(bs.ib_window(samples, 12, 18), {"ports": 1, "sent_gb": 0.0, "received_gb": 0.0})
        self.assertEqual(bs.ib_window(samples, 10, 30)["ports"], 1)
        self.assertIsNone(bs.ib_window(samples, 5, 40))

    def test_sanitize(self):
        text = bs.sanitize({"line": "ip=10.1.2.3 model='/mnt/azureml/cr/j/abc123/exe/wd/hub-model' on host-77"},
                           ["host-77"])
        self.assertNotIn("10.1.2.3", json.dumps(text))
        self.assertNotIn("abc123", json.dumps(text))
        self.assertNotIn("host-77", json.dumps(text))

    def test_missing_gpu_telemetry_is_flagged_not_silent(self):
        record = {"workload": "W1", "point_key": "p0", "name": "W1-p0", "phase": "slo-A", "layout": "A", "round": 1,
                  "kind": "open", "returncode": 1, "t_start_unix": 100.0, "t_end_unix": 200.0}
        sample = (150.0, 0, 1410.0, 300.0, 60.0, 0)
        point_ = bs.compact_point(record, ".", PROTOCOL, {"node0": [sample], "node1": []}, {})
        self.assertEqual(point_["flags"], ["node1: no GPU telemetry samples during the point (throttle gate not applied)"])
        self.assertTrue(point_["valid"])
        nothing = bs.compact_point(record, ".", PROTOCOL, {}, {})
        self.assertIn("no GPU telemetry file (throttle gate not applied)", nothing["flags"])

    def test_soak_rules(self):
        rows = [[60 + i, 0.5, 0.05, 30.0, 1, 512, 512] for i in range(0, 2640, 2)]
        soak = point(3, "slo-A", "soak/p0", kind="open", rate=1.0, steady={}, requests=rows,
                     counters={"vllm:num_preemptions_total": 0}, series=[[t, 0.3, 0] for t in range(0, 2700, 10)])
        soak["window"] = [60, 2700]
        self.assertTrue(bs.soak_check(soak, PROTOCOL["workloads"]["soak"]["pass"])["passed"])
        drift = [[r[0], 0.5 if r[0] < 1500 else 0.8] + r[2:] for r in rows]
        rising = dict(soak, requests=drift, series=[[t, 0.3 + t / 10000, 0] for t in range(0, 2700, 10)])
        problems = bs.soak_check(rising, PROTOCOL["workloads"]["soak"]["pass"])["problems"]
        self.assertTrue(any("ttft_p90" in p for p in problems))
        self.assertTrue(any("KV usage rose" in p for p in problems))


class AnalyzeTests(unittest.TestCase):
    RATES = [0.25, 0.6, 1.0, 1.5]

    def round_points(self, round_, c2_tpot, a2_tpot=20.6):
        points = []
        for phase, scale, tpot in (("slo-A", 1.0, 20.5), ("slo-C2", 1.25, c2_tpot), ("slo-D", 0.9, 20.2)):
            goodputs = [1.0, 0.96 if scale >= 1 else 0.93, 0.8 if scale < 1.2 else 0.95, 0.3 if scale < 1.2 else 0.5]
            points += open_points(round_, phase, self.RATES, goodputs, [100 * scale, 220 * scale, 330 * scale, 360 * scale])
            points.append(point(round_, phase, "R/isl1024-c1", steady={"tpot_p50": tpot, "ttft_p50": 400.0}))
        points.append(point(round_, "slo-A2", "R/isl1024-c1", steady={"tpot_p50": a2_tpot, "ttft_p50": 405.0}))
        points += [p for p in open_points(round_, "slo-A2", self.RATES, [1.0, 0.95, 0.8, 0.3], [100, 225, 330, 360])
                   if p["point"] in ("p1", "p2")]
        points.append(point(round_, "slo-A", "R/isl32768-c32", steady={"output_tok_s": 61.0}, flags=["capacity-bound"]))
        points.append(point(round_, "slo-C2", "R/isl32768-c32", steady={"output_tok_s": 72.0}))
        return points

    def package(self, round_, points):
        plans = {}
        for p in points:
            suite = "partr" if p["workload"] == "R" else "slo"
            plans.setdefault(p["phase"], {}).setdefault(suite, {"plan": []})["plan"].append(p["name"])
        for phase in ("slo-A", "slo-C2"):
            plans[phase]["realtext"] = {"plan": []}
            if round_ == 3:
                plans[phase]["soak"] = {"plan": []}
        plans["slo-D"]["soak"] = {"plan": []} if round_ == 3 else None
        plans["slo-D"] = {k: v for k, v in plans["slo-D"].items() if v is not None}
        return {"rounds": [round_], "points": points, "plans": plans, "parity": {}}

    def test_rounds_decisions_and_lambda(self):
        packages = [self.package(r, self.round_points(r, c2_tpot=41.0 + r * 0.1)) for r in (1, 2, 3)]
        evidence = bs.analyze(packages, PROTOCOL, resamples=0)
        self.assertEqual(evidence["valid_rounds"], [1, 2, 3])
        c2 = evidence["comparisons"]["slo-C2 vs slo-A"]
        self.assertEqual(c2["lambda_star.W1.standard"]["decision"], "better")
        self.assertEqual(c2["R/isl1024-c1.tpot_p50"]["decision"], "worse")
        self.assertEqual(c2["R/isl1024-c1.tpot_p50"]["drift_corrected_rounds"], [1, 2, 3])
        self.assertEqual(c2["lambda_star.W1.standard"]["drift_corrected_rounds"], [], "A2 has no lambda*")
        self.assertEqual(c2["cost_per_m_output.W1.standard.spot"]["decision"], "better")
        self.assertTrue(evidence["lambda_star"]["round 1 slo-A"]["W1.standard"]["bracketed"])
        self.assertIn("W1", evidence["next_grid"])
        self.assertFalse(evidence["extension"]["recommended"])
        self.assertIn("R/isl1024-c1.tpot_p50", evidence["a2_noise"])

    def test_invalid_point_invalidates_the_round_and_noise_extends(self):
        packages = [self.package(r, self.round_points(r, c2_tpot=41.0, a2_tpot=20.5 * (1.03 if r == 2 else 0.97)))
                    for r in (1, 2, 3)]
        packages[0]["points"][0]["valid"] = False
        evidence = bs.analyze(packages, PROTOCOL, resamples=0)
        self.assertFalse(evidence["round_validity"][1]["valid"])
        self.assertEqual(evidence["valid_rounds"], [2, 3])
        self.assertEqual(len(evidence["invalid_attempts"]), 1)
        three = [self.package(r, self.round_points(r, 41.0, a2_tpot=20.5 * (1.03 if r == 2 else 0.97))) for r in (1, 2, 3)]
        extension = bs.analyze(three, PROTOCOL, resamples=0)["extension"]
        self.assertTrue(extension["recommended"])
        self.assertTrue(any("ln(A2/A)" in reason for reason in extension["reasons"]))

    def test_kv_bound_points_are_excluded_and_dx2_scaling(self):
        packages = [self.package(r, self.round_points(r, 41.0)) for r in (1, 2, 3)]
        for pkg in packages:
            round_ = pkg["rounds"][0]
            for phase, tok_s, flags in (("slo-A", 404.0, ["kv-bound"] if round_ == 2 else []), ("slo-C2", 454.0, [])):
                extra = point(round_, phase, "R/isl1024-c32", steady={"output_tok_s": tok_s}, flags=flags)
                pkg["points"].append(extra)
                pkg["plans"][phase]["partr"]["plan"].append(extra["name"])
        evidence = bs.analyze(packages, PROTOCOL, resamples=0)
        self.assertEqual(evidence["valid_rounds"], [1, 2, 3])
        self.assertEqual(evidence["comparisons"]["slo-C2 vs slo-A"]["R/isl1024-c32.output_tok_s"]["decision"],
                         "excluded (kv-bound)")
        dx = open_points(1, "dx1-D", [0.25, 0.6, 1.0, 1.5], [1.0, 0.95, 0.5, 0.2], [1, 2, 3, 4])
        dx += open_points(1, "dx2-D", [0.5, 1.2, 2.0, 3.0], [1.0, 0.95, 0.5, 0.2], [1, 2, 3, 4])
        scaling = bs.analyze([{"rounds": [1], "points": dx, "plans": {}, "parity": {}}], PROTOCOL, resamples=0)["dx2_scaling"]
        self.assertAlmostEqual(scaling["round 1 W1"]["ratio"], 2.0)
        self.assertTrue(scaling["round 1 W1"]["passed"])
        self.assertIn("slo-C2 vs slo-A", bs.markdown(evidence))

    def test_command_line_reads_fetch_results_output(self):
        packages = [self.package(r, self.round_points(r, 41.0)) for r in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, pkg in enumerate(packages):
                path = Path(folder, f"round{index + 1}.json")
                path.write_text(json.dumps({"job": "collect", "status": "FINISHED", "reports": {"slo.package": pkg}}))
                paths.append(str(path))
            out, md = Path(folder, "evidence.json"), Path(folder, "summary.md")
            self.assertEqual(bs.main(["analyze", *paths, "--out", str(out), "--md", str(md), "--resamples", "0"]), 0)
            evidence = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(evidence["valid_rounds"], [1, 2, 3])
            self.assertIn("lambda*", md.read_text(encoding="utf-8"))


class PackageTests(unittest.TestCase):
    def test_package_from_job_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            verify = root / "node0" / "verify"
            verify.mkdir(parents=True)
            (root / "node1").mkdir()
            (root / "node0" / "env.json").write_text(json.dumps({"hostname": "gpu-node-17"}))
            (root / "node1" / "env.json").write_text(json.dumps({"hostname": "gpu-node-18"}))
            bench = detailed([0, 70, 100], [0.4, 0.5, 0.6], [[0.02] * 9] * 3, [10, 10, 10])
            (verify / "bench-slo-A-W1-p1-r0.6.json").write_text(json.dumps(bench))
            record = {"record_type": "bench_point", "name": "W1-p1-r0.6", "phase": "slo-A", "layout": "A", "round": 1,
                      "workload": "W1", "point_key": "p1", "point_index": 1, "kind": "open", "request_rate": 0.6,
                      "window": [60, 360], "slo_workload": "W1", "attempt": 1, "returncode": 0, "seed": 1030100,
                      "result_json": "/mnt/azureml/cr/j/abc/exe/wd/outputs/node0/verify/bench-slo-A-W1-p1-r0.6.json",
                      "gates": {"valid": True, "failed": [], "flags": []}, "flags": [], "t_start_unix": 0,
                      "t_end_unix": 1, "deltas": {"totals": {"vllm:num_preemptions_total": 0}},
                      "metrics_jsonl": "metrics-slo-A-W1-p1-r0.6.jsonl"}
            (verify / "slo-A.slo-W1-p1-r0.6.json").write_text(json.dumps(record))
            (verify / "slo-A.slo.json").write_text(json.dumps({"plan": ["W1-p1-r0.6"]}))
            raw = {"record_type": "parity", "p1": {"d": {"chosen": [-1.0, -2.0], "top1": [1, 2]}},
                   "p2": {"d@0": {"generated": [1, 2, 3]}}}
            (verify / "parity-slo-A.json").write_text(json.dumps({**raw, "phase": "slo-A"}))
            (verify / "parity-slo-C2.json").write_text(json.dumps({**raw, "phase": "slo-C2"}))
            (root / "node0" / "effective-slo-A.json").write_text(json.dumps({"lines": ["(RayWorkerProc ip=10.0.0.5) gpu-node-18"]}))
            (root / "node0" / "phases.jsonl").write_text(json.dumps({"phase": "slo-A", "status": "passed"}) + "\n")
            data = bs.package(root, PROTOCOL)
        text = json.dumps(data)
        for secret in ("10.0.0.5", "gpu-node-17", "gpu-node-18", "/mnt/azureml/cr/j/abc"):
            self.assertNotIn(secret, text)
        self.assertEqual(data["rounds"], [1])
        self.assertEqual(data["plans"]["slo-A"]["slo"]["plan"], ["W1-p1-r0.6"])
        only = data["points"][0]
        self.assertEqual((only["id"], only["valid"]), ("W1/p1", True))
        self.assertEqual(only["steady"]["requests"], 2, "requests that started inside [60, 360) s")
        self.assertEqual(len(only["requests"]), 3)
        self.assertEqual(data["parity"]["slo-C2"]["mean_abs_logprob_diff"], 0.0)
        self.assertEqual(sorted(data["hosts_sha256"]), ["node0", "node1"])

    def test_screening_package_uses_the_jobs_first_launch_and_keeps_the_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            verify = root / "node0" / "verify"
            verify.mkdir(parents=True)
            raw = lambda phase, chosen: {"record_type": "parity", "phase": phase, "long": False, "screen": True,
                                         "p1": {"d": {"chosen": chosen, "top1": [1, 2]}}, "p2": {"d@0": {"generated": [1, 2, 3]}}}
            for phase, chosen in (("scr-A", [-1.0, -2.0]), ("scr-C4", [-1.1, -2.0]), ("scr-D", [-1.0, -2.0])):
                (verify / f"parity-{phase}.json").write_text(json.dumps(raw(phase, chosen)))
            (verify / "scr-C4.screen.json").write_text(json.dumps({
                "plan": ["R-isl1024-c1"], "facts": {"max_model_len": 131072}, "steps": {"parity": {"ok": True}}, "summary": []}))
            (root / "node0" / "start-scr-C4.json").write_text(json.dumps({"length_fallback": None, "seconds_to_healthy": 200}))
            (root / "node0" / "protocol-plan.json").write_text(json.dumps({"round": 0, "plan": ["scr-A", "scr-C4"], "layouts": []}))
            (root / "node0" / "reclaim-scr-C4.json").write_text(json.dumps({"summary": "killed 3 processes, removed 1 Ray objects"}))
            (root / "node1").mkdir()
            (root / "node1" / "tail-vllm-scr-PD-replica1.log").write_text("x" * 20000 + "\nNIXL handshake with 10.1.2.3 failed\n")
            data = bs.package(root, PROTOCOL)
        self.assertEqual(data["screen_reference"], "scr-A", "scr-A comes first in the reference list")
        self.assertEqual(sorted(data["parity"]), ["scr-C4", "scr-D"])
        self.assertAlmostEqual(data["parity"]["scr-C4"]["mean_abs_logprob_diff"], 0.05)
        self.assertEqual(data["plans"]["scr-C4"]["screen"]["plan"], ["R-isl1024-c1"])
        self.assertEqual(data["screens"]["scr-C4"]["facts"], {"max_model_len": 131072})
        self.assertEqual(data["start"]["scr-C4"]["seconds_to_healthy"], 200)
        self.assertEqual(data["protocol_plan"]["plan"], ["scr-A", "scr-C4"])
        self.assertEqual(data["reclaim"]["scr-C4"]["summary"], "killed 3 processes, removed 1 Ray objects")
        tail = data["replica_tails"]["scr-PD"]
        self.assertLessEqual(len(tail), bs.TAIL_CHARS)
        self.assertIn("NIXL handshake with <ip> failed", tail)


class ReportingTests(unittest.TestCase):
    def test_screening_packages_round_trip_through_report_chunks_and_fetch_results(self):
        tags = {}
        mlflow = types.ModuleType("mlflow")
        mlflow.set_tags = tags.update
        tracking = types.ModuleType("mlflow.tracking")
        tracking.MlflowClient = object
        with mock.patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.tracking": tracking}):
            report = load_module("axk2_report_under_test", ROOT / "aml" / "src" / "report.py")
            fetch = load_module("axk2_fetch_under_test", ROOT / "aml" / "fetch_results.py")
            for job in PROTOCOL["screen"]["jobs"]:
                with self.subTest(job=job):
                    tags.clear()
                    data = bs.sanitize(synthetic_screening_package(job))
                    for key, payload in bs.split_package(data).items():
                        with redirect_stdout(io.StringIO()):
                            report.send_json(key, bs.encode_report(payload))
                    counts = {k[:-len(".chunks")]: int(v) for k, v in tags.items() if k.endswith(".chunks")}
                    self.assertLessEqual(max(counts.values()), bs.MAX_CHUNKS)
                    self.assertTrue(all(len(v) <= report.CHUNK for v in tags.values()))
                    run = types.SimpleNamespace(data=types.SimpleNamespace(tags=dict(tags)), info=types.SimpleNamespace(status="FINISHED"))
                    status, reports = fetch.tags_json(mock.Mock(get_run=mock.Mock(return_value=run)), "job")
                    merged = bs.merge_package(reports)
                    order = lambda points: sorted(points, key=lambda p: (p["phase"], p["name"], p["attempt"]))
                    self.assertEqual(order(merged["points"]), order(data["points"]))
                    self.assertEqual(merged["missing_parts"], [])
                    self.assertEqual(merged["missing_launches"], [])
                    self.assertEqual({k: v for k, v in merged.items() if k not in ("points", "missing_parts", "missing_launches")},
                                     {k: v for k, v in data.items() if k != "points"})
                    self.assertEqual(status, "FINISHED")

    def test_parts_are_halved_under_the_bound_and_missing_parts_are_reported(self):
        data = synthetic_screening_package("nvfp4_2node")
        parts = bs.split_package(data, max_chars=40_000)
        self.assertTrue(any(name.endswith("-p2") for name in parts), sorted(parts))
        self.assertTrue(all(len(bs.encode_report(v)) <= 40_000 or len(v.get("points", [])) == 1
                            for k, v in parts.items() if k != bs.REPORT_KEY))
        merged = bs.merge_package(parts)
        self.assertEqual(len(merged["points"]), len(data["points"]))
        gone = dict(parts)
        lost = sorted(k for k in gone if k != bs.REPORT_KEY)[0]
        del gone[lost]
        merged = bs.merge_package(gone)
        self.assertEqual(merged["missing_parts"], [lost[len(bs.REPORT_KEY) + 1:]])
        self.assertEqual(merged["missing_launches"], [parts[lost]["launch"]])
        self.assertEqual(len(merged["points"]), len(data["points"]) - len(parts[lost]["points"]))
        whole = {"schema": "axk2-slo-package/1", "points": [], "rounds": [1]}
        self.assertEqual(bs.merge_package({bs.REPORT_KEY: whole}), whole, "an unsplit package still reads")
        self.assertIsNone(bs.merge_package({"node0.env": {}}))
        rng = random.Random(3)
        heavy = dict(data, effective={p: dict(v, lines={"x": [f"{rng.getrandbits(1200):0300x}" for _ in range(200)]})
                                      for p, v in data["effective"].items()})
        index = bs.split_package(heavy, max_chars=60_000)[bs.REPORT_KEY]
        self.assertEqual(index["trimmed"], ["effective.*.lines"])
        self.assertNotIn("lines", next(iter(index["effective"].values())))
        self.assertLess(len(bs.encode_report(index)), 60_000)

    def test_report_package_sends_changed_keys_with_the_mlflow_packages(self):
        sent = []

        def fake_run(command, env=None, check=False, timeout=None, capture_output=False, text=False):
            sent.append((command[2], env.get("PYTHONPATH"), Path(command[4]).read_text()[:2], command[1]))
            return mock.Mock(returncode=0, stdout=f"AXK2_REPORT {command[2]} z:...", stderr="")

        data = synthetic_screening_package("nvfp4")
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(bs.subprocess, "run", fake_run), \
                mock.patch.dict(os.environ, {"AXK2_REPORT_PKGS": "/pkgs", "PYTHONPATH": "/elsewhere"}):
            state = str(Path(folder, "view", "reported.json"))
            first = bs.report_package(data, state)
            count = len(sent)
            order = [key for key, *_ in sent]
            second = bs.report_package(data, state)
            data["points"].append(dict(data["points"][0], phase="scr-D42x", name="extra"))
            bs.report_package(data, state)
            later = [k for k, *_ in sent[count:]]
            everything = bs.report_package(data)
        self.assertEqual(sorted(first), sorted(bs.split_package(synthetic_screening_package("nvfp4"))))
        self.assertTrue(all(v["sent"] for v in first.values()))
        self.assertFalse(any(v["sent"] for v in second.values()), "nothing changed, nothing sent")
        self.assertEqual(sorted(later), ["slo.package", "slo.package.scr-D42x"])
        self.assertTrue(all(v["sent"] for v in everything.values()), "without a state file every key goes out (exit)")
        self.assertEqual({(env, head) for _, env, head, _ in sent}, {("/pkgs", "z:")})
        self.assertTrue(all(script.endswith("report.py") for *_, script in sent))
        self.assertEqual(order[-1], "slo.package", "the index goes out after every part it names")

    def test_report_output_decides_what_was_sent(self):
        data = synthetic_screening_package("nvfp4")
        outputs = {"AXK2_REPORT_WARN mlflow unavailable: No module named 'mlflow'": False,
                   "AXK2_REPORT_WARN set_tags attempt 1 failed for k: busy\nAXK2_REPORT_WARN set_tags attempt 4 failed for k: busy": False,
                   "AXK2_REPORT_WARN set_tags attempt 1 failed for k: busy\nAXK2_REPORT k z:": True,
                   "AXK2_REPORT k z:": True}
        for output, expected in outputs.items():
            with self.subTest(output=output[:50]), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(bs.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=output, stderr="")), \
                    redirect_stdout(io.StringIO()):
                state = Path(folder, "reported.json")
                summary = bs.report_package(data, str(state))
                self.assertTrue(all(v["sent"] == expected for v in summary.values()))
                recorded = json.loads(state.read_text()) if state.exists() else {}
                self.assertEqual(bool(recorded), expected, "a key that did not go out is sent again next time")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), mock.patch.object(
                bs.subprocess, "run", side_effect=bs.subprocess.TimeoutExpired("report.py", 300)):
            self.assertFalse(any(v["sent"] for v in bs.report_package(data, str(Path(folder, "s.json"))).values()))

    def test_package_keeps_node1_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "node0" / "verify").mkdir(parents=True)
            (Path(folder) / "node0" / "peer-scr-C4m.json").write_text(json.dumps({"kv_tokens": [216000, 216000], "nodes": {}}))
            (Path(folder) / "node0" / "preflight-scr-PD.json").write_text(json.dumps({"choice": None, "reason": "x"}))
            (Path(folder) / "node0" / "descs-scr-PD.json").write_text(json.dumps({"summary": "1 'Registering descs' lines"}))
            data = bs.package(folder, PROTOCOL)
        self.assertEqual(data["peer"], {"scr-C4m": {"kv_tokens": [216000, 216000], "nodes": {}}})
        self.assertEqual((data["preflight"]["scr-PD"]["reason"], data["descs"]["scr-PD"]["summary"]),
                         ("x", "1 'Registering descs' lines"))

    def test_package_command_prints_the_key_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "node0" / "verify").mkdir(parents=True)
            with redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(bs.main(["package", folder]), 0)
        self.assertEqual(json.loads(shown.getvalue())["keys"], {"slo.package": {"chunks": 1}})


def screen_points(phase, tpot_1k, tpot_8k, c32_8k, flags=(), valid=True):
    return [point(0, phase, "R/isl1024-c1", steady={"tpot_p50": tpot_1k, "ttft_p50": 300.0}),
            point(0, phase, "R/isl8192-c1", steady={"tpot_p50": tpot_8k, "ttft_p50": 900.0}),
            point(0, phase, "R/isl8192-c32", steady={"output_tok_s": c32_8k}, flags=flags, valid=valid)]


def screening_job(launches, parity=None, start=None, failed=()):
    points = [p for phase, values in launches.items() for p in screen_points(phase, *values)]
    return {"rounds": [0], "points": points, "plans": {}, "parity": parity or {}, "start": start or {},
            "phases": [{"phase": p, "status": "failed" if p in failed else "passed", "note": "x" if p in failed else ""}
                       for p in launches]}


class SelectionTests(unittest.TestCase):
    GOOD = {"mean_abs_logprob_diff": 0.01, "top1_agreement": 0.99, "greedy_diverged_before_16": 0}

    KV_BOUND = {"flags": ["kv-bound"], "peaks": {"kv_cache_usage": 0.99, "running": 26}}

    def kv_bound(self, package, phases):
        """Marks R/isl8192-c32 of these launches KV-bound (32 x 9,216 tokens exceed their KV cache)."""
        package["points"] = [dict(p, **self.KV_BOUND) if p["phase"] in phases and p["id"] == "R/isl8192-c32" else p
                             for p in package["points"]]

    def packages(self):
        fp8 = screening_job({"scr-A": (30, 32, 400), "scr-C2": (40, 42, 450), "scr-C4": (44, 46, 520),
                             "scr-C8": (50, 47, 600), "scr-C16": (35, 35, 700), "scr-B0": (31, 33, 900),
                             "scr-A4": (31, 33, 500)},
                            parity={p: self.GOOD for p in ("scr-C2", "scr-C4", "scr-C8", "scr-C16", "scr-A4")}
                            | {"scr-B0": dict(self.GOOD, mean_abs_logprob_diff=0.2)},
                            start={"scr-C16": {"length_fallback": {"requested": 131072, "estimated": 70000}}})
        self.kv_bound(fp8, {"scr-A4"})
        nvfp4 = screening_job({"scr-D": (25, 27, 340), "scr-E": (28, 29, 330), "scr-Err": (27, 28, 320),
                               "scr-D24": (31, 33, 400), "scr-D8": (20, 20, 500), "scr-D42": (25, 27, 900)},
                              parity={p: self.GOOD for p in ("scr-E", "scr-Err", "scr-D24", "scr-D8", "scr-D42")},
                              failed=("scr-D8",))
        nvfp4["points"] = [dict(p, valid=False) if p["phase"] == "scr-D42" and p["id"] == "R/isl8192-c32" else p
                           for p in nvfp4["points"]]
        self.kv_bound(nvfp4, {"scr-D", "scr-E", "scr-Err"})
        two = {"rounds": [0], "plans": {}, "parity": {"scr-Dx2": self.GOOD}, "start": {},
               "phases": [{"phase": "scr-Dx1", "status": "passed"}, {"phase": "scr-Dx2", "status": "passed"},
                          {"phase": "scr-PD", "status": "failed", "note": "experimental phase: the second server never became healthy"}],
               "points": open_points(0, "scr-Dx1", [0.25, 0.6, 1.0, 1.5], [1.0, 0.95, 0.5, 0.2], [1, 2, 3, 4])
               + open_points(0, "scr-Dx2", [0.5, 1.2, 2.0, 3.0], [1.0, 0.95, 0.5, 0.2], [1, 2, 3, 4])}
        return [fp8, nvfp4, two]

    def test_selection_rule(self):
        selection = bs.select(self.packages(), PROTOCOL)
        fp8, nvfp4 = selection["families"]["fp8"], selection["families"]["nvfp4"]
        self.assertEqual(fp8["ranking"], ["scr-C4", "scr-A4", "scr-C2"], "a KV-bound ranking point does not exclude A4")
        self.assertEqual(fp8["choice"], "C4")
        problems = {p: " ".join(c["problems"]) for p, c in fp8["candidates"].items()}
        self.assertIn("R/isl1024-c1.tpot_p50 is 1.67 x", problems["scr-C8"])
        self.assertIn("could not hold --max-model-len 131072", problems["scr-C16"])
        self.assertIn("parity: mean |d logprob| 0.2000 > 0.05", problems["scr-B0"])
        self.assertEqual(problems["scr-A4"], "")
        self.assertEqual(fp8["candidates"]["scr-A4"]["ranking_point"],
                         {"flags": ["kv-bound"], "peak_kv_cache_usage": 0.99, "peak_running": 26})
        self.assertEqual(problems["scr-C1"], "not screened")
        self.assertEqual(nvfp4["ranking"], ["scr-D", "scr-E", "scr-Err"])
        self.assertEqual(nvfp4["choice"], "D", "D is KV-bound at the ranking point and still selected")
        self.assertEqual(nvfp4["candidates"]["scr-D"]["ranking_point"]["flags"], ["kv-bound"])
        nproblems = {p: " ".join(c["problems"]) for p, c in nvfp4["candidates"].items()}
        self.assertIn("R/isl1024-c1.tpot_p50 is 1.24 x the reference's (limit 1.2)", nproblems["scr-D24"])
        self.assertIn("phase failed", nproblems["scr-D8"])
        self.assertIn("R/isl8192-c32.output_tok_s missing", nproblems["scr-D42"])
        self.assertEqual((selection["round_layouts"], selection["sensitivity"]), ("C4,D", "C4s"))
        self.assertEqual(selection["submit"], "--set environment_variables.ROUND_LAYOUTS=C4,D")
        self.assertEqual(selection["rates"]["slo_rates_env"], ["SLO_RATES_A", "SLO_RATES_C4", "SLO_RATES_D"])
        self.assertEqual(selection["rates"]["soak_rates"], "SOAK_RATES=A=<req/s>,C4=<req/s>,D=<req/s>")
        self.assertEqual(selection["dx2"]["decision"], "pass")
        self.assertAlmostEqual(selection["dx2"]["ratio"], 2.0)
        self.assertEqual((selection["pd"]["scr-PD"]["status"], selection["pd"]["scr-PD"]["selected"]), ("failed", False))
        pairs = {p["name"]: p["ratios"] for p in selection["pairs"]}
        self.assertAlmostEqual(pairs["D x 2 against D"]["lambda_star.W1.standard"], 2.0)
        markdown = bs.selection_markdown(selection)
        self.assertIn("C4,D", markdown)
        self.assertIn("scr-D: eligible, 340 [kv-bound; peak KV 0.990, peak running 26]", markdown)

    def test_both_fp8_jobs_report_every_launch_against_their_own_reference(self):
        fp8, nvfp4, two = self.packages()
        fp8["phases"] = [dict(p, status="failed", note="the server never became healthy") if p["phase"] == "scr-C4" else p
                         for p in fp8["phases"]]
        dp = screening_job({"scr-A": (40, 42, 380), "scr-C4m": (52, 55, 650), "scr-C8m": (58, 60, 680), "scr-C16m": (45, 45, 720)},
                           parity={p: self.GOOD for p in ("scr-C4m", "scr-C8m", "scr-C16m")},
                           start={"scr-C16m": {"max_model_len": 60000, "length_fallback": {"requested": 131072, "estimated": 60000}}})
        fp8["source"], dp["source"] = "screen-fp8", "screen-fp8-dp"
        selection = bs.select([fp8, dp, nvfp4, two], PROTOCOL)
        launches, family = selection["launches"], selection["families"]["fp8"]
        self.assertEqual((launches["scr-C4"]["status"], launches["scr-C4m"]["status"]), ("failed", "passed"), "both reported")
        self.assertEqual((launches["scr-A"]["source"], launches["scr-A@screen-fp8-dp"]["source"]), ("screen-fp8", "screen-fp8-dp"))
        c4m = family["candidates"]["scr-C4m"]
        self.assertEqual(c4m["reference"], "scr-A@screen-fp8-dp", "the TPOT reference is the A of the same job")
        self.assertTrue(c4m["eligible"], "52/40 ms: 1.3 x its own A (it would be 1.73 x the other job's)")
        self.assertIn("phase failed", " ".join(family["candidates"]["scr-C4"]["problems"]))
        self.assertIn("could not hold --max-model-len 131072", " ".join(family["candidates"]["scr-C16m"]["problems"]))
        self.assertEqual(family["ranking"], ["scr-C8m", "scr-C4m", "scr-A4", "scr-C2"])
        self.assertEqual((family["choice"], selection["round_layouts"], selection["sensitivity"]), ("C8m", "C8m,D", "C8ms"))
        self.assertEqual(selection["rates"]["slo_rates_env"], ["SLO_RATES_A", "SLO_RATES_C8m", "SLO_RATES_D"])
        markdown = bs.selection_markdown(selection)
        for row in ("| scr-C4 | failed |", "| scr-C4m | passed |", "| scr-A@screen-fp8-dp | passed |"):
            self.assertIn(row, markdown)
        self.assertEqual(sorted(k for k in family["candidates"] if k.startswith("scr-C4")), ["scr-C4", "scr-C4m"])

    def test_max_model_len_is_the_served_value(self):
        nvfp4 = screening_job({"scr-D": (25, 27, 340), "scr-D8": (20, 20, 500)},
                              start={"scr-D": {"max_model_len": 131072},
                                     "scr-D8": {"max_model_len": 31616, "length_fallback": {"requested": 131072, "estimated": 31616}}})
        nvfp4["effective"] = {"scr-D": {"max_model_len": 131072}, "scr-D8": {"max_model_len": 262144, "kv_tokens": [32896] * 8}}
        fp8 = screening_job({"scr-A": (30, 32, 400), "scr-A-auto": (31, 33, 410)}, start={"scr-A-auto": {"max_model_len": 262144}})
        fp8["effective"] = {"scr-A-auto": {"max_model_len": 262144}, "scr-A": {"max_model_len": 131072}}
        launches = bs.screen_launches([fp8, nvfp4], PROTOCOL)
        self.assertEqual(launches["scr-D8"]["max_model_len"], 31616, "-1 served 31,616, not the config line's 262,144")
        self.assertEqual(launches["scr-A-auto"]["max_model_len"], 262144, "A-auto does serve 262,144")
        self.assertEqual(launches["scr-A"]["max_model_len"], 131072, "no start record: the engine config value")
        self.assertIn("| scr-D8 | passed | 31616 (fallback) |", bs.selection_markdown(bs.select([fp8, nvfp4], PROTOCOL)))

    def test_node1_engines_count_in_the_kv_totals(self):
        two = screening_job({"scr-Dx1": (25, 27, 340), "scr-Dx2": (25, 27, 680), "scr-PD": (25, 27, 300)})
        two["effective"] = {"scr-Dx1": {"kv_tokens": [253120]}, "scr-Dx2": {"kv_tokens": [253120]}, "scr-PD": {"kv_tokens": [250000]}}
        two["peer"] = {"scr-Dx2": {"kv_tokens": [253120]}}
        two["replica_tails"] = {"scr-PD": "GPU KV cache size: 1 tokens\n=== axk2 replica start scr-PD-replica1 ===\n"
                                          "(EngineCore pid=1) INFO [kv_cache_utils.py:1] GPU KV cache size: 249,984 tokens\n"}
        dp = screening_job({"scr-A": (30, 32, 400), "scr-C8m": (40, 42, 680)})
        dp["effective"] = {"scr-A": {"kv_tokens": [600000, 600000]}, "scr-C8m": {"kv_tokens": [172000] * 4}}
        launches = bs.screen_launches([two, dp], PROTOCOL)
        scopes = {p: (launches[p]["kv_tokens_total"], launches[p]["kv_scope"]) for p in ("scr-Dx1", "scr-Dx2", "scr-PD", "scr-A", "scr-C8m")}
        self.assertEqual(scopes, {"scr-Dx1": (253120, "all engines"), "scr-Dx2": (506240, "node0 and node1"),
                                  "scr-PD": (499984, "node0 and node1"), "scr-A": (1200000, "all engines"),
                                  "scr-C8m": (688000, "node0 only")})
        self.assertEqual(bs.kv_from_text(two["replica_tails"]["scr-PD"]), [249984], "only the latest start counts")
        rows = [r for r in bs.selection_markdown(bs.select([two, dp], PROTOCOL)).splitlines() if r.startswith("| scr-C8m |")]
        self.assertIn("172,000-172,000 (4, 688,000) node0 only |", rows[0])

    def test_the_pd_rerun_is_reported_with_cross_job_comparisons(self):
        fp8, nvfp4, two = self.packages()
        two["source"] = "screen-nvfp4-2node"
        rerun = {"rounds": [0], "plans": {}, "parity": {"scr-PD": self.GOOD}, "start": {}, "source": "screen-nvfp4-pd",
                 "preflight": {"scr-PD": {"choice": "UCX_MEMTYPE_CACHE=n", "node0": {}, "node1": {}}},
                 "phases": [{"phase": "scr-Dref", "status": "passed"}, {"phase": "scr-PD", "status": "passed"}],
                 "points": open_points(0, "scr-PD", [0.5, 1.2, 2.0, 3.0], [1.0, 0.95, 0.5, 0.2], [1, 2, 3, 4])}
        selection = bs.select([fp8, nvfp4, two, rerun], PROTOCOL)
        self.assertEqual({k: (v["status"], v["source"]) for k, v in selection["pd"].items()},
                         {"scr-PD": ("failed", "screen-nvfp4-2node"), "scr-PD@screen-nvfp4-pd": ("passed", "screen-nvfp4-pd")})
        self.assertEqual(selection["pd"]["scr-PD@screen-nvfp4-pd"]["preflight"], {"choice": "UCX_MEMTYPE_CACHE=n"})
        pairs = {(p["name"], p["x"]): p for p in selection["pairs"] if p["name"].startswith("PD against")}
        rerun_vs_d = pairs[("PD against D", "scr-PD@screen-nvfp4-pd")]
        self.assertEqual((rerun_vs_d["base"], rerun_vs_d["cross_job"]), ("scr-Dx1", True))
        self.assertAlmostEqual(rerun_vs_d["ratios"]["lambda_star.W1.standard"], 2.0, msg="the rerun's own ladder")
        self.assertEqual(pairs[("PD against D x 2", "scr-PD@screen-nvfp4-pd")]["base"], "scr-Dx2")
        self.assertFalse(pairs[("PD against D", "scr-PD")]["cross_job"], "the first PD's own job had Dx1")
        markdown = bs.selection_markdown(selection)
        self.assertIn("PD against D (scr-PD@screen-nvfp4-pd / scr-Dx1, cross-job)", markdown)
        self.assertIn("PD (experimental, never selected) scr-PD@screen-nvfp4-pd: passed; NIXL preflight: UCX_MEMTYPE_CACHE=n", markdown)

    def test_a_failed_reference_selects_nothing_in_its_family(self):
        packages = self.packages()
        packages[1]["phases"][0]["status"] = "failed"
        selection = bs.select(packages, PROTOCOL)
        self.assertIsNone(selection["families"]["nvfp4"]["choice"])
        self.assertEqual(selection["round_layouts"], "C4")
        self.assertEqual(selection["rates"]["soak_rates_keys"], ["A", "C4"])
        self.assertTrue(all("the reference scr-D did not pass" in c["problems"][0]
                            for c in selection["families"]["nvfp4"]["candidates"].values()))

    def test_dx2_decision_with_bounds(self):
        bracketed = lambda v: {"value": v}
        self.assertEqual(bs.dx2_decision({"value": None, "low": 3.0}, bracketed(1.2), 1.9), "pass")
        self.assertEqual(bs.dx2_decision(bracketed(2.0), {"value": None, "low": 1.5}, 1.9), "fail")
        self.assertEqual(bs.dx2_decision({"value": None, "low": 1.5}, {"value": None, "low": 0.6}, 1.9), "undecided")
        self.assertEqual(bs.dx2_decision({"value": None, "high": 0.5}, bracketed(0.6), 1.9), "fail")
        self.assertEqual(bs.dx2_decision(bracketed(1.1), bracketed(0.6), 1.9), "fail")
        self.assertEqual(bs.dx2_decision(None, bracketed(0.6), 1.9), "undecided")
        self.assertEqual(bs.lambda_star([(1, 0.95, 1, 1), (2, 0.93, 2, 2)])["low"], 2)
        self.assertEqual(bs.lambda_star([(1, 0.5, 1, 1), (2, 0.3, 2, 2)])["high"], 1)

    def test_select_command_reads_split_fetch_results_output(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, pkg in enumerate(self.packages()):
                path = Path(folder, f"screen{index}.json")
                path.write_text(json.dumps({"job": f"j{index}", "status": "FINISHED", "reports": bs.split_package(pkg)}))
                paths.append(str(path))
            out, md = Path(folder, "selection.json"), Path(folder, "selection.md")
            with redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(bs.main(["select", *paths, "--out", str(out), "--md", str(md)]), 0)
            printed = json.loads(shown.getvalue())
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["round_layouts"], "C4,D")
            self.assertIn("| scr-C4 | passed |", md.read_text(encoding="utf-8"))
        self.assertEqual(printed, {"round_layouts": "C4,D", "sensitivity": "C4s", "dx2": "pass",
                                   "submit": "--set environment_variables.ROUND_LAYOUTS=C4,D",
                                   "slo_rates_env": ["SLO_RATES_A", "SLO_RATES_C4", "SLO_RATES_D"],
                                   "soak_rates": "SOAK_RATES=A=<req/s>,C4=<req/s>,D=<req/s>"})


class SelectedRoundAnalysisTests(unittest.TestCase):
    def test_analyze_follows_the_recorded_round_layouts(self):
        maker = AnalyzeTests()
        packages = []
        for round_ in (1, 2, 3):
            points = [dict(p, phase=p["phase"].replace("slo-C2", "slo-C4"), layout=p["layout"].replace("C2", "C4"))
                      for p in maker.round_points(round_, c2_tpot=41.0)]
            plans = {}
            for p in points:
                plans.setdefault(p["phase"], {}).setdefault("partr" if p["workload"] == "R" else "slo", {"plan": []})["plan"].append(p["name"])
            for phase in ("slo-A", "slo-C4", "slo-D"):
                if phase != "slo-D":
                    plans[phase]["realtext"] = {"plan": []}
                if round_ == 3:
                    plans[phase]["soak"] = {"plan": []}
            packages.append({"rounds": [round_], "points": points, "plans": plans, "parity": {},
                             "protocol_plan": {"round": round_, "layouts": ["C4", "D"], "plan": []}})
        evidence = bs.analyze(packages, PROTOCOL, resamples=0)
        self.assertEqual(evidence["valid_rounds"], [1, 2, 3])
        self.assertEqual(evidence["round_layouts"], {"1": ["C4", "D"], "2": ["C4", "D"], "3": ["C4", "D"]})
        self.assertIn("slo-C4 vs slo-A", evidence["comparisons"])
        self.assertIn("slo-C4s vs slo-C4", evidence["comparisons"])
        self.assertNotIn("slo-C2 vs slo-A", evidence["comparisons"])
        self.assertEqual(evidence["comparisons"]["slo-C4 vs slo-A"]["lambda_star.W1.standard"]["decision"], "better")
        packages[0]["missing_launches"] = ["cap-C4"]
        self.assertTrue(bs.analyze(packages, PROTOCOL, resamples=0)["round_validity"][1]["valid"],
                        "a lost capacity-phase part does not invalidate the round")
        packages[0]["missing_launches"] = ["slo-C4"]
        validity = bs.analyze(packages, PROTOCOL, resamples=0)["round_validity"]
        self.assertFalse(validity[1]["valid"])
        self.assertIn("slo-C4: a part of the results package never arrived", validity[1]["problems"])
        self.assertTrue(validity[2]["valid"])


if __name__ == "__main__":
    unittest.main()
