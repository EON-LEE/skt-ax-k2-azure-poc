import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "aml" / "src" / "bench"), str(ROOT / "aml" / "src")]
import bench_stats as bs  # noqa: E402
del sys.path[:2]

PROTOCOL = bs.load_protocol()
W1 = PROTOCOL["slo"]["tiers"]["standard"]["W1"]


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


if __name__ == "__main__":
    unittest.main()
