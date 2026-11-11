import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "aml" / "src" / "bench"), str(ROOT / "aml" / "src")]
import client_tests  # noqa: E402
import slo_suites as ss  # noqa: E402
import verify_suites  # noqa: E402
del sys.path[:2]

PROTOCOL = ss.load_protocol()
METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="axk2"} 3.0
vllm:num_requests_running{engine="1",model_name="axk2"} 5.0
vllm:kv_cache_usage_perc{engine="0",model_name="axk2"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="axk2"} 0.5
vllm:prompt_tokens_total{engine="0",model_name="axk2"} 1000.0
vllm:prompt_tokens_total{engine="1",model_name="axk2"} 3000.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="axk2"} 4.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="axk2"} 1.0
vllm:num_preemptions_total{engine="0",model_name="axk2"} 0.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="axk2"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="axk2"} 2.0
python_gc_objects_collected_total{generation="0"} 1.0
"""


def args(**overrides):
    values = dict(base="http://x", tag="slo-A", out_dir="o", tokenizer="/models/fp8", round=1, layout="A",
                  phase="slo-A", rates_key="A", options={}, rates={}, slo="", soak_rates={}, realtext_path="",
                  commit="abc", protocol=str(ss.PROTOCOL))
    values.update(overrides)
    return argparse.Namespace(**values)


class PlanTests(unittest.TestCase):
    def test_seed_scheme(self):
        self.assertEqual(ss.protocol_seed(2, 3, 1), 2030100)
        self.assertEqual(ss.protocol_seed(2, 3, 1, rep=4), 2030104)
        self.assertEqual(ss.warmup_seed(3, 1), 9030100)
        for round_ in (1, 2, 3):
            seeds = [p["seed"] for p in ss.partr_points(PROTOCOL, round_) + ss.slo_points(PROTOCOL, round_, ["S1", "W1", "W2"])]
            self.assertEqual(len(seeds), len(set(seeds)), "a seed per point")
            self.assertTrue(all(1000000 * round_ <= s < 1000000 * (round_ + 1) for s in seeds))

    def test_same_prompts_for_every_layout_and_a_prime_reuses_a(self):
        full = {p["name"]: p["seed"] for p in ss.slo_points(PROTOCOL, 2, ["W1"])}
        a_prime = ss.slo_points(PROTOCOL, 2, ["W1"], point_filter={"W1": {1, 2}})
        self.assertEqual(sorted(p["index"] for p in a_prime), [1, 2])
        self.assertTrue(all(full[p["name"]] == p["seed"] for p in a_prime))
        c32 = {p["name"]: p["seed"] for p in ss.partr_points(PROTOCOL, 1, "c32")}
        self.assertEqual(c32, {p["name"]: p["seed"] for p in ss.partr_points(PROTOCOL, 1) if p["name"] in c32})
        self.assertEqual(ss.rate_order(1, 3, 4), ss.rate_order(1, 3, 4))
        self.assertEqual(sorted(ss.rate_order(1, 3, 4)), [0, 1, 2, 3])

    def test_open_points_size_and_part_r_rounds(self):
        w1 = {p["index"]: p for p in ss.slo_points(PROTOCOL, 1, ["W1"])}
        self.assertEqual(w1[0]["prompts"], 90)
        self.assertEqual(w1[3]["prompts"], 540)
        self.assertEqual(w1[3]["window"], [60, 360])
        self.assertEqual(w1[3]["slo_workload"], "W1")
        self.assertIn("R-isl32768-c32", [p["name"] for p in ss.partr_points(PROTOCOL, 1)])
        self.assertNotIn("R-isl32768-c32", [p["name"] for p in ss.partr_points(PROTOCOL, 2)])
        self.assertEqual([p["name"] for p in ss.partr_points(PROTOCOL, 0, "short")], ["R-isl1024-c1", "R-isl1024-c32"])
        dx2 = ss.slo_points(PROTOCOL, 1, ["W1"], rate_set="dx2")
        self.assertEqual(sorted(p["rate"] for p in dx2), PROTOCOL["rate_sets"]["dx2"]["W1"])

    def test_round_plans(self):
        self.assertEqual(ss.round_plan(PROTOCOL, 1, "fp8"), ["slo-A", "slo-B0", "slo-C8m", "slo-A2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 2, "fp8"),
                         ["slo-A", "slo-C8m", "slo-B0", "slo-C8ms", "slo-A2", "cap-A", "cap-B0", "cap-C8m"])
        self.assertEqual(ss.round_plan(PROTOCOL, 3, "fp8"), ["slo-A", "slo-B0", "slo-C8m", "slo-A2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 1, "nvfp4"), ["slo-D", "slo-D42", "slo-D24", "slo-D2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 2, "nvfp4"),
                         ["slo-D", "slo-D24", "slo-D42", "slo-D2", "cap-D", "cap-D42", "cap-D24"])
        self.assertEqual(ss.round_plan(PROTOCOL, 5, "nvfp4"), ["slo-D", "slo-D42", "slo-D24", "slo-D2"])
        self.assertEqual((PROTOCOL["rounds"]["sensitivity_rounds"], PROTOCOL["rounds"]["capacity_rounds"]), ([2], [2]),
                         "round 1 carries the accuracy run; the sensitivity run and the capacity phases wait for round 2")
        with self.assertRaises(ValueError):
            ss.round_plan(PROTOCOL, 6, "fp8")
        self.assertEqual(ss.expand_plan(PROTOCOL, ["slo-round-fp8", "diag-A"], 2)[-1], "diag-A")
        for bad in (["epc-fp8-tp8pp2"], ["slo-round"], ["slo-round-fp8", "slo-round-nvfp4"]):
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                ss.expand_plan(PROTOCOL, bad, 1)
        jobs = PROTOCOL["rounds"]["jobs"]
        self.assertEqual(ss.round_jobs(PROTOCOL), {"slo-round-fp8": "fp8", "slo-round-nvfp4": "nvfp4"})
        self.assertEqual((jobs["fp8"]["nodes"], jobs["nvfp4"]["nodes"]), (2, 1), "a 2-node and a 1-node job: three nodes")
        self.assertEqual((jobs["fp8"]["control"], jobs["nvfp4"]["control"]), ("A2", "D2"), "both jobs relaunch their reference")
        selection = PROTOCOL["rounds"]["selection"]
        self.assertEqual(sorted(selection["pre_registered"] + selection["added_after_screening"]),
                         sorted(jobs["fp8"]["layouts"] + jobs["nvfp4"]["layouts"]))
        self.assertEqual(selection["added_after_screening"], ["C8m", "D24"])
        self.assertTrue(any("C8m and D24" in d["what"] for d in PROTOCOL["deviations"]))

    def test_seeds_and_rate_order_are_shared_by_both_round_jobs(self):
        for round_ in (1, 2, 3):
            seen = {}
            for job in ("fp8", "nvfp4"):
                for phase in ss.round_plan(PROTOCOL, round_, job):
                    spec = ss.phase_spec(PROTOCOL, phase, round_)
                    if "slo" not in spec["suites"]:
                        continue
                    options = spec["options"].get("slo", {})
                    filters = {k[:-len("_points")]: set(v) for k, v in options.items() if k.endswith("_points")}
                    points = ss.slo_points(PROTOCOL, round_, options.get("workloads", ["S1", "W1", "W2"]),
                                           ss.phase_rates(PROTOCOL, spec["rates_key"]), filters)
                    for p in points:
                        self.assertEqual(seen.setdefault(p["name"], p["seed"]), p["seed"], "common random numbers")
                    order = [p["name"] for p in points if p["workload"] == "W1"]
                    full = [p["name"] for p in ss.slo_points(PROTOCOL, round_, ["W1"], ss.phase_rates(PROTOCOL, "A"))]
                    self.assertEqual(order, [n for n in full if n in order], f"{phase}: the round's rate order")
            self.assertTrue(all(1000000 * round_ <= s < 1000000 * (round_ + 1) for s in seen.values()))

    def test_every_phase_gets_the_common_flags(self):
        server = PROTOCOL["server"]
        for phase in PROTOCOL["phases"]:
            with self.subTest(phase=phase):
                spec = ss.phase_spec(PROTOCOL, phase, 1)
                flags = spec["args"]
                pairs = dict(zip(flags, flags[1:]))
                capacity = PROTOCOL["phases"][phase].get("max_model_len") == "capacity"
                self.assertEqual(pairs["--max-model-len"], "-1" if capacity else "131072")
                self.assertEqual(pairs["--api-server-count"], "1")
                # Only a layout that records its own value (PD, see its diagnosis note) leaves the common 0.9.
                layout = ss.resolve_layout(PROTOCOL, PROTOCOL["phases"][phase]["layout"])
                self.assertEqual(pairs["--gpu-memory-utilization"],
                                 str(layout.get("gpu_memory_utilization", server["gpu_memory_utilization"])))
                self.assertEqual("gpu_memory_utilization" in layout, PROTOCOL["phases"][phase]["layout"] == "PD")
                self.assertEqual(flags.count("--max-model-len"), 1)
                self.assertIn("--async-scheduling", flags)
                self.assertNotIn("--enforce-eager", flags)
                self.assertEqual(flags[:len(server["launcher_flags"])], server["launcher_flags"])
                self.assertEqual(flags[-len(server["common_flags"]):], server["common_flags"])
                self.assertEqual(spec["env"]["VLLM_SERVER_DEV_MODE"], "1")
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D2")["rates_key"], "D")
        c2s, c2 = ss.phase_spec(PROTOCOL, "slo-C2s"), ss.phase_spec(PROTOCOL, "slo-C2")
        self.assertEqual((c2s["max_num_seqs"], c2s["max_num_batched_tokens"]), (64, 4096))
        self.assertEqual((c2["max_num_seqs"], c2["max_num_batched_tokens"]), (128, 8192))
        self.assertEqual(c2s["rates_key"], "C2")
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-A2")["rates_key"], "A")
        self.assertIn("--expert-placement-strategy", ss.phase_spec(PROTOCOL, "diag-C1rr")["args"])
        self.assertIn("--decode-context-parallel-size", ss.phase_spec(PROTOCOL, "diag-B0dcp")["args"])
        self.assertNotIn("soak", ss.phase_spec(PROTOCOL, "slo-A", 2)["suites"])
        self.assertIn("soak", ss.phase_spec(PROTOCOL, "slo-A", 3)["suites"])
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D")["nodes"], 1)
        self.assertEqual(ss.phase_spec(PROTOCOL, "dx2-D")["replicas"], 2)

    @unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs a POSIX bash")
    def test_shell_spec_round_trips_through_bash(self):
        spec = ss.phase_spec(PROTOCOL, "slo-C2", 1)
        script = ss.shell_spec(spec) + '\nprintf "%s\\n" "${SPEC_ARGS[@]}"; printf "%s\\n" "${SPEC_ENV[@]}"; echo "$SPEC_OPTIONS"'
        lines = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(lines[:len(spec["args"])], spec["args"])
        self.assertIn("VLLM_RAY_DP_PACK_STRATEGY=strict", lines)
        self.assertEqual(json.loads(lines[-1]), spec["options"])


class MetricsAndGateTests(unittest.TestCase):
    def test_prometheus_engines_stay_apart(self):
        values = ss.engine_values(ss.parse_prometheus(METRICS))
        self.assertEqual(sorted(values), ["0", "1"])
        self.assertEqual(values["0"]["vllm:num_requests_running"], 3.0)
        self.assertEqual(values["1"]["vllm:num_requests_running"], 5.0)
        self.assertEqual(values["0"]["vllm:request_success_total"], 5.0, "finished_reason label sets add up per engine")
        self.assertEqual(values["0"]["vllm:time_to_first_token_seconds_count"], 5.0)
        self.assertNotIn("vllm:time_to_first_token_seconds_bucket", values["0"])
        tagged = ss.engine_values(ss.parse_prometheus('vllm:num_requests_running{replica="1",engine="0"} 2.0'))
        self.assertEqual(list(tagged), ["1:0"])

    def test_sampler_peaks_per_engine_and_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "m.jsonl")
            sampler = ss.MetricsSampler("http://x", path=path)
            with open(path, "a") as handle:
                sampler.add(ss.engine_values(ss.parse_prometheus(METRICS)), 1.0, handle)
                sampler.add({"0": {"vllm:num_requests_running": 7.0}, "1": {"vllm:num_requests_running": 1.0}}, 2.0, handle)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        summary = sampler.summary()
        self.assertEqual(summary["peak_running_requests"], 7.0, "the largest engine, not a sum")
        self.assertEqual(summary["peak_running_requests_sum_over_engines"], 8.0)
        self.assertEqual(summary["peak_kv_cache_usage"], 0.5)
        self.assertEqual(summary["engines"], 2)
        self.assertEqual([line["t"] for line in lines], [1.0, 2.0])

    def test_counter_deltas(self):
        before = {"0": {"vllm:prompt_tokens_total": 100.0, "vllm:num_requests_running": 1.0}}
        after = {"0": {"vllm:prompt_tokens_total": 1124.0, "vllm:num_requests_running": 4.0},
                 "1": {"vllm:prompt_tokens_total": 1024.0}}
        deltas = ss.counter_deltas(before, after)
        self.assertEqual(deltas["per_engine"]["0"], {"vllm:prompt_tokens_total": 1024.0})
        self.assertEqual(deltas["totals"]["vllm:prompt_tokens_total"], 2048.0)

    def gate(self, kind="closed", **record):
        base = {"idle_before": True, "prefix_cache_reset": True, "returncode": 0, "failed": 0,
                "total_input_tokens": 2048, "concurrency": 32, "peak_kv_cache_usage": 0.3,
                "peak_running_requests_sum_over_engines": 32,
                "deltas": {"totals": {"vllm:prompt_tokens_total": 2048.0, "vllm:prefix_cache_queries_total": 2048.0,
                                      "vllm:prefix_cache_hits_total": 0.0, "vllm:num_preemptions_total": 0.0}}}
        base.update(record)
        return ss.evaluate_gates(base, kind, PROTOCOL["gates"])

    def test_gates(self):
        self.assertEqual(self.gate(), {"valid": True, "failed": [], "flags": []})
        totals = lambda **v: {"totals": {"vllm:prompt_tokens_total": 2048.0, "vllm:prefix_cache_queries_total": 2048.0,
                                         "vllm:prefix_cache_hits_total": 0.0, "vllm:num_preemptions_total": 0.0, **v}}
        self.assertFalse(self.gate(deltas=totals(**{"vllm:prompt_tokens_total": 2100.0}))["valid"])
        self.assertFalse(self.gate(deltas=totals(**{"vllm:prefix_cache_hits_total": 64.0}))["valid"])
        self.assertEqual(self.gate(deltas=totals(**{"vllm:num_preemptions_total": 3.0}))["flags"], ["capacity-bound"])
        self.assertIn("kv-bound", self.gate(peak_kv_cache_usage=0.97)["flags"])
        self.assertIn("kv-bound", self.gate(peak_running_requests_sum_over_engines=20)["flags"])
        self.assertFalse(self.gate(failed=1)["valid"])
        self.assertTrue(self.gate("open", failed=3, concurrency=None)["valid"])
        self.assertFalse(self.gate(prefix_cache_reset=False)["valid"])
        self.assertFalse(self.gate(idle_before=False)["valid"])
        self.assertFalse(self.gate(deltas=None)["valid"])


class BenchCommandTests(unittest.TestCase):
    def test_open_and_closed_loop_commands(self):
        command = ss.bench_command(args(), "f.json", None, 216, 1024, 512, 1030100, request_rate=0.6, burstiness=1.0,
                                   goodput={"ttft": 3000, "tpot": 100}, sampling={"temperature": 0.6, "top_p": 0.95},
                                   range_ratio=0, metadata={"layout": "A", "round": 1})
        text = " ".join(command)
        self.assertIn("--request-rate 0.6 --burstiness 1.0", text)
        self.assertNotIn("--max-concurrency", command)
        self.assertIn("--save-detailed", command)
        self.assertIn("--metric-percentiles 50,90,95,99", text)
        self.assertIn("--goodput ttft:3000 tpot:100", text)
        self.assertIn("--temperature 0.6 --top-p 0.95", text)
        self.assertIn("--random-range-ratio 0", text)
        self.assertEqual(command[-3:], ["--metadata", "layout=A", "round=1"])
        self.assertNotIn("--num-warmups", command)
        closed = ss.bench_command(args(), "f.json", 32, 64, 1024, 1024, 7)
        self.assertIn("--max-concurrency", closed)
        self.assertNotIn("--request-rate", closed)
        custom = ss.bench_command(args(), "f.json", None, 60, 1024, 512, 7, request_rate=1.0, dataset="d.jsonl")
        self.assertIn("--dataset-name custom --dataset-path d.jsonl --custom-output-len 512", " ".join(custom))

    def test_run_point_reruns_an_invalid_point_once_with_the_same_seed(self):
        point = ss.slo_points(PROTOCOL, 1, ["W1"])[0]
        calls, emitted = [], []

        def fake_bench(_, name, concurrency, prompts, input_len, output_len, timeout, seed, **options):
            calls.append((name, seed, options.get("request_rate"), options.get("goodput")))
            return {"name": name, "returncode": 0, "failed": 0, "total_input_tokens": 100, "concurrency": None,
                    "peak_kv_cache_usage": 0.2}

        snapshots = iter([{"0": {"vllm:prompt_tokens_total": 0.0}}, {"0": {"vllm:prompt_tokens_total": 999.0}},
                          {"0": {"vllm:prompt_tokens_total": 0.0}}, {"0": {"vllm:prompt_tokens_total": 100.0}}])
        with mock.patch.object(ss, "bench", fake_bench), mock.patch.object(ss, "wait_idle", return_value=(True, 0.0)), \
                mock.patch.object(ss, "reset_prefix_cache", return_value=True), \
                mock.patch.object(ss, "snapshot", side_effect=lambda *a: next(snapshots)), \
                mock.patch.object(ss.vs, "emit", side_effect=lambda *a: emitted.append(a[2])), \
                mock.patch.object(ss.vs, "server_healthy", return_value=True):
            attempts = ss.run_point(args(), "slo", point, PROTOCOL)
        self.assertEqual([a["valid"] for a in attempts], [False, True])
        self.assertEqual([c[0] for c in calls], [point["name"], point["name"] + "-retry"])
        self.assertEqual({c[1] for c in calls}, {point["seed"]})
        self.assertEqual(calls[0][3], {"ttft": 3000, "tpot": 100})
        self.assertEqual(emitted, [f"slo-{point['name']}", f"slo-{point['name']}-retry"])
        self.assertEqual(attempts[1]["record_type"], "bench_point")


class ParityTests(unittest.TestCase):
    def test_teacher_forced_and_greedy_parsing(self):
        entries = [None, {"5": {"logprob": -0.1, "rank": 1}}, {"9": {"logprob": -0.2, "rank": 1}, "7": {"logprob": -1.5, "rank": 3}}]
        payload = {"choices": [{"prompt_logprobs": entries, "logprobs": {"tokens": ["token_id:11", "token_id:12"]}}]}
        with mock.patch.object(ss.vs, "call", return_value=(payload, 0.5)) as call:
            result = ss.teacher_forced("http://x", "d", [1, 5, 7])
            body = call.call_args[0][2]
            greedy = ss.greedy("http://x", "d@0", [1, 2], 2)
        self.assertEqual((body["prompt_logprobs"], body["max_tokens"], body["temperature"]), (20, 1, 0.0))
        self.assertEqual(result["chosen"], [-0.1, -1.5])
        self.assertEqual(result["rank"], [1, 3])
        self.assertEqual(result["top1"], [5, 9])
        self.assertEqual(greedy["generated"], [11, 12])

    def test_greedy_keeps_the_chosen_logprobs(self):
        payload = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:12"], "token_logprobs": [-0.123456, None]},
                                "finish_reason": "length"}]}
        with mock.patch.object(ss.vs, "call", return_value=(payload, 0.5)):
            result = ss.greedy("http://x", "d@0", [1, 2], 2)
        self.assertEqual((result["generated"], result["logprobs"], result["finish_reason"]), ([11, 12], [-0.12346, None], "length"))

    def test_long_check_is_greedy_without_prompt_logprobs(self):
        bodies = []

        def call(base, path, body, timeout):
            bodies.append(body)
            return {"choices": [{"logprobs": {"tokens": [f"token_id:{i}" for i in range(body["max_tokens"])],
                                              "token_logprobs": [-0.5] * body["max_tokens"]}, "finish_reason": "length"}]}, 1.0
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ss.vs, "call", call), \
                mock.patch.object(ss, "tokenize", lambda base, text, special: list(range(40000))), \
                mock.patch.object(ss, "reset_prefix_cache", lambda base: True):
            result = ss.suite_paritylong(args(out_dir=folder))
            raw = json.loads(Path(folder, "parity-slo-A-long.json").read_text())
        self.assertEqual(len(bodies), 2, "the two long documents")
        for body in bodies:
            self.assertNotIn("prompt_logprobs", body, "no fp32 log-softmax over a whole prefill chunk")
            self.assertEqual((len(body["prompt"]), body["max_tokens"], body["temperature"], body["logprobs"]),
                             (32768, PROTOCOL["parity"]["p3"]["max_tokens"], 0.0, 1))
        self.assertEqual(raw["long"], True)
        self.assertEqual(raw["p1"], {})
        self.assertEqual(sorted(raw["p2"]), sorted(json.loads(ss.CORPUS.read_text(encoding="utf-8"))["long_documents"]["names"]))
        self.assertTrue(result["method"].startswith("P3"))

    def test_long_documents_follow_the_corpus_rule(self):
        corpus = {"tokens_per_document": 4, "documents": [{"id": "a", "text": "ab"}, {"id": "b", "text": "cd"}],
                  "long_documents": {"tokens": 9, "separator": "|", "names": ["long-a", "long-b"]}}
        fake = lambda base, text, special: [ord(c) for c in text]
        with mock.patch.object(ss, "tokenize", fake):
            short = ss.parity_documents("http://x", corpus, long=False)
            long = dict(ss.parity_documents("http://x", corpus, long=True))
        self.assertEqual(short, [("a", [97, 98]), ("b", [99, 100])])
        self.assertEqual("".join(map(chr, long["long-a"])), "ab|cd|ab|")
        self.assertEqual("".join(map(chr, long["long-b"])), "cd|ab|cd|")

    def test_corpus_is_public_domain_and_long_enough(self):
        corpus = json.loads(ss.CORPUS.read_text(encoding="utf-8"))
        documents = corpus["documents"]
        self.assertEqual([d["language"] for d in documents].count("ko"), 8)
        self.assertEqual([d["language"] for d in documents].count("en"), 8)
        self.assertLess(ss.CORPUS.stat().st_size, 400_000)
        for doc in documents:
            with self.subTest(doc=doc["id"]):
                self.assertTrue(doc["source"] and doc["rights"])
                self.assertRegex(doc["rights"], r"[Pp]ublic domain|Article 7")
                self.assertGreaterEqual(len(doc["text"]), 6500 if doc["language"] == "ko" else 10000)

    def test_realtext_windows_and_dataset(self):
        corpus = {"documents": [{"id": "a", "text": "x" * 40}, {"id": "b", "text": "y" * 40}]}
        with mock.patch.object(ss, "tokenize", lambda base, text, special: list(range(len(text)))):
            windows = ss.realtext_windows("http://x", corpus, 16, 8)
        self.assertEqual(len(windows), 8)
        self.assertEqual(len({(doc, start) for doc, start, _ in windows}), 8)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(ss, "detokenize", lambda base, ids: f"text {ids[0]} {len(ids)}"):
            path = Path(folder, "d.jsonl")
            result = ss.realtext_dataset(args(), windows, 5, 11, 512, path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            with self.assertRaises(ValueError):
                ss.realtext_dataset(args(), windows, 9, 11, 512, path)
        self.assertEqual(len(lines), 5)
        self.assertEqual(len({line["source"] for line in lines}), 5, "no window twice")
        self.assertEqual(result["dataset_source"], "parity_corpus.json windows")


class LegacySeedTests(unittest.TestCase):
    def test_sweep_and_latency_use_a_seed_per_point(self):
        seeds = []
        fake = lambda *a, seed=2026, **k: seeds.append(seed) or {"name": a[1], "returncode": 0}
        legacy = argparse.Namespace(base="http://x", tag="t", out_dir="o", max_num_batched_tokens=8192, max_model_len=65536)
        with mock.patch.object(verify_suites, "bench", fake), mock.patch.object(verify_suites, "emit"):
            verify_suites.suite_sweepsub(legacy)
            verify_suites.suite_latency(legacy)
        self.assertEqual(len(seeds), 3 + len(client_tests.benchmark_plan(65536)))
        self.assertEqual(len(set(seeds[:3])), 3)
        self.assertEqual(len(set(seeds[3:])), len(seeds) - 3)

    def test_client_tests_bench_passes_the_seed(self):
        run = mock.Mock(return_value=mock.Mock(returncode=1, stderr="x"))
        with mock.patch.object(client_tests.subprocess, "run", run):
            record = client_tests.bench(argparse.Namespace(base="http://x", tokenizer="t"), "p", 1, 4, 1024, 256, "o",
                                        seed=2031)
        command = run.call_args[0][0]
        self.assertEqual(command[command.index("--seed") + 1], "2031")
        self.assertEqual(record["seed"], 2031)


def flag_map(flags):
    return {flag: value for flag, value in zip(flags, flags[1:]) if flag.startswith("--")}


class ScreeningLayoutTests(unittest.TestCase):
    EXPECTED = {  # layout: (TP, PP, DP, DP per node, executor or DP backend, expert parallel)
        "A4": (4, 4, 1, 1, "ray", False), "C4": (4, 1, 4, 2, "ray", True), "C8": (2, 1, 8, 4, "ray", True),
        "C16": (1, 1, 16, 8, "ray", True), "E": (8, 1, 1, 1, "mp", True), "Err": (8, 1, 1, 1, "mp", True),
        "D8": (1, 1, 8, 8, "mp", True), "D24": (4, 1, 2, 2, "mp", True), "D42": (4, 2, 1, 1, "mp", False)}

    def test_new_layout_flags(self):
        for name, (tp, pp, dp, local, backend, ep) in self.EXPECTED.items():
            with self.subTest(layout=name):
                layout = ss.resolve_layout(PROTOCOL, name)
                flags = flag_map(layout["args"])
                self.assertEqual((int(flags["--tensor-parallel-size"]), int(flags["--pipeline-parallel-size"])), (tp, pp))
                self.assertEqual(int(flags.get("--data-parallel-size", 1)), dp)
                self.assertEqual(int(flags.get("--data-parallel-size-local", 1)), local)
                self.assertEqual("--enable-expert-parallel" in layout["args"], ep)
                self.assertEqual(flags.get("--data-parallel-backend", flags.get("--distributed-executor-backend")), backend)
                if dp > 1:
                    self.assertEqual(flags["--all2all-backend"], "allgather_reducescatter")
                    self.assertEqual(tp * pp * local, 8, "one node's 8 GPUs per DP group of engines")
                    self.assertEqual(dp // local, layout["nodes"])
                if backend == "ray" and dp > 1:
                    self.assertEqual(layout["env"]["VLLM_RAY_DP_PACK_STRATEGY"], "strict")
                self.assertEqual(tp * pp * dp, layout["gpus"])
                self.assertEqual(layout["gpus"], 8 * layout["nodes"])
                self.assertEqual(layout["checkpoint"], "fp8" if name.startswith(("A", "C")) else "nvfp4")
        self.assertEqual(ss.resolve_layout(PROTOCOL, "Err")["args"][-2:], ["--expert-placement-strategy", "round_robin"])
        self.assertIn("--decode-context-parallel-size", ss.resolve_layout(PROTOCOL, "B0dcp")["args"])

    def test_sensitivity_layouts_keep_the_system_wide_budget_of_a(self):
        server = PROTOCOL["server"]
        for name in ("C2s", "C4s", "C8s", "C16s"):
            with self.subTest(layout=name):
                layout = ss.resolve_layout(PROTOCOL, name)
                dp = int(flag_map(layout["args"])["--data-parallel-size"])
                self.assertTrue(layout["sensitivity"])
                self.assertEqual(layout["max_num_seqs"] * dp, server["max_num_seqs"])
                self.assertEqual(layout["max_num_batched_tokens"] * dp, server["max_num_batched_tokens"])

    def test_dp_ep_layouts_share_c2s_gathered_moe_batch(self):
        # allgather_reducescatter gathers DP x max_num_batched_tokens tokens per MoE forward (vLLM 0.23), and the
        # Marlin MoE workspace grows with them: the two-node FP8 DP layouts keep C2's 16,384 in total.
        for name, per_engine in (("C2", 8192), ("C4", 4096), ("C8", 2048), ("C16", 1024), ("D8", 1024), ("D24", 8192)):
            with self.subTest(layout=name):
                spec = ss.phase_spec(PROTOCOL, f"slo-{name}" if f"slo-{name}" in PROTOCOL["phases"] else f"scr-{name}", 1)
                flags = flag_map(spec["args"])
                self.assertEqual((spec["max_num_batched_tokens"], flags["--max-num-batched-tokens"]), (per_engine, str(per_engine)))
                if name.startswith("C"):
                    self.assertEqual(per_engine * int(flags["--data-parallel-size"]), 16384)
        self.assertIn("allgather_reducescatter", PROTOCOL["server"]["dp_ep_batch_rule"]["reason"])
        self.assertIn("cp_utils.py", PROTOCOL["layouts"]["B0dcp"]["expected"])

    def test_pd_layout_and_screening_phase_specs(self):
        pd = ss.phase_spec(PROTOCOL, "scr-PD", 0)
        self.assertEqual((pd["proxy"], pd["replicas"], pd["nodes"], pd["experimental"]), ("pd", 2, 1, True))
        roles = [json.loads(flag_map(args)["--kv-transfer-config"]) for args in (pd["args"], pd["peer_args"])]
        self.assertEqual([r["kv_role"] for r in roles], ["kv_producer", "kv_consumer"])
        self.assertTrue(all(r["kv_connector"] == "NixlConnector" and r["kv_load_failure_policy"] == "fail" for r in roles))
        self.assertEqual([a for a in pd["args"] if a not in pd["peer_args"]], [json.dumps(roles[0])])
        self.assertEqual(pd["node_env"], ["VLLM_NIXL_SIDE_CHANNEL_HOST"])
        self.assertEqual(pd["env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"], "5600")
        # PD alone leaves GPU headroom: the prefill server ran out of memory at the common 0.9 (layout note).
        self.assertEqual(pd["gpu_memory_utilization"], 0.85)
        self.assertEqual([flag_map(a)["--gpu-memory-utilization"] for a in (pd["args"], pd["peer_args"])], ["0.85", "0.85"])
        for phase in ("scr-D", "scr-Dref", "scr-Dx2", "scr-A", "scr-C4m"):
            with self.subTest(common_memory=phase):
                spec = ss.phase_spec(PROTOCOL, phase, 0)
                self.assertEqual(flag_map(spec["args"])["--gpu-memory-utilization"],
                                 str(PROTOCOL["server"]["gpu_memory_utilization"]))
        dx2 = ss.phase_spec(PROTOCOL, "scr-Dx2", 0)
        self.assertEqual((dx2["proxy"], dx2["peer_args"]), ("lb", dx2["args"]))
        self.assertEqual(ss.phase_spec(PROTOCOL, "scr-C4", 0)["peer_args"], [])
        for phase in [p for p in PROTOCOL["phases"] if p.startswith("scr-")]:
            with self.subTest(phase=phase):
                spec = ss.phase_spec(PROTOCOL, phase, 0)
                self.assertEqual(spec["suites"], ["screen"])
                self.assertEqual(spec["length_fallback"], phase != "scr-A-auto")
                self.assertEqual(flag_map(spec["args"])["--api-server-count"], "1")
        with self.assertRaises(ValueError):
            ss.phase_entry(PROTOCOL, "slo-PD")

    @unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs a POSIX bash")
    def test_pd_shell_spec_round_trips_through_bash(self):
        spec = ss.phase_spec(PROTOCOL, "scr-PD", 0)
        script = ss.shell_spec(spec) + '\nprintf "%s\\n" "${SPEC_PEER_ARGS[@]}"; echo "$SPEC_PROXY $SPEC_EXPERIMENTAL ' \
                                       '$SPEC_FALLBACK ${SPEC_NODE_ENV[*]}"'
        lines = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(lines[:-1], spec["peer_args"])
        self.assertEqual(lines[-1], "pd 1 1 VLLM_NIXL_SIDE_CHANNEL_HOST")


class RoundLayoutTests(unittest.TestCase):
    def test_round_plan_takes_the_selected_layouts(self):
        self.assertEqual(ss.round_plan(PROTOCOL, 2, "fp8", "C4,A4"),
                         ["slo-A", "slo-A4", "slo-C4", "slo-C4s", "slo-A2", "cap-A", "cap-C4", "cap-A4"])
        self.assertEqual(ss.round_plan(PROTOCOL, 3, "fp8", "C4,B0dcp"), ["slo-A", "slo-C4", "slo-B0dcp", "slo-A2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 3, "nvfp4", "E,D8"), ["slo-D", "slo-E", "slo-D8", "slo-D2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 1, "fp8", ""), ss.round_plan(PROTOCOL, 1, "fp8"))
        for job, bad in (("fp8", "A"), ("fp8", "A2"), ("fp8", "C2s"), ("fp8", "PD"), ("fp8", "X9"), ("fp8", "C4,C4"),
                         ("fp8", "D42"), ("nvfp4", "D"), ("nvfp4", "D2"), ("nvfp4", "C8m"), ("nvfp4", "PD"), ("nvfp4", "A")):
            with self.subTest(job=job, layouts=bad), self.assertRaises(ValueError):
                ss.round_plan(PROTOCOL, 1, job, bad)
        self.assertEqual(ss.expand_plan(PROTOCOL, ["slo-round-nvfp4", "scr-D8"], 2, ["D24"])[-1], "scr-D8")
        uncontrolled = json.loads(json.dumps(PROTOCOL))
        uncontrolled["rounds"]["jobs"]["nvfp4"]["control"] = None
        self.assertEqual(ss.round_plan(uncontrolled, 3, "nvfp4"), ["slo-D", "slo-D42", "slo-D24"])
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D2", 2)["suites"], ["parity", "warmup", "partr", "slo", "paritylong"])
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D2", 2)["options"], ss.phase_spec(PROTOCOL, "slo-A2", 2)["options"],
                         "D2 mirrors A2")

    def test_generic_phases_for_selected_layouts(self):
        c4, d24 = ss.phase_spec(PROTOCOL, "slo-C4", 1), ss.phase_spec(PROTOCOL, "slo-D24", 3)
        self.assertEqual(c4["suites"], ["parity", "warmup", "partr", "slo", "realtext", "paritylong"])
        self.assertEqual(d24["suites"], ["parity", "warmup", "partr", "slo", "realtext", "soak", "paritylong"],
                         "W1real for every NVFP4 round layout")
        c4s = ss.phase_spec(PROTOCOL, "slo-C4s", 1)
        self.assertEqual((c4s["suites"], c4s["rates_key"], c4s["options"]["partr"]), (["warmup", "partr", "slo"], "C4", "c32"))
        self.assertEqual(ss.phase_spec(PROTOCOL, "cap-C8", 1)["max_model_len"], -1)
        dx2 = ss.phase_spec(PROTOCOL, "dx2-D24", 1)
        self.assertEqual((dx2["label"], dx2["replicas"], dx2["rates_key"], dx2["proxy"]), ("D24x2", 2, "Dx2", "lb"))
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D", 1)["suites"],
                         ["parity", "warmup", "partr", "slo", "realtext", "accuracy", "paritylong"],
                         "accuracy before the long check, which stays last")
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-D", 3)["suites"],
                         ["parity", "warmup", "partr", "slo", "realtext", "soak", "paritylong"])
        self.assertEqual(ss.phase_spec(PROTOCOL, "slo-A", 1)["suites"],
                         ["parity", "warmup", "partr", "slo", "realtext", "accuracy", "paritylong"])
        for phase in ("slo-A2", "slo-D2", "slo-B0", "slo-C8m", "slo-D42", "slo-D24"):
            self.assertNotIn("accuracy", ss.phase_spec(PROTOCOL, phase, 1)["suites"], phase)
        for phase in ("slo-D42", "slo-D24", "slo-B0", "slo-C8m"):
            self.assertIn("realtext", ss.phase_spec(PROTOCOL, phase, 1)["suites"], phase)
        for phase in ("slo-E", "slo-D2", "slo-A2"):
            self.assertNotIn("realtext", ss.phase_spec(PROTOCOL, phase, 1)["suites"], phase)
        fp8_only = json.loads(json.dumps(PROTOCOL))
        fp8_only["workloads"]["W1real"]["layouts"] = []
        self.assertNotIn("realtext", ss.phase_spec(fp8_only, "slo-D24", 1)["suites"])
        for phase in PROTOCOL["phases"]:
            suites = PROTOCOL["phases"][phase]["suites"]
            if "paritylong" in suites:
                self.assertEqual(suites[-1], "paritylong", f"{phase}: the long check runs last")

    def test_frozen_rate_grids_and_overrides(self):
        grids, ran = PROTOCOL["rate_grids"]["grids"], []
        fake = lambda arguments, suite, points, protocol, method: ran.append(points) or {"points": []}
        with mock.patch.object(ss, "run_points", fake):
            frozen = ss.suite_slo(args(phase="slo-C8m", layout="C8", rates_key="C8m", options={"slo": {"workloads": ["W1", "W2"]}}))
            override = ss.suite_slo(args(phase="slo-C8m", layout="C8", rates_key="C8m", rates={"W1": [0.5, 0.7]},
                                         options={"slo": {"workloads": ["W1", "W2"]}}))
            ss.suite_slo(args(phase="dx2-D", layout="Dx2", rates_key="Dx2", options={"slo": {"workloads": ["W1"], "rate_set": "dx2"}}))
        rates = lambda points, workload: sorted(p["rate"] for p in points if p["workload"] == workload)
        self.assertEqual((rates(ran[0], "W1"), rates(ran[0], "W2")), (grids["C8m"]["W1"], grids["C8m"]["W2"]))
        self.assertEqual((rates(ran[1], "W1"), rates(ran[1], "W2")), ([0.5, 0.7], grids["C8m"]["W2"]))
        self.assertEqual((frozen["rates_override"], override["rates_override"]), (None, {"W1": [0.5, 0.7]}))
        self.assertEqual(rates(ran[2], "W1"), grids["Dx2"]["W1"], "the frozen grid wins over the screening's rate set")
        for workload in ("W1", "W2"):
            self.assertEqual(grids["Dx2"][workload], [round(2 * r, 2) for r in grids["D"][workload]])
        names = {name for spec in PROTOCOL["rounds"]["jobs"].values() for name in (spec["reference"], *spec["layouts"])}
        self.assertTrue(names <= set(grids), "every round layout has a frozen grid")
        self.assertEqual(len({json.dumps(grids[name]) for name in names}), 1, "one grid per workload for every layout")
        self.assertTrue({0.6, 1.0} <= set(grids["A"]["W1"]), "the screening's passing rate and D's failing rate")
        self.assertEqual(PROTOCOL["workloads"]["W1real"]["rates"], [0.6, 1.0], "W1real at two W1 grid rates")
        for phase in ("slo-A2", "slo-C8ms", "slo-D2"):
            spec = ss.phase_spec(PROTOCOL, phase, 1)
            self.assertEqual(ss.phase_rates(PROTOCOL, spec["rates_key"]), grids[spec["rates_key"]], phase)

    def test_rate_keys_are_the_candidates_own_names(self):
        keys = {phase: ss.phase_spec(PROTOCOL, phase, 3)["rates_key"]
                for phase in ("slo-Err", "slo-C1rr", "slo-B0dcp", "slo-C4", "slo-A2", "slo-C2s", "slo-C4s", "dx2-D24")}
        self.assertEqual(keys, {"slo-Err": "Err", "slo-C1rr": "C1rr", "slo-B0dcp": "B0dcp", "slo-C4": "C4", "slo-A2": "A",
                                "slo-C2s": "C2", "slo-C4s": "C4", "dx2-D24": "Dx2"})
        for phase, key in (("slo-Err", "Err"), ("slo-C1rr", "C1rr")):
            with self.subTest(phase=phase):
                spec = ss.phase_spec(PROTOCOL, phase, 3)
                self.assertIn("soak", spec["suites"])
                ran = []
                fake = lambda arguments, suite, points, protocol, method: ran.append(points) or {"points": []}
                with mock.patch.object(ss, "run_points", fake):
                    ss.suite_soak(args(round=3, layout=spec["label"], phase=phase, rates_key=spec["rates_key"],
                                       soak_rates={key: 0.9, "E": 0.3, "C1": 0.4}))
                self.assertEqual(ran[0][0]["rate"], 0.9, "the round-3 soak finds the candidate's own key")

    def test_mp_multinode_runs_node1_headless_with_node0s_engine_flags(self):
        launcher = PROTOCOL["server"]["launchers"]["mp-multinode"]
        value = lambda args, flag: args[args.index(flag) + 1]
        parallel = {flag: 1 for flag in ("--tensor-parallel-size", "--data-parallel-size", "--data-parallel-size-local",
                                         "--data-parallel-backend", "--distributed-executor-backend", "--data-parallel-address",
                                         "--data-parallel-rpc-port", "--max-num-batched-tokens")}
        c2 = ss.phase_spec(PROTOCOL, "scr-C2", 0)
        for phase, tp, dp, local, batched in (("scr-C4m", 4, 4, 2, 4096), ("scr-C8m", 2, 8, 4, 2048), ("scr-C16m", 1, 16, 8, 1024)):
            with self.subTest(phase=phase):
                spec = ss.phase_spec(PROTOCOL, phase, 0)
                node0, node1 = spec["args"], spec["peer_args"]
                self.assertEqual((spec["label"], spec["layout"], spec["launcher"], spec["nodes"], spec["replicas"], spec["proxy"]),
                                 (phase[4:-1], phase[4:], "mp-multinode", 2, 1, ""))
                for flag, expected in (("--tensor-parallel-size", tp), ("--data-parallel-size", dp),
                                       ("--data-parallel-size-local", local), ("--max-num-batched-tokens", batched),
                                       ("--data-parallel-backend", "mp"), ("--distributed-executor-backend", "mp"),
                                       ("--data-parallel-address", "@head_ip"), ("--data-parallel-rpc-port", 13345),
                                       ("--max-model-len", 131072), ("--api-server-count", 1), ("--max-num-seqs", 128)):
                    self.assertEqual(value(node0, flag), str(expected), flag)
                self.assertFalse({"--headless", "--data-parallel-start-rank"} & set(node0), "node0 is not hybrid LB")
                self.assertEqual(node1[-3:], ["--headless", "--data-parallel-start-rank", str(local)])
                self.assertEqual(node1[:-3], ss.without_flags(node0, launcher["api_only_flags"]), "identical engine flags")
                self.assertFalse(set(launcher["api_only_flags"]) & set(node1), "no API-server flag on node1")
                self.assertEqual(value(node1, "--reasoning-parser"), "deepseek_v3", "an engine argument in vLLM 0.23")
                self.assertEqual(spec["node_env"], ["VLLM_HOST_IP"])
                self.assertNotIn("VLLM_RAY_DP_PACK_STRATEGY", spec["env"])
                self.assertEqual(ss.without_flags(node0, parallel), ss.without_flags(c2["args"], parallel),
                                 "C2's common and engine flags apart from the parallel layout and the batch size")
                self.assertIn("SPEC_LAUNCHER=mp-multinode", ss.shell_spec(spec))
        self.assertEqual(ss.phase_spec(PROTOCOL, "scr-C2", 0)["peer_args"], [], "the Ray layouts keep one server")
        sensitivity = ss.phase_spec(PROTOCOL, "slo-C4ms", 1)
        self.assertEqual((sensitivity["launcher"], sensitivity["max_num_seqs"], sensitivity["max_num_batched_tokens"],
                          sensitivity["rates_key"]), ("mp-multinode", 32, 2048, "C4m"))
        self.assertEqual(ss.phase_spec(PROTOCOL, "cap-C16m", 1)["peer_args"][:6],
                         ["--reasoning-parser", "deepseek_v3", "--max-model-len", "-1", "--gpu-memory-utilization", "0.9"])
        self.assertEqual(ss.round_plan(PROTOCOL, 2, "fp8", "C8m"),
                         ["slo-A", "slo-C8m", "slo-C8ms", "slo-A2", "cap-A", "cap-C8m"])
        self.assertEqual(ss.without_flags(["--a", "1", "--b", "--c", "2"], {"--a": 1, "--b": 0}), ["--c", "2"])

    def test_plan_command_records_the_round(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder, "plan.json")
            with mock.patch("builtins.print") as shown:
                self.assertEqual(ss.main(["plan", "--round", "2", "--layouts", "B0,C8m", "--record", str(record),
                                          "slo-round-fp8"]), 0)
            data = json.loads(record.read_text())
            with mock.patch("builtins.print") as deviating:
                ss.main(["plan", "--round", "1", "--layouts", "D24", "--record", str(record), "slo-round-nvfp4"])
            nvfp4 = json.loads(record.read_text())
            with mock.patch("builtins.print"):
                ss.main(["plan", "--round", "0", "--layouts", "B0", "--record", str(record), "scr-A", "scr-C4"])
            screening = json.loads(record.read_text())
        self.assertEqual(shown.call_args[0][0], "slo-A slo-C8m slo-B0 slo-C8ms slo-A2 cap-A cap-B0 cap-C8m")
        self.assertEqual((data["round"], data["job"], data["reference"], data["control"], data["layouts"], data["deviation"]),
                         (2, "fp8", "A", "A2", ["B0", "C8m"], False))
        self.assertEqual((nvfp4["job"], nvfp4["control"], nvfp4["layouts"], nvfp4["frozen_layouts"], nvfp4["deviation"],
                          nvfp4["plan"]), ("nvfp4", "D2", ["D24"], ["D42", "D24"], True, ["slo-D", "slo-D24", "slo-D2"]))
        self.assertIn("deviation", " ".join(str(call) for call in deviating.call_args_list))
        self.assertEqual((screening["job"], screening["layouts"], screening["plan"]), (None, [], ["scr-A", "scr-C4"]))


class ScreenSuiteTests(unittest.TestCase):
    def test_screen_points_seeds_and_ladders(self):
        points = ss.screen_points(PROTOCOL, 0)
        self.assertEqual([p["name"] for p in points], ["R-isl1024-c1", "R-isl1024-c32", "R-isl8192-c1", "R-isl8192-c32",
                                                       "R-isl32768-c1", "R-isl32768-c32", "W1-p1-r0.6"])
        self.assertTrue(all(p["seed"] < 1000000 for p in points), "ROUND 0 seeds stay apart from every round")
        capacity = points[5]
        self.assertEqual((capacity["output_len"], capacity["capacity_point"], capacity["concurrency"]), (64, True, 32))
        self.assertEqual(points[1]["output_len"], 1024)
        self.assertEqual(points[-1]["seed"], ss.protocol_seed(0, 3, 1))
        self.assertEqual(PROTOCOL["workloads"]["W1"]["rates"][PROTOCOL["screen"]["w1"]["point"]], PROTOCOL["screen"]["w1"]["rate"])
        w1 = [p["rate"] for p in ss.screen_points(PROTOCOL, 0, "W1")[6:]]
        dx2 = [p["rate"] for p in ss.screen_points(PROTOCOL, 0, "dx2")[6:]]
        self.assertEqual(sorted(w1), PROTOCOL["workloads"]["W1"]["rates"])
        self.assertEqual(sorted(dx2), PROTOCOL["rate_sets"]["dx2"]["W1"])
        self.assertEqual([p["index"] for p in ss.screen_points(PROTOCOL, 0, "W1")[6:]], ss.rate_order(0, 3, 4))

    def test_a_light_reference_runs_facts_and_parity_only(self):
        spec = ss.phase_spec(PROTOCOL, "scr-Dref", 0)
        self.assertEqual((spec["label"], spec["layout"], spec["nodes"], spec["replicas"], spec["suites"]),
                         ("Dref", "D", 1, 1, ["screen"]))
        self.assertEqual(spec["options"]["screen"]["steps"], ["facts", "parity"])
        with mock.patch.object(ss, "screen_facts", return_value={"max_model_len": 131072}), \
                mock.patch.object(ss, "screen_parity", return_value={"documents": 4}), \
                mock.patch.object(ss, "suite_warmup") as warmup, mock.patch.object(ss, "run_points") as points, \
                mock.patch.object(ss.vs, "server_healthy", return_value=True):
            result = ss.suite_screen(args(round=0, tag="scr-Dref", phase="scr-Dref", layout="Dref", options=spec["options"]))
        warmup.assert_not_called()
        points.assert_not_called()
        self.assertEqual({k: v["ok"] for k, v in result["steps"].items()}, {"facts": True, "parity": True})
        self.assertEqual((result["plan"], result["points"]), ([], []))
        self.assertNotIn("aborted", result, "no points planned, none missing")
        self.assertEqual(PROTOCOL["screen"]["references"][-1], "scr-Dref", "the PD rerun job's parity reference")

    def test_screen_suite_records_failed_steps_and_still_measures(self):
        metrics = ('vllm:cache_config_info{block_size="64",engine="0",num_gpu_blocks="3300"} 1.0\n'
                   'vllm:cache_config_info{block_size="64",engine="1",num_gpu_blocks="3301"} 1.0\n')
        ran = {}

        def fake_run_points(arguments, suite, points, protocol, method):
            ran.update(suite=suite, names=[p["name"] for p in points])
            return {"method": method, "plan": ran["names"], "points": [{"name": ran["names"][0]}]}

        with mock.patch.object(ss.vs, "call", return_value=({"data": [{"max_model_len": 131072}]}, 0.1)), \
                mock.patch.object(ss, "fetch_metrics", return_value=metrics), \
                mock.patch.object(ss, "tokenize", side_effect=RuntimeError("HTTP 500")), \
                mock.patch.object(ss, "suite_warmup", return_value={}), \
                mock.patch.object(ss.vs, "server_healthy", return_value=True), \
                mock.patch.object(ss, "run_points", fake_run_points):
            result = ss.suite_screen(args(round=0, tag="scr-C2", phase="scr-C2", layout="C2"))
            with self.assertRaises(ValueError):
                ss.suite_screen(args(round=1))
        self.assertEqual(result["facts"], {"max_model_len": 131072, "kv_tokens_per_engine": {"0": 211200, "1": 211264}})
        self.assertEqual({k: v["ok"] for k, v in result["steps"].items()},
                         {"facts": True, "parity": False, "warmup": True, "points": True})
        self.assertIn("HTTP 500", result["steps"]["parity"]["error"])
        self.assertEqual(ran["suite"], "screen")
        self.assertEqual(len(result["plan"]), 7)
        self.assertNotIn("aborted", result)

    def test_screen_parity_writes_raw_data_for_the_reference_comparison(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(ss, "tokenize", lambda base, text, special: list(range(3000))), \
                mock.patch.object(ss, "reset_prefix_cache", return_value=True), \
                mock.patch.object(ss, "teacher_forced", lambda base, name, ids: {"tokens": len(ids), "chosen": [-0.5]}), \
                mock.patch.object(ss, "greedy", lambda base, name, ids, n: {"prompt_tokens": len(ids), "generated": [1] * n}):
            summary = ss.screen_parity(args(round=0, tag="scr-A", phase="scr-A", out_dir=folder), PROTOCOL)
            raw = json.loads(Path(folder, "parity-scr-A.json").read_text())
        settings = PROTOCOL["screen"]["parity"]
        self.assertEqual(sorted(raw["p1"]), sorted(settings["documents"]))
        self.assertTrue(all(v["tokens"] == settings["tokens"] for v in raw["p1"].values()))
        self.assertTrue(all(v["prompt_tokens"] == 512 and len(v["generated"]) == 128 for v in raw["p2"].values()))
        self.assertEqual((raw["phase"], raw["long"], raw["screen"]), ("scr-A", False, True))
        self.assertIn("first launch", summary["method"])

    def test_prefill_decode_prompt_token_gate_counts_the_prefill_server(self):
        per_engine = {"prefill:0": {"vllm:prompt_tokens_total": 2048.0}, "decode:0": {"vllm:prompt_tokens_total": 2048.0}}
        record = {"returncode": 0, "failed": 0, "total_input_tokens": 2048, "idle_before": True, "prefix_cache_reset": True,
                  "concurrency": None, "deltas": {"per_engine": per_engine, "totals": {"vllm:prompt_tokens_total": 4096.0}}}
        self.assertTrue(ss.evaluate_gates(record, "open", PROTOCOL["gates"])["valid"])
        doubled = {"0": per_engine["prefill:0"], "1": per_engine["decode:0"]}
        record["deltas"]["per_engine"] = doubled
        self.assertFalse(ss.evaluate_gates(record, "open", PROTOCOL["gates"])["valid"], "two plain engines must add up")


if __name__ == "__main__":
    unittest.main()
