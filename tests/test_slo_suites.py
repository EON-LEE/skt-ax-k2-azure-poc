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
        self.assertEqual(ss.round_plan(PROTOCOL, 1),
                         ["slo-A", "slo-C2", "slo-D", "slo-C2s", "slo-A2", "cap-A", "cap-C2", "cap-D"])
        self.assertEqual(ss.round_plan(PROTOCOL, 2), ["slo-A", "slo-D", "slo-C2", "slo-A2"])
        self.assertEqual(ss.round_plan(PROTOCOL, 3), ["slo-A", "slo-C2", "slo-D", "slo-A2"])
        with self.assertRaises(ValueError):
            ss.round_plan(PROTOCOL, 6)
        self.assertEqual(ss.expand_plan(PROTOCOL, ["slo-round", "diag-A"], 2)[-1], "diag-A")
        with self.assertRaises(ValueError):
            ss.expand_plan(PROTOCOL, ["epc-fp8-tp8pp2"], 1)

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
                self.assertEqual(pairs["--gpu-memory-utilization"], "0.9")
                self.assertEqual(flags.count("--max-model-len"), 1)
                self.assertIn("--async-scheduling", flags)
                self.assertNotIn("--enforce-eager", flags)
                self.assertEqual(flags[:len(server["launcher_flags"])], server["launcher_flags"])
                self.assertEqual(flags[-len(server["common_flags"]):], server["common_flags"])
                self.assertEqual(spec["env"]["VLLM_SERVER_DEV_MODE"], "1")
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


if __name__ == "__main__":
    unittest.main()
