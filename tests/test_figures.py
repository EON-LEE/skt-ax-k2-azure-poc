import importlib.util
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report"

import sys
sys.path.insert(0, str(REPORT))
try:
    import make_figures
finally:
    sys.path.pop(0)


class FigureDataTests(unittest.TestCase):
    def test_load_throughput_series(self):
        data = make_figures.load_throughput()
        series = data["series"]
        self.assertEqual(len(series["b200_fp8_k2"]["points"]), 8)
        self.assertEqual(len(series["a100_fp8_native"]["points"]), 8)
        self.assertEqual(len(series["a100_fp8_dense"]["points"]), 8)
        self.assertEqual(len(series["a100_nvfp4"]["points"]), 6)
        native = {p["input_tokens"]: p for p in series["a100_fp8_native"]["points"]}
        dense = {p["input_tokens"]: p for p in series["a100_fp8_dense"]["points"]}
        nvfp4 = {p["input_tokens"]: p for p in series["a100_nvfp4"]["points"]}
        b200 = {p["input_tokens"]: p for p in series["b200_fp8_k2"]["points"]}
        self.assertAlmostEqual(native[1024]["total_token_throughput"], 814.5934910317782)
        self.assertAlmostEqual(dense[8192]["total_token_throughput"], 3070.3062463058586)
        self.assertAlmostEqual(nvfp4[32768]["total_token_throughput"], 1674.0)
        self.assertEqual(b200[120000]["total_tok_s"], 12300.0)
        self.assertAlmostEqual(native[8192]["total_token_throughput"] / b200[8192]["total_tok_s"], 0.5050629, places=5)
        self.assertAlmostEqual((nvfp4[1024]["total_token_throughput"] / 8) / (native[1024]["total_token_throughput"] / 16), 2.2708, places=3)

    def test_published_json_shape(self):
        payload = json.loads((REPORT / "skt_published.json").read_text(encoding="utf-8"))
        self.assertIn("series", payload)
        required = {"source", "hardware", "gpu_count", "conditions", "points"}
        for name, item in payload["series"].items():
            with self.subTest(name=name):
                self.assertTrue(required <= set(item))
                self.assertGreater(len(item["source"]), 10)
                self.assertIsInstance(item["gpu_count"], int)
                self.assertTrue(item["points"])
                for point in item["points"]:
                    self.assertIn("input_tokens", point)
                    self.assertIn("total_tok_s", point)
                    self.assertIn("output_tok_s", point)
        fig9 = payload["series"]["b200_nvfp4_k2_fig9"]
        self.assertEqual(fig9["gpu_count"], 4)
        self.assertIn("SERVING_DESIGN", fig9["conditions"])
        self.assertEqual([p["total_tok_s"] for p in fig9["points"][:3]], [2300, 3300, 4800])

    def test_render_outputs_when_matplotlib_available(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not importable")
        base = REPORT / "figures"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="_test_figures_", dir=base) as tmp:
            out = Path(tmp)
            make_figures.render_all(out)
            expected = {
                "fig7-overlay.png",
                "fig9-overlay.png",
                "per-gpu-throughput.png",
                "cost-per-1m-output-tokens.png",
                "throughput.json",
                "throughput-tables.md",
            }
            self.assertEqual(expected, {p.name for p in out.iterdir()})
            for name in expected:
                path = out / name
                self.assertGreater(path.stat().st_size, 100)
            for name in [n for n in expected if n.endswith(".png")]:
                self.assertLess((out / name).stat().st_size, 400_000)

    def test_throughput_tables_carry_review_caveats(self):
        """2026-11-09 review: prefix-cache-tainted points are flagged, cost is marked superseded, low priority is gone."""
        self.assertNotIn("low_priority", make_figures.PRICE_METERS)
        base = REPORT / "figures"
        with tempfile.TemporaryDirectory(prefix="_test_tables_", dir=base) as tmp:
            make_figures.write_tables(make_figures.load_throughput(), Path(tmp))
            text = (Path(tmp) / "throughput-tables.md").read_text(encoding="utf-8")
        self.assertIn("| 1K | 1,900 | 815 | 0.43 | 980 | 0.52 |", text)
        self.assertIn("| 8K | 4,800 | 2,424† | 0.51† | 3,070† | 0.64† |", text)
        self.assertIn("| 16K | 6,300 | 2,275 | 0.36 | 3,637 | 0.58 |", text)
        self.assertIn("B200 대비 1K 0.43, 8K 0.39, 32K 0.25", text)
        self.assertIn("| 4K | 450 | 106† | 138† | 210 | 1,200 |", text)
        self.assertIn("그림에 표시되지 않은 가정", text)
        self.assertIn("| C128, 1K in, 256 out | FP8 2 nodes | 28.01 | 17.93 | 12.32 | 8.82 |", text)
        self.assertIn("재측정 전까지 사용하지 않는다", text)
        self.assertIn("2026-03-31부터 Spot", text)
        self.assertNotIn("| Low priority |", text)
        self.assertIn("](../REVIEW_2026-11-09.md)", text)
        self.assertTrue((REPORT / "REVIEW_2026-11-09.md").is_file())

class BenchmarkTests(unittest.TestCase):
    def evidence(self, folder):
        path = Path(folder) / "a100-benchmarks.json"
        path.write_text(json.dumps({"run": "ev-test", "summary": {
            "aime": {"score": 0.9, "ci95_half": 0.05, "n_items": 30, "repeats": 8},
            "kobalt": {"score": 0.7, "ci95_half": 0.03, "n_items": 700, "repeats": 1},
            "niah": {"score": 1.0, "ci95_half": None, "grid": {"8192": {"0.5": True}}}}}), encoding="utf-8")
        return path

    def test_rows_merge_published_and_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = make_figures.load_benchmarks(self.evidence(tmp))
        rows = {r["benchmark"]: r for r in bench["rows"]}
        self.assertEqual(rows["AIME26"]["a100"], 90.0)
        self.assertEqual(rows["AIME26"]["a100_ci95"], 5.0)
        self.assertEqual(rows["AIME26"]["skt"], 97.1)
        self.assertIsNone(rows["CLIcK"]["a100"])
        self.assertIsNone(rows["GPQA Diamond"]["a100"])
        self.assertEqual(bench["niah"]["a100"], 100.0)
        self.assertEqual(bench["run"], "ev-test")

    def test_table_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = make_figures.load_benchmarks(self.evidence(tmp))
            make_figures.write_benchmark_table(bench, Path(tmp))
            table = (Path(tmp) / "benchmark-table.md").read_text(encoding="utf-8")
            empty = make_figures.load_benchmarks(Path(tmp) / "absent.json")
        self.assertIn("| Math | AIME26 | 97.1 | **90.0** ± 5.0 |", table)
        self.assertIn("| Korean | CLIcK | 91.6 | 측정 전 |", table)
        self.assertIn("| Science & Knowledge | GPQA Diamond | 85.6 | – (gated) |", table)
        self.assertIn("| Long context | NIAH | 100 | **100.0** |", table)
        self.assertTrue(all(r["a100"] is None for r in empty["rows"]))
        self.assertIsNone(empty["run"])

    def test_partial_run_and_niah_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a100-benchmarks.json"
            path.write_text(json.dumps({"run": "ev-part", "summary": {
                "aime": {"score": 1.0, "ci95_half": 0.0, "n_items": 10, "items_expected": 30, "repeats": 2,
                         "n_generations": 10, "generations_expected": 60},
                "click": {"score": 0.855, "ci95_half": 0.061, "n_items": 131, "items_expected": 200, "repeats": 1,
                          "n_generations": 131, "generations_expected": 200}}}), encoding="utf-8")
            verification = Path(tmp) / "verification.json"
            verification.write_text(json.dumps({"per_phase": {"native-pp2": {"niah": {
                "hits": 9, "total": 9, "by_length": {"32768": {}, "131072": {}, "262144": {}}}}}}), encoding="utf-8")
            bench = make_figures.load_benchmarks(path, niah_fallback=verification)
            make_figures.write_benchmark_table(bench, Path(tmp))
            table = (Path(tmp) / "benchmark-table.md").read_text(encoding="utf-8")
            without = make_figures.load_benchmarks(path)
        self.assertIn("| Math | AIME26 | 97.1 | **100.0** (n=10/60 gen) |", table)
        self.assertIn("| Korean | CLIcK | 91.6 | **85.5** ± 6.1 (n=131/200) |", table)
        self.assertIn("| Long context | NIAH | 100 | **100.0** (9/9, 32K, 128K, 256K) |", table)
        self.assertIn("중간에 멈췄다", table)
        self.assertIn("먼저 돌린 검증 실행", table)
        self.assertIsNone(without["niah"]["a100"])

    def test_render_benchmarks(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not importable")
        with tempfile.TemporaryDirectory() as tmp:
            bench = make_figures.load_benchmarks(self.evidence(tmp))
            self.assertTrue(make_figures.render_benchmarks(bench, Path(tmp)))
            self.assertGreater((Path(tmp) / "benchmarks-vs-published.png").stat().st_size, 10_000)
            self.assertFalse(make_figures.render_benchmarks(make_figures.load_benchmarks(Path(tmp) / "x.json"), Path(tmp)))


class LogprobFigureTests(unittest.TestCase):
    def test_counts_and_recomputed_means(self):
        summary = make_figures.logprob_summary(make_figures.load_logprob())
        expected = {
            "native_tp8": (0.011357863592830761, 0.01162928085916264, 0.01108293307628752),
            "native_tp8_pp2": (0.011741000136817625, 0.011979215606307733, 0.011499701184246772),
        }
        self.assertEqual(set(summary), set(expected))
        for key, (total, inside, outside) in expected.items():
            with self.subTest(group=key):
                stats = summary[key]
                self.assertEqual(stats["positions"], 14618)
                self.assertEqual(stats["within_first_2048"], 7356)
                self.assertEqual(stats["beyond_first_2048"], 7262)
                self.assertAlmostEqual(stats["mean_abs_delta"], total, places=14)
                self.assertAlmostEqual(stats["within_mean_abs_delta"], inside, places=14)
                self.assertAlmostEqual(stats["beyond_mean_abs_delta"], outside, places=14)

    def test_invalid_series_rejected(self):
        original = make_figures.load_logprob()
        mutations = [
            lambda d: d["prompts"][0]["oracle"].pop(),
            lambda d: d["prompts"][0]["native_tp8"].__setitem__(0, float("nan")),
            lambda d: d["prompts"][0]["native_tp8"].__setitem__(0, float("-inf")),
            lambda d: d["prompts"][0]["native_tp8"].__setitem__(0, 0.1),
            lambda d: d["prompts"][0]["native_tp8"].__setitem__(0, True),
            lambda d: d["prompts"][0].update(dense=[]),
            lambda d: d["prompts"].pop(),
            lambda d: d.update(schema_version=2),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                data = copy.deepcopy(original)
                mutate(data)
                with self.assertRaises(ValueError):
                    make_figures.validate_logprob(data)

    def test_bom_extraction_preserves_only_plot_series(self):
        original = make_figures.load_logprob()
        reports = {"oracle": {}, "native": {}}
        specs = {"oracle": ("oracle", "oracle.series")}
        specs.update({
            key: ("native", spec[2]) for key, spec in make_figures.LOGPROB_GROUPS.items()
        })
        for key, (report, series) in specs.items():
            reports[report][series] = {
                p["name"] + ".chosen_logprob": p[key] for p in original["prompts"]
            }
            reports[report][series]["ko_short.top1"] = [123] * 12
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for report in reports:
                path = Path(tmp) / (report + ".json")
                path.write_text(
                    json.dumps({"reports": reports[report], "discard_metadata": "not_for_export"}),
                    encoding="utf-8-sig",
                )
                paths[report] = path
            out = Path(tmp) / "series.json"
            extracted = make_figures.extract_logprob(paths["oracle"], paths["native"], out)
            self.assertEqual(extracted["prompts"], original["prompts"])
            self.assertEqual(make_figures.load_logprob(out), extracted)
            self.assertNotIn("not_for_export", out.read_text(encoding="utf-8"))
            self.assertNotIn("top1", out.read_text(encoding="utf-8"))
            for key, (report, _) in specs.items():
                self.assertEqual(
                    extracted["sources"][key]["sha256"],
                    hashlib.sha256(paths[report].read_bytes()).hexdigest(),
                )
            self.assertEqual(set(extracted), {"schema_version", "sources", "prompts"})

    def test_two_groups_title_caption_and_histogram(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not importable")
        from matplotlib import pyplot as plt
        fig = make_figures.build_logprob_figure(make_figures.load_logprob())
        try:
            self.assertEqual(fig._suptitle.get_text(), make_figures.LOGPROB_TITLE)
            self.assertEqual(fig.texts[1].get_text().replace("\n", " "), make_figures.LOGPROB_CAPTION)
            left, right = fig.axes
            self.assertEqual(len(left.collections), 2)
            self.assertEqual([len(c.get_offsets()) for c in left.collections], [14618, 14618])
            self.assertEqual(
                [c.get_label() for c in left.collections],
                [spec[0] for spec in make_figures.LOGPROB_GROUPS.values()],
            )
            self.assertIn("log-prob", left.get_xlabel())
            self.assertIn("log-prob", left.get_ylabel())
            self.assertIn("0.1", right.get_xlabel())
            self.assertEqual(len(right.patches), 2)
            for histogram in right.patches:
                self.assertEqual(sum(histogram.get_xy()[:, 1]) / 2, 14618)
        finally:
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            make_figures.render_all(tmp, only="logprob")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["logprob-two.png"])
            self.assertGreater((Path(tmp) / "logprob-two.png").stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
