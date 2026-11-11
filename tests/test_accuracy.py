import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "aml" / "src" / "bench"), str(ROOT / "aml" / "src")]
import accuracy  # noqa: E402
import slo_suites as ss  # noqa: E402
del sys.path[:2]

EVALS = accuracy.load_evals()
PROTOCOL = ss.load_protocol()


def protocol(kobalt=12, click=18, **settings):
    data = json.loads(json.dumps(PROTOCOL))
    data["accuracy"]["datasets"]["kobalt"]["items"] = kobalt
    data["accuracy"]["datasets"]["click"]["items"] = click
    data["accuracy"].update(concurrency=4, **settings)
    return data


def kobalt_items(n=40):
    return [{"key": f"kb-{i}", "gold": "ABCDEFGHIJ"[i % 10],
             "messages": [{"role": "user", "content": f"item kb-{i}"}], "meta": {"class": "Syntax", "level": 1}}
            for i in range(n)]


def click_items(n=60):
    return [{"key": f"ck-{i}", "gold": "ABCD"[i % 4], "messages": [{"role": "user", "content": f"item ck-{i}"}],
             "meta": {"n_choices": 4, "group": "Culture", "category": "History"}} for i in range(n)]


def reply(content, finish="stop", tokens=100):
    return {"content": content, "reasoning_chars": 50, "finish_reason": finish, "usage": {"completion_tokens": tokens},
            "ttft": 0.1, "seconds": 0.2}


def answers(wrong=(), cut=()):
    """A fake streamed chat: the gold answer unless the item is in `wrong`; items in `cut` reach max_tokens."""
    gold = {item["key"]: item["gold"] for item in kobalt_items() + click_items()}

    def chat(base, body, timeout=3600, stop=None):
        key = re.search(r"item (\S+)", body["messages"][-1]["content"]).group(1)
        letter = ("B" if gold[key] == "A" else "A") if key in wrong else gold[key]
        text = f"<think>reasoning</think>정답은 {letter}입니다." if key.startswith("kb") else f"정답: {letter}"
        return reply(text, "length" if key in cut else "stop")
    return chat


def run(settings=None, chat=None, healthy=True, **patches):
    arguments = argparse.Namespace(base="http://x", phase="slo-D", layout="D", tag="slo-D")
    with mock.patch.object(EVALS, "load_kobalt", patches.get("load_kobalt", lambda d: kobalt_items())), \
            mock.patch.object(EVALS, "load_click", patches.get("load_click", lambda d: click_items())), \
            mock.patch.object(EVALS, "stream_chat", chat or answers()), \
            mock.patch.object(EVALS, "wait_healthy", lambda base, limit=0, stop=None: healthy), \
            mock.patch.object(accuracy, "LOAD_PAUSE", 0):
        return accuracy.run_suite(arguments, settings or protocol(), data_dir="unused")


class AccuracySuiteTests(unittest.TestCase):
    def test_protocol_settings(self):
        settings = PROTOCOL["accuracy"]
        self.assertEqual({k: v["items"] for k, v in settings["datasets"].items()}, {"kobalt": 700, "click": 300})
        self.assertEqual(settings["timeout_seconds"], 10800)
        self.assertEqual((settings["request"]["temperature"], settings["comparison"]["margin"]), (0.0, -0.03))
        self.assertEqual(settings["pair"], {"fp8": "slo-A", "nvfp4": "slo-D"})
        self.assertEqual(settings["rounds"], [1])
        self.assertEqual(settings["phases"], [p for p in PROTOCOL["phases"] if "accuracy" in PROTOCOL["phases"][p]["suites"]])
        self.assertIn("pinned revision", settings["datasets"]["kobalt"]["source"])

    def test_the_same_seeded_items_scored_as_evals_does(self):
        first, second = run(), run()
        keys = lambda result: [row[:2] for row in result["items"]]
        self.assertEqual(keys(first), keys(second), "the same items in the same order on every server")
        self.assertEqual(first["item_fields"], list(accuracy.ITEM_FIELDS))
        chosen = {d: sorted(k for ds, k in keys(first) if ds == d) for d in ("kobalt", "click")}
        rank = lambda name, key: hashlib.sha256(f"20261110|sample|{name}|{key}".encode()).hexdigest()
        for name, items, n in (("kobalt", kobalt_items(), 12), ("click", click_items(), 18)):
            expected = sorted(sorted((i["key"] for i in items), key=lambda k: rank(name, k))[:n])
            self.assertEqual(chosen[name], expected, "evals.py sample() with the protocol's seed")
        self.assertEqual({k: (v["items"], v["accuracy"]) for k, v in first["summary"].items()},
                         {"kobalt": (12, 1.0), "click": (18, 1.0)})
        self.assertEqual((first["unscored"], first.get("failed")), (0, None))
        self.assertEqual(first["pins"], {"kobalt": EVALS.KOBALT["rev"], "click": EVALS.CLICK["rev"]})

    def test_all_700_kobalt_items_and_300_seeded_click_items(self):
        settings = PROTOCOL["accuracy"]
        with mock.patch.object(EVALS, "load_kobalt", lambda d: kobalt_items(700)), \
                mock.patch.object(EVALS, "load_click", lambda d: click_items(1995)):
            suites = accuracy.select_items(EVALS, settings, "unused")
        self.assertEqual([i["key"] for i in suites["kobalt"].items], [f"kb-{i}" for i in range(700)], "every KoBALT item, in order")
        self.assertEqual(len(suites["click"].items), 300)
        rank = lambda key: hashlib.sha256(f"20261110|sample|click|{key}".encode()).hexdigest()
        self.assertEqual(sorted(i["key"] for i in suites["click"].items),
                         sorted(sorted((f"ck-{i}" for i in range(1995)), key=rank)[:300]))

    def test_a_dead_server_stops_every_item_at_once(self):
        def refused(base, body, timeout=3600, stop=None):
            raise ConnectionRefusedError("connection refused")
        began = time.time()
        result = run(chat=refused, healthy=False)
        self.assertLess(time.time() - began, 30, "no item waits for the 3-hour limit")
        self.assertEqual(result["stopped"], "the server stopped answering /health")
        errors = {dict(zip(accuracy.ITEM_FIELDS, row))["error"] for row in result["items"]}
        self.assertEqual(errors, {"stopped: the server stopped answering /health"})
        self.assertEqual(result["unscored"], 30)

    def test_greedy_thinking_requests_and_cut_replies_are_wrong(self):
        bodies, base = [], answers(cut=tuple(f"kb-{i}" for i in range(40)), wrong=("ck-1", "ck-2", "ck-3", "ck-5"))
        result = run(chat=lambda b, body, timeout=3600, stop=None: bodies.append(body) or base(b, body))
        self.assertTrue(all(b["temperature"] == 0.0 and b["top_p"] == 1.0 and b["stream"]
                            and b["chat_template_kwargs"] == {"enable_thinking": True} for b in bodies))
        self.assertEqual({b["max_tokens"] for b in bodies}, {16384, 8192})
        self.assertEqual(result["summary"]["kobalt"]["accuracy"], 0.0, "a reply cut at max_tokens counts as wrong")
        self.assertEqual(result["summary"]["kobalt"]["hit_limit"], 12)
        rows = {row[1]: dict(zip(accuracy.ITEM_FIELDS, row)) for row in result["items"]}
        for key, row in rows.items():
            if key.startswith("ck"):
                self.assertEqual(row["correct"], key not in ("ck-1", "ck-2", "ck-3", "ck-5"), key)

    def test_retries_and_request_errors(self):
        calls = {}

        def flaky(base, body, timeout=3600, stop=None):
            key = re.search(r"item (\S+)", body["messages"][-1]["content"]).group(1)
            calls[key] = calls.get(key, 0) + 1
            if key.startswith("kb") and calls[key] == 1:
                raise ConnectionResetError("server restarting")
            if key.startswith("ck"):
                raise urllib.error.HTTPError("http://x", 400, "bad", {}, io.BytesIO(b"bad request"))
            return answers()(base, body)
        result = run(chat=flaky)
        self.assertEqual(result["summary"]["kobalt"]["accuracy"], 1.0, "a connection failure is retried")
        self.assertTrue(all(calls[k] == 2 for k in calls if k.startswith("kb")))
        self.assertTrue(all(calls[k] == 1 for k in calls if k.startswith("ck")), "a 400 is not retried")
        self.assertEqual((result["summary"]["click"]["unscored"], result["unscored"]), (18, 18))

    def test_a_download_failure_fails_only_the_accuracy_suite(self):
        def offline(data_dir):
            raise OSError("huggingface.co unreachable")
        result = run(load_click=offline)
        self.assertTrue(result["failed"].startswith("datasets: OSError"), result)
        self.assertNotIn("items", result)
        emitted = []
        arguments = argparse.Namespace(base="http://x", tag="slo-A", suites="accuracy", out_dir=tempfile.mkdtemp(),
                                       round=1, phase="slo-A", layout="A", protocol=str(ss.PROTOCOL))
        with mock.patch.object(EVALS, "load_kobalt", offline), mock.patch.object(accuracy, "LOAD_PAUSE", 0), \
                mock.patch.object(ss.vs, "emit", lambda out, tag, name, data: emitted.append((name, data))), \
                mock.patch.object(ss.vs, "server_healthy", lambda base: True):
            self.assertEqual(ss.run_suites(arguments), 0, "the phase's client does not fail")
        self.assertEqual(emitted[0][0], "accuracy")
        self.assertIn("unreachable", emitted[0][1]["failed"])

    def test_the_time_limit_drops_the_rest(self):
        def slow(base, body, timeout=3600, stop=None):
            time.sleep(0.05)
            return answers()(base, body)
        result = run(settings=protocol(timeout_seconds=0.12), chat=slow)
        self.assertGreater(result["unscored"], 0)
        errors = {dict(zip(accuracy.ITEM_FIELDS, row))["error"] for row in result["items"]} - {None}
        self.assertEqual(errors, {"stopped: the suite's time limit"})
        self.assertEqual(result["stopped"], "the suite's time limit")


if __name__ == "__main__":
    unittest.main()
