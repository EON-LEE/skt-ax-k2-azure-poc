"""Pre-registered accuracy subset of the serving protocol (slo_protocol.json accuracy, docs/BENCHMARK_PROTOCOL.md).

slo_suites.py runs it as the suite `accuracy` of slo-A (FP8 round job) and slo-D (NVFP4 round job) in round 1,
before the last suite (P3), so it needs no launch of its own. Both servers answer the same items: all 700 KoBALT
items and 300 seeded CLIcK items. Datasets, prompts and scorers come from aml/src/evals.py: pinned revisions,
hash-checked downloads with retries, the final answer after the reasoning. Requests are greedy (temperature 0)
with thinking on; a reply that reaches max_tokens counts as wrong. bench_stats.py pairs the two servers' items
(McNemar, non-inferiority margin).

Any failure (download, client, time limit, server gone) is part of the result, never an exception. Only this suite
fails; the phase's other suites and the round's validity do not depend on it.
"""
import concurrent.futures as futures
import hashlib
import sys
import threading
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEM_FIELDS = ("dataset", "key", "correct", "pred", "gold", "finish_reason", "completion_tokens", "error")
LOAD_TRIES, LOAD_PAUSE = 3, 60.0  # each try is evals.fetch's own four attempts


def load_evals():
    """aml/src/evals.py, from the folder above bench/ (/tmp/axk2src in a job)."""
    parent = str(HERE.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        import evals
    finally:
        if added:
            sys.path.remove(parent)
    return evals


def select_items(evals, settings, data_dir):
    """{dataset: evals.Suite} with the protocol's items: evals.py's loaders and its sample() (seed, item count)."""
    loaders = {"kobalt": (evals.load_kobalt, evals.score_kobalt), "click": (evals.load_click, evals.score_click)}
    chosen = {}
    for name, spec in settings["datasets"].items():
        load, score = loaders[name]
        for attempt in range(1, LOAD_TRIES + 1):
            try:
                items = load(data_dir)
                break
            except Exception:
                if attempt == LOAD_TRIES:
                    raise
                time.sleep(LOAD_PAUSE)
        suite = evals.Suite(name, items, score, spec["max_tokens"], 1)
        evals.sample(suite, spec["items"], settings["seed"])
        chosen[name] = suite
    return chosen


def ordered_units(suites, seed):
    """Both datasets in one order (long KoBALT replies mixed with short CLIcK ones), the same on every server."""
    units = [(suite, item) for suite in suites.values() for item in suite.items]
    return sorted(units, key=lambda u: hashlib.sha256(f"{seed}|order|{u[0].name}|{u[1]['key']}".encode()).hexdigest())


def request_body(evals, suite, item, request):
    return {"model": evals.MODEL, "messages": item["messages"], "max_tokens": suite.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": request["temperature"], "top_p": request["top_p"],
            "chat_template_kwargs": {"enable_thinking": request["thinking"]},
            "seed": int(hashlib.sha256(evals.unit_id(suite.name, item["key"], 0).encode()).hexdigest()[:7], 16)}


class Halt:
    """The stop shared by every item: once set (time limit, server gone) the open requests are dropped and the
    remaining items return at once, with the reason."""

    def __init__(self):
        self.event, self.reason = threading.Event(), None

    def set(self, reason):
        if not self.event.is_set():
            self.reason = reason
            self.event.set()


def score_item(evals, base, suite, item, settings, halt):
    """One item's record. A connection failure waits for /health and retries; an HTTP 400 does not. When /health does
    not come back, every remaining item stops."""
    record, error, attempt = {"dataset": suite.name, "key": item["key"], "gold": item["gold"]}, None, 0
    stop = halt.event
    while attempt < settings["attempts"] and not stop.is_set():
        attempt += 1
        try:
            reply = evals.stream_chat(base, request_body(evals, suite, item, settings["request"]), stop=stop)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                detail = ""
            error = f"HTTP {exc.code}: {detail}"
            if exc.code == 400:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        else:
            content = evals.final_answer(reply["content"])
            try:
                record.update(suite.score(item, content))
            except Exception as exc:  # a scorer failure is a wrong answer, reported
                record.update(correct=False, pred=None, score_error=f"{type(exc).__name__}: {exc}"[:200])
            record.update(finish_reason=reply["finish_reason"], hit_limit=reply["finish_reason"] == "length",
                          completion_tokens=(reply.get("usage") or {}).get("completion_tokens"),
                          seconds=reply.get("seconds"), attempts=attempt)
            if record["hit_limit"]:
                record["correct"] = False
            return record
        if attempt < settings["attempts"] and not stop.is_set() and not evals.wait_healthy(base, limit=600, stop=stop):
            halt.set("the server stopped answering /health")
            break
    if stop.is_set():
        error = f"stopped: {halt.reason}"
    record.update(correct=None, pred=None, error=error or "not run", attempts=attempt)
    return record


def run(evals, base, units, settings, halt=None):
    """Every unit with `concurrency` in flight; at the time limit, or when the server is gone, the open requests are
    dropped and the rest is not run."""
    halt, done, lock = halt or Halt(), [0], threading.Lock()
    timer = threading.Timer(settings["timeout_seconds"], halt.set, args=("the suite's time limit",))
    timer.daemon = True
    timer.start()

    def one(unit):
        record = score_item(evals, base, unit[0], unit[1], settings, halt)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(units):
                print(f"AXK2_ACCURACY {done[0]}/{len(units)} answered", flush=True)
        return record

    try:
        with futures.ThreadPoolExecutor(settings["concurrency"]) as pool:
            return list(pool.map(one, units))
    finally:
        timer.cancel()


def summarize(records):
    """Per dataset: items, scored, correct, accuracy, replies cut at max_tokens, unscored items, mean completion tokens."""
    out = {}
    for name in sorted({r["dataset"] for r in records}):
        rows = [r for r in records if r["dataset"] == name]
        scored = [r for r in rows if r.get("correct") is not None]
        tokens = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
        out[name] = {"items": len(rows), "scored": len(scored), "correct": sum(1 for r in scored if r["correct"]),
                     "accuracy": round(sum(1 for r in scored if r["correct"]) / len(scored), 4) if scored else None,
                     "hit_limit": sum(1 for r in rows if r.get("hit_limit")), "unscored": len(rows) - len(scored),
                     "score_errors": sum(1 for r in rows if r.get("score_error")),
                     "mean_completion_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None}
    return out


def run_suite(args, protocol, data_dir=None):
    """The suite's result: settings, dataset pins, per-dataset summary and one row per item (ITEM_FIELDS); `failed`
    names what stopped it."""
    settings, began = protocol["accuracy"], time.time()
    result = {"method": "pre-registered accuracy subset (slo_protocol.json accuracy): seeded KoBALT and CLIcK items, "
                        "greedy with thinking, scored as aml/src/evals.py does",
              "phase": args.phase or args.tag, "layout": args.layout or args.tag,
              "settings": {k: settings[k] for k in ("seed", "datasets", "request", "concurrency", "timeout_seconds")}}
    try:
        evals = load_evals()
        result["pins"] = {"kobalt": evals.KOBALT["rev"], "click": evals.CLICK["rev"]}
        suites = select_items(evals, settings, data_dir or settings["data_dir"])
    except Exception as exc:
        return {**result, "failed": f"datasets: {type(exc).__name__}: {exc}"[:600], "seconds": round(time.time() - began, 1)}
    halt = Halt()
    try:
        records = run(evals, args.base, ordered_units(suites, settings["seed"]), settings, halt)
    except Exception as exc:
        return {**result, "failed": f"client: {type(exc).__name__}: {exc}"[:600], "seconds": round(time.time() - began, 1)}
    unscored = sum(1 for r in records if r.get("correct") is None)
    return {**result, "summary": summarize(records), "unscored": unscored, "stopped": halt.reason,
            "item_fields": list(ITEM_FIELDS),
            "items": [[r.get(field) for field in ITEM_FIELDS] for r in records],
            "seconds": round(time.time() - began, 1)}
