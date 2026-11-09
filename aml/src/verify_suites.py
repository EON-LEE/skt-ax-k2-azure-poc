"""Verification suites for the items still open after the first A100 runs.

tools        OpenAI tool calling through vLLM's hermes parser (the model card's parser): single,
             parallel, multi-argument, thinking + tool, no-tool, multi-turn with a tool result,
             streaming, named and required tool_choice.
niah         SKT's own long-context check (examples/vllm/niah_test.py from SKT-AI/A.X-K2): needle
             at depths 0.15/0.5/0.85 in 32K, 128K and 256K-token prompts, a distinct code per case.
determinism  Greedy output of one request: twice computed fresh, once from the prefix cache, and
             twice inside a batch of 16 concurrent requests.
docsweep     The tech report's serving benchmark (Fig. 7): vllm bench serve, random dataset,
             concurrency 32, 1,024 output tokens, input 1K-120K, with peak running/waiting
             requests and KV-cache usage sampled from /metrics.
docsweep32k  The same at input 1K-32K (for single-node runs, where 64K/120K would dominate the bill).
sweepsub     The same at input 1K/4K/8K only (for alternative layouts).
latency      The plan of the earlier 16-GPU FP8 latency/throughput table (client_tests.benchmark_plan):
             1 request at 1K/8K/32K input, 1K input at concurrency 1-128, 16K input at 4 and 16.
functional   Short Korean/English/arithmetic/code checks.
specstats    Speculative-decoding acceptance from /metrics.
epmatrix     The expert-parallel comparison's matrix, identical for every layout: input 1K/8K/32K x
             concurrency 1 and 32, 1,024 output tokens (TTFT, TPOT/ITL, throughput, peak KV use).

Each suite is written to <out-dir>/<tag>.<suite>.json and reported as soon as it finishes.
"""
import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from client_tests import benchmark_plan

MODEL = "axk2"
REPORT = Path(__file__).with_name("report.py")

# Tech report Fig. 7 (FP8, BF16 KV, one B200 node, concurrency 32, 1K output): (total, output) tok/s
DOC_FIG7 = {1024: (1900, 965), 2048: (2700, 897), 4096: (3600, 724), 8192: (4800, 540),
            16384: (6300, 367), 32768: (8200, 247), 65536: (10600, 165), 120000: (12300, 104)}
# Tech report Fig. 8 (FP8 + EAGLE3, same conditions): total tok/s
DOC_FIG8_TOTAL = {1024: 2400, 2048: 3100, 4096: 4000, 8192: 5900, 16384: 8200, 32768: 10400,
                  65536: 13800, 120000: 15100}

NEEDLE_TMPL = "A.X K2 프로젝트의 내부 인증 코드는 {code}이다. 이 코드는 반드시 기억해야 한다."
QUESTION = ("\n\n위 문서에서 'A.X K2 프로젝트의 내부 인증 코드'를 찾아라. "
            "코드 값만 정확히 답하고 다른 말은 하지 마라.")
FILLER_UNIT = ("그는 조용히 창밖을 바라보았다. 거리에는 사람들이 오가고 있었고, 낮은 구름이 "
               "천천히 흘러갔다. 오래된 건물의 벽에는 담쟁이가 자라 있었다. ")

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, for example Seoul"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights", "description": "Search flights between two cities on a date.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["origin", "destination", "date"]}}},
]


def call(base, path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    began = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.time() - began


def emit(out_dir, tag, suite, data):
    path = Path(out_dir) / f"{tag}.{suite}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if os.environ.get("AXK2_REPORT_PKGS"):
        env["PYTHONPATH"] = os.environ["AXK2_REPORT_PKGS"]
    subprocess.run([sys.executable, str(REPORT), f"node0.{tag}.{suite}", "--file", str(path)],
                   env=env, check=False, timeout=900)


def chat_request(base, messages, tools=None, tool_choice=None, thinking=False, max_tokens=512,
                 temperature=0.0, extra=None, timeout=3600):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    body.update(extra or {})
    payload, seconds = call(base, "/v1/chat/completions", body, timeout)
    choice = payload["choices"][0]
    message = choice["message"]
    return {"content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
            "tool_calls": message.get("tool_calls") or [], "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"), "seconds": round(seconds, 2)}


def parsed_calls(tool_calls):
    calls = []
    for item in tool_calls:
        function = item.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {"_unparsable": function.get("arguments")}
        calls.append({"name": function.get("name"), "arguments": arguments, "id": item.get("id")})
    return calls


def mentions(value, *names):
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(name.lower() in text for name in names)


def stream_tool_call(base, messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 512, "temperature": 0.0,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    calls, content, chunks = {}, "", 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunks += 1
            delta = json.loads(line[5:])["choices"][0].get("delta", {})
            content += delta.get("content") or ""
            for item in delta.get("tool_calls") or []:
                slot = calls.setdefault(item.get("index", 0), {"name": "", "arguments": ""})
                function = item.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    result = []
    for slot in calls.values():
        try:
            arguments = json.loads(slot["arguments"] or "{}")
        except ValueError:
            arguments = {"_unparsable": slot["arguments"]}
        result.append({"name": slot["name"], "arguments": arguments})
    return {"calls": result, "content": content, "chunks": chunks}


def suite_tools(args):
    cases = []

    def record(name, passed, detail):
        cases.append({"name": name, "passed": bool(passed), **detail})

    user = [{"role": "user", "content": "서울의 현재 날씨를 알려줘."}]
    reply = chat_request(args.base, user, tools=TOOLS)
    calls = parsed_calls(reply["tool_calls"])
    record("single_ko", len(calls) == 1 and calls[0]["name"] == "get_weather"
           and mentions(calls[0]["arguments"], "서울", "seoul") and "<tool_call>" not in reply["content"],
           {"calls": calls, "content": reply["content"][:200], "finish_reason": reply["finish_reason"]})
    first_call = reply["tool_calls"][:1]

    reply = chat_request(args.base, [{"role": "user", "content": "서울과 부산의 현재 날씨를 각각 알려줘."}], tools=TOOLS)
    calls = parsed_calls(reply["tool_calls"])
    cities = [c["arguments"] for c in calls if c["name"] == "get_weather"]
    record("parallel_ko", len(cities) >= 2 and any(mentions(c, "서울", "seoul") for c in cities)
           and any(mentions(c, "부산", "busan") for c in cities),
           {"calls": calls, "content": reply["content"][:200]})

    reply = chat_request(args.base, [{"role": "user", "content":
                                      "Find flights from Seoul to Tokyo on 2026-11-20."}], tools=TOOLS)
    calls = parsed_calls(reply["tool_calls"])
    record("multi_argument_en", len(calls) >= 1 and calls[0]["name"] == "search_flights"
           and mentions(calls[0]["arguments"], "seoul", "icn", "gmp", "서울")
           and mentions(calls[0]["arguments"], "tokyo", "nrt", "hnd", "도쿄")
           and mentions(calls[0]["arguments"], "2026-11-20"),
           {"calls": calls, "content": reply["content"][:200]})

    reply = chat_request(args.base, user, tools=TOOLS, thinking=True, max_tokens=2048)
    calls = parsed_calls(reply["tool_calls"])
    record("thinking_plus_tool", len(calls) >= 1 and calls[0]["name"] == "get_weather"
           and "<tool_call>" not in reply["content"] and "</think>" not in reply["content"],
           {"calls": calls, "content": reply["content"][:200], "reasoning_chars": len(reply["reasoning"]),
            "reasoning_head": reply["reasoning"][:200]})

    reply = chat_request(args.base, [{"role": "user", "content": "대한민국의 수도는 어디인가요? 도시 이름만 답하세요."}],
                         tools=TOOLS)
    record("no_tool_needed", not reply["tool_calls"] and mentions(reply["content"], "서울", "seoul"),
           {"content": reply["content"][:200], "tool_calls": reply["tool_calls"]})

    if first_call:
        call_id = first_call[0].get("id") or "call_0"
        messages = user + [
            {"role": "assistant", "content": None, "tool_calls": first_call},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps({"city": "서울", "temperature_c": 21, "condition": "맑음"}, ensure_ascii=False)},
        ]
        reply = chat_request(args.base, messages, tools=TOOLS)
        record("multi_turn_tool_result", not reply["tool_calls"] and "21" in reply["content"],
               {"content": reply["content"][:300]})
    else:
        record("multi_turn_tool_result", False, {"skipped": "no first tool call"})

    streamed = stream_tool_call(args.base, user)
    record("streaming", len(streamed["calls"]) == 1 and streamed["calls"][0]["name"] == "get_weather"
           and mentions(streamed["calls"][0]["arguments"], "서울", "seoul") and "<tool_call>" not in streamed["content"],
           streamed)

    reply = chat_request(args.base, [{"role": "user", "content": "부산 날씨는 어때?"}], tools=TOOLS,
                         tool_choice={"type": "function", "function": {"name": "get_weather"}})
    calls = parsed_calls(reply["tool_calls"])
    record("named_tool_choice", len(calls) == 1 and calls[0]["name"] == "get_weather"
           and mentions(calls[0]["arguments"], "부산", "busan"), {"calls": calls})

    reply = chat_request(args.base, [{"role": "user", "content": "인천에서 오사카로 2026-12-01에 가는 항공편을 찾아줘."}],
                         tools=TOOLS, tool_choice="required")
    calls = parsed_calls(reply["tool_calls"])
    record("required_tool_choice", len(calls) >= 1 and calls[0]["name"] == "search_flights",
           {"calls": calls})
    return {"server_flags": "--tool-call-parser hermes --enable-auto-tool-choice --reasoning-parser deepseek_v3",
            "passed": sum(c["passed"] for c in cases), "total": len(cases), "cases": cases}


def count_chat(base, content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    return call(base, "/tokenize", body, 600)[0]["count"]


def count_text(base, text):
    return call(base, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False}, 600)[0]["count"]


def build_niah(base, target, depth, code):
    needle = NEEDLE_TMPL.format(code=code)
    overhead = count_chat(base, needle + QUESTION)
    unit = count_text(base, FILLER_UNIT)
    units = max(1, (target - overhead) // unit)
    body, n = None, 0
    for _ in range(12):
        head = int(units * depth)
        body = FILLER_UNIT * head + needle + FILLER_UNIT * (units - head) + QUESTION
        n = count_chat(base, body)
        if n <= target:
            break
        units -= max(1, int((n - target) / unit) + 1)
    return body, n


def suite_niah(args):
    lengths = [int(x) for x in args.niah_lengths.split(",")]
    depths = [0.15, 0.5, 0.85]
    cases = []
    for length in lengths:
        target = min(length, args.max_model_len - 48)
        for index, depth in enumerate(depths):
            code = f"KX-{length // 1024}K-{index}{index}{index}7"
            content, tokens = build_niah(args.base, target, depth, code)
            began = time.time()
            try:
                reply = chat_request(args.base, [{"role": "user", "content": content}], max_tokens=32, timeout=3600)
                text = reply["content"].strip()
                cases.append({"length": length, "depth": depth, "prompt_tokens": reply["usage"]["prompt_tokens"],
                              "built_tokens": tokens, "code": code, "hit": code in text, "answer": text[:80],
                              "seconds": round(time.time() - began, 1)})
            except Exception as exc:
                cases.append({"length": length, "depth": depth, "built_tokens": tokens, "code": code, "hit": False,
                              "error": str(exc)[:400], "seconds": round(time.time() - began, 1)})
            print(json.dumps(cases[-1], ensure_ascii=False), flush=True)
    by_length = {}
    for case in cases:
        entry = by_length.setdefault(str(case["length"]), {"hits": 0, "total": 0})
        entry["hits"] += case["hit"]
        entry["total"] += 1
    return {"method": "SKT-AI/A.X-K2 examples/vllm/niah_test.py (same needle, filler, question, depths, codes); "
                      "one request at a time, temperature 0, 32 output tokens, thinking off",
            "hits": sum(c["hit"] for c in cases), "total": len(cases), "by_length": by_length, "cases": cases}


DET_PROMPT = [{"role": "user", "content": "서울의 한강을 세 문장으로 설명한 뒤, 같은 내용을 영어 세 문장으로 다시 써 주세요."}]
OTHER_PROMPTS = [f"{topic}에 대해 다섯 문장으로 설명해 주세요." for topic in
                 ("부산 해운대", "제주도의 날씨", "경주 불국사", "인공지능의 역사", "반도체 공정", "축구의 규칙",
                  "커피의 원산지", "한국의 전통 음식", "태양계 행성", "광합성", "블록체인", "고래의 생태",
                  "피아노의 역사", "기후 변화", "올림픽의 기원")]


def generate(base, messages, salt, max_tokens=200):
    reply = chat_request(base, messages, max_tokens=max_tokens, extra={"cache_salt": salt, "seed": 0})
    return reply["content"]


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def suite_determinism(args):
    run = f"{int(time.time())}"
    alone_1 = generate(args.base, DET_PROMPT, f"fresh-a-{run}")
    alone_2 = generate(args.base, DET_PROMPT, f"fresh-b-{run}")
    cached = generate(args.base, DET_PROMPT, f"fresh-a-{run}")

    def batched(salt):
        with futures.ThreadPoolExecutor(16) as pool:
            target = pool.submit(generate, args.base, DET_PROMPT, salt)
            others = [pool.submit(generate, args.base, [{"role": "user", "content": p}], f"{salt}-{i}")
                      for i, p in enumerate(OTHER_PROMPTS)]
            for other in others:
                other.result()
            return target.result()

    batch_1 = batched(f"batch-a-{run}")
    batch_2 = batched(f"batch-b-{run}")
    runs = {"alone_fresh_1": alone_1, "alone_fresh_2": alone_2, "alone_prefix_cache_hit": cached,
            "in_batch_of_16_1": batch_1, "in_batch_of_16_2": batch_2}
    comparison = {name: {"identical_to_alone_fresh_1": text == alone_1,
                         "first_difference_char": first_difference(alone_1, text)}
                  for name, text in runs.items() if name != "alone_fresh_1"}
    return {"batch_invariant_mode": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
            "method": "greedy (temperature 0), 200 tokens; cache_salt isolates or shares the prefix cache",
            "all_identical": all(v["identical_to_alone_fresh_1"] for v in comparison.values()),
            "comparison": comparison, "outputs": {k: v[:600] for k, v in runs.items()}}


class MetricsSampler(threading.Thread):
    NAMES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")

    def __init__(self, base, interval=2.0):
        super().__init__(daemon=True)
        self.base, self.interval, self.done = base, interval, threading.Event()
        self.peak = {name: 0.0 for name in self.NAMES}
        self.samples = 0

    def run(self):
        while not self.done.is_set():
            try:
                text = urllib.request.urlopen(self.base + "/metrics", timeout=5).read().decode()
                totals = {name: 0.0 for name in self.NAMES}
                for line in text.splitlines():
                    for name in self.NAMES:
                        if line.startswith(name + "{") or line.startswith(name + " "):
                            totals[name] += float(line.rsplit(" ", 1)[1])
                for name, value in totals.items():
                    self.peak[name] = max(self.peak[name], value)
                self.samples += 1
            except Exception:
                pass
            self.done.wait(self.interval)


BENCH_KEYS = ["completed", "duration", "total_input_tokens", "total_output_tokens", "request_throughput",
              "output_throughput", "total_token_throughput", "max_output_tokens_per_s", "max_concurrent_requests",
              "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
              "p90_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms", "median_e2el_ms"]


def bench(args, name, concurrency, prompts, input_len, output_len, timeout, seed=2026):
    filename = f"bench-{args.tag}-{name}.json"
    command = ["vllm", "bench", "serve", "--backend", "openai", "--base-url", args.base,
               "--endpoint", "/v1/completions", "--model", MODEL, "--tokenizer", args.tokenizer,
               "--dataset-name", "random", "--random-input-len", str(input_len),
               "--random-output-len", str(output_len), "--num-prompts", str(prompts),
               "--max-concurrency", str(concurrency), "--ignore-eos", "--seed", str(seed),
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
               "--save-result", "--result-dir", args.out_dir, "--result-filename", filename]
    sampler = MetricsSampler(args.base)
    sampler.start()
    began = time.time()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
        returncode, stderr = run.returncode, run.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, stderr = -1, f"timeout after {timeout}s: {exc}"
    sampler.done.set()
    sampler.join(10)
    record = {"name": name, "concurrency": concurrency, "num_prompts": prompts, "input_len": input_len,
              "output_len": output_len, "seed": seed, "returncode": returncode,
              "wall_seconds": round(time.time() - began, 1),
              "peak_running_requests": sampler.peak["vllm:num_requests_running"],
              "peak_waiting_requests": sampler.peak["vllm:num_requests_waiting"],
              "peak_kv_cache_usage": round(sampler.peak["vllm:kv_cache_usage_perc"], 4)}
    path = Path(args.out_dir) / filename
    if returncode == 0 and path.exists():
        data = json.loads(path.read_text())
        record.update({k: data.get(k) for k in BENCH_KEYS})
    else:
        record["stderr_tail"] = stderr[-1500:]
    print(json.dumps(record), flush=True)
    return record


def sweep(args, isls, prompts_for, suite):
    points = []
    for isl in isls:
        prompts = prompts_for(isl)
        timeout = 1800 if isl <= 16384 else 3600 if isl <= 32768 else 5400
        record = bench(args, f"isl{isl}", 32, prompts, isl, 1024, timeout, seed=2026 + isl)
        doc = DOC_FIG7.get(isl)
        if doc and record.get("total_token_throughput"):
            record["doc_fig7_b200_total_tok_s"], record["doc_fig7_b200_output_tok_s"] = doc
            record["ratio_total_vs_doc"] = round(record["total_token_throughput"] / doc[0], 3)
            record["ratio_output_vs_doc"] = round(record["output_throughput"] / doc[1], 3)
        if DOC_FIG8_TOTAL.get(isl):
            record["doc_fig8_b200_eagle3_total_tok_s"] = DOC_FIG8_TOTAL[isl]
        points.append(record)
        emit(args.out_dir, args.tag, f"{suite}-isl{isl}", record)
    batched = args.max_num_batched_tokens
    note = ("(vLLM's B200 default); same conditions as the tech report Fig. 7 (B200 single node)"
            if batched == 8192 else "(the tech report's B200 run used 8192)")
    return {"method": "vllm bench serve --dataset-name random --ignore-eos, concurrency 32, output 1024 tokens, "
                      f"a seed per point; server --max-num-batched-tokens {batched} {note}",
            "points": points}


def suite_docsweep(args):
    isls = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 120000]
    return sweep(args, isls, lambda isl: 64 if isl <= 16384 else 32, "docsweep")


def suite_docsweep32k(args):
    isls = [1024, 2048, 4096, 8192, 16384, 32768]
    return sweep(args, isls, lambda isl: 64 if isl <= 16384 else 32, "docsweep32k")


def suite_sweepsub(args):
    return sweep(args, [1024, 4096, 8192], lambda isl: 64, "sweepsub")


def suite_latency(args):
    points = []
    for index, (name, concurrency, prompts, input_len, output_len) in enumerate(benchmark_plan(args.max_model_len)):
        record = bench(args, name, concurrency, prompts, input_len, output_len, 3600, seed=2027 + index)
        points.append(record)
        emit(args.out_dir, args.tag, f"bench-{name}", record)
    return {"method": "client_tests.benchmark_plan, the plan of the earlier 16-GPU FP8 table: vllm bench serve "
                      "--dataset-name random --ignore-eos, a seed per point, 256 output tokens; server "
                      f"--max-num-batched-tokens {args.max_num_batched_tokens}",
            "points": points}


EPMATRIX_INPUTS = (1024, 8192, 32768)
EPMATRIX_OUTPUT = 1024


def epmatrix_plan():
    """(name, concurrency, prompts, input tokens, seed, timeout seconds) of the expert-parallel comparison.
    Each point has its own seed, so no point reuses another point's prompts from the prefix cache."""
    plan = []
    for isl in EPMATRIX_INPUTS:
        for concurrency in (1, 32):
            prompts = 4 if concurrency == 1 else 64 if isl <= 8192 else 32
            timeout = 1200 if concurrency == 1 or isl <= 1024 else 1800 if isl <= 8192 else 2700
            plan.append((f"isl{isl}-c{concurrency}", concurrency, prompts, isl, 2026 + isl // 1024 * 10 + concurrency,
                         timeout))
    return plan


def server_healthy(base, wait=90):
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/health", timeout=10)
            return True
        except Exception:
            time.sleep(5)
    return False


def suite_epmatrix(args):
    """The same matrix for every layout: inputs 1K/8K/32K x concurrency 1 and 32, 1,024 output tokens.
    Before the points of an input length, two discarded requests of that length (own seed) warm up
    kernels whose shapes depend on it. A point that fails while the server stops answering ends the suite."""
    points, warmups = [], []
    for isl in EPMATRIX_INPUTS:
        warm = bench(args, f"warmup-isl{isl}", 2, 2, isl, 64, 900, seed=7000 + isl // 1024)
        warmups.append({k: warm.get(k) for k in ("name", "returncode", "wall_seconds")})
        for name, concurrency, prompts, input_len, seed, timeout in epmatrix_plan():
            if input_len != isl:
                continue
            record = bench(args, name, concurrency, prompts, input_len, EPMATRIX_OUTPUT, timeout, seed=seed)
            points.append(record)
            emit(args.out_dir, args.tag, f"epmatrix-{name}", record)
            if record["returncode"] != 0 and not server_healthy(args.base):
                return {"method": EPMATRIX_METHOD, "points": points, "warmups": warmups,
                        "aborted": f"the server stopped answering /health after {name}"}
    return {"method": EPMATRIX_METHOD, "points": points, "warmups": warmups}


EPMATRIX_METHOD = ("vllm bench serve --dataset-name random --ignore-eos, the model's default sampling, inputs "
                   "1K/8K/32K x max concurrency 1 (4 prompts) and 32 (64 prompts, 32 at 32K), 1,024 output tokens, "
                   "a seed per point; server --max-num-batched-tokens 8192")
EPMATRIX_SUMMARY_KEYS = ("name", "returncode", "completed", "request_throughput", "output_throughput",
                         "total_token_throughput", "median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms",
                         "median_itl_ms", "peak_running_requests", "peak_waiting_requests", "peak_kv_cache_usage")


def suite_functional(args):
    checks = [("ko_capital", "대한민국의 수도는 어디인가요? 도시 이름만 답하세요.", "서울"),
              ("en_multiply", "What is 17 multiplied by 23? Reply with only the number.", "391"),
              ("ko_translate", "다음 문장을 영어로 번역하세요: 나는 오늘 아침에 커피를 마셨다.", "coffee"),
              ("code_is_prime", "Write a Python function named is_prime(n). Return only the code.", "def is_prime")]
    results = []
    for name, question, expected in checks:
        reply = chat_request(args.base, [{"role": "user", "content": question}], max_tokens=320)
        results.append({"name": name, "passed": expected in reply["content"], "content": reply["content"][:300]})
    return {"passed": sum(r["passed"] for r in results), "total": len(results), "checks": results}


def spec_counters(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    totals = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            totals[name] = totals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return totals


def suite_specstats(args):
    counters = spec_counters(args.base)
    drafts = counters.get("vllm:spec_decode_num_drafts_total", 0.0)
    accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    return {"counters": counters,
            "mean_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
            "draft_acceptance_rate": round(accepted / drafted, 3) if drafted else None,
            "doc_reference": "tech report: 2.24 tokens committed per step on average (B200, random dataset)"}


SUITES = {"tools": suite_tools, "niah": suite_niah, "determinism": suite_determinism,
          "docsweep": suite_docsweep, "docsweep32k": suite_docsweep32k, "sweepsub": suite_sweepsub,
          "latency": suite_latency, "functional": suite_functional, "specstats": suite_specstats,
          "epmatrix": suite_epmatrix}
SUMMARY_KEYS = ("name", "returncode", "output_throughput", "total_token_throughput", "median_ttft_ms",
                "median_tpot_ms", "ratio_total_vs_doc", "peak_running_requests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--suites", required=True, help="comma-separated: " + ",".join(SUITES))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="the server's setting, recorded only")
    parser.add_argument("--niah-lengths", default="32768,131072,262144")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    status = 0
    for name in args.suites.split(","):
        began = time.time()
        try:
            data = SUITES[name](args)
            data["suite_seconds"] = round(time.time() - began, 1)
            if "points" not in data:
                emit(args.out_dir, args.tag, name, data)
            else:
                keys = EPMATRIX_SUMMARY_KEYS if name == "epmatrix" else SUMMARY_KEYS
                summary = [{k: p.get(k) for k in keys} for p in data["points"]]
                emit(args.out_dir, args.tag, name, {**{k: v for k, v in data.items() if k != "points"},
                                                    "summary": summary})
            if data.get("aborted"):
                status = 1
                break
        except Exception as exc:
            status = 1
            emit(args.out_dir, args.tag, f"{name}-error", {"error": f"{type(exc).__name__}: {exc}"[:2000],
                                                           "seconds": round(time.time() - began, 1)})
    return status


if __name__ == "__main__":
    sys.exit(main())
