"""Client checks against the OpenAI-compatible endpoint of the multi-node vLLM deployment.

smoke  : 2-layer real-weight cut. Prompt log-probabilities for fixed token ids (compared against
         a single-GPU run of the same server and against the official Transformers reference),
         determinism and concurrent requests.
full   : complete 61-layer model. Functional chat checks (Korean/English, reasoning mode),
         arithmetic accuracy, needle retrieval inside and beyond index_topk, determinism,
         streaming latency and `vllm bench serve` load tests.
compare: summarise the difference between two smoke result files.
"""
import argparse
import concurrent.futures as futures
import json
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "axk2"


def call(base, path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    began = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.time() - began


def prompt_logprobs(base, ids, max_tokens=8):
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": 5, "prompt_logprobs": 20, "return_tokens_as_token_ids": True}
    payload, seconds = call(base, "/v1/completions", body)
    choice = payload["choices"][0]
    entries = choice.get("prompt_logprobs") or []
    chosen, top1 = [], []
    for position in range(1, len(ids)):
        entry = entries[position] or {}
        token = str(ids[position])
        chosen.append(round(entry[token]["logprob"], 5) if token in entry else None)
        best = min(entry.items(), key=lambda kv: kv[1].get("rank", 1 << 30)) if entry else None
        top1.append(int(best[0]) if best else None)
    generated = [int(t.split(":", 1)[1]) for t in choice["logprobs"]["tokens"]]
    return {"chosen_logprob": chosen, "top1": top1, "generated": generated,
            "seconds": round(seconds, 3), "usage": payload.get("usage")}


def chat(base, messages, max_tokens=256, thinking=False, temperature=0.0, timeout=3600):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking}}
    payload, seconds = call(base, "/v1/chat/completions", body, timeout)
    message = payload["choices"][0]["message"]
    return {"content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "usage": payload.get("usage"), "seconds": round(seconds, 2)}


def run_smoke(args):
    prompts = json.loads(Path(args.prompts).read_text())
    result = {"suite": "smoke", "tag": args.tag, "models": call(args.base, "/v1/models")[0], "prompts": {}}
    for prompt in prompts:
        result["prompts"][prompt["name"]] = {"num_tokens": prompt["num_tokens"],
                                             **prompt_logprobs(args.base, prompt["token_ids"])}
    repeat = {}
    for name in ("ko_short", "mixed_2600"):
        ids = next(p["token_ids"] for p in prompts if p["name"] == name)
        again = prompt_logprobs(args.base, ids)
        first = result["prompts"][name]
        diffs = [abs(a - b) for a, b in zip(first["chosen_logprob"], again["chosen_logprob"])
                 if a is not None and b is not None]
        repeat[name] = {"max_abs_diff": max(diffs) if diffs else None,
                        "generated_equal": first["generated"] == again["generated"]}
    result["determinism"] = repeat
    texts = [f"Request {i}: summarise why pipeline parallelism helps large models." for i in range(8)]
    with futures.ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda t: call(args.base, "/v1/completions",
                                               {"model": MODEL, "prompt": t, "max_tokens": 32,
                                                "temperature": 0.0}), texts))
    result["concurrency"] = {"requests": len(replies),
                             "completed": sum(1 for r, _ in replies if r["choices"][0]["text"] is not None),
                             "max_seconds": round(max(s for _, s in replies), 2)}
    return result


def needle_prompt(rng, words, code, position=0.4):
    filler = []
    sentences = ["The river flows through the old city and past the market square.",
                 "서울의 오래된 골목에는 작은 서점과 찻집이 많이 있습니다.",
                 "Engineers measured the bridge again before the winter season.",
                 "도서관은 주말에도 늦은 시간까지 문을 엽니다.",
                 "A light rain fell over the harbour while the ships waited."]
    while len(filler) < words:
        filler.extend(rng.choice(sentences).split())
    filler = filler[:words]
    filler.insert(int(len(filler) * position), f"The secret code is {code}.")
    return " ".join(filler)


BENCH_KEYS = ["completed", "request_throughput", "output_throughput", "total_token_throughput",
              "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
              "p99_tpot_ms", "median_itl_ms", "p99_itl_ms", "median_e2el_ms"]


def bench(args, name, concurrency, prompts, input_len, output_len, out_dir, seed=2026):
    filename = f"bench-{name}.json"
    command = ["vllm", "bench", "serve", "--backend", "openai", "--base-url", args.base,
               "--endpoint", "/v1/completions", "--model", MODEL, "--tokenizer", args.tokenizer,
               "--dataset-name", "random", "--random-input-len", str(input_len),
               "--random-output-len", str(output_len), "--num-prompts", str(prompts),
               "--max-concurrency", str(concurrency), "--ignore-eos", "--seed", str(seed),
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
               "--save-result", "--result-dir", str(out_dir), "--result-filename", filename]
    began = time.time()
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        returncode, stderr = run.returncode, run.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, stderr = -1, f"timeout: {exc}"
    record = {"name": name, "concurrency": concurrency, "num_prompts": prompts, "input_len": input_len,
              "output_len": output_len, "seed": seed, "returncode": returncode,
              "wall_seconds": round(time.time() - began, 1)}
    path = Path(out_dir) / filename
    if returncode == 0 and path.exists():
        data = json.loads(path.read_text())
        record.update({k: data.get(k) for k in BENCH_KEYS})
    else:
        record["stderr_tail"] = stderr[-1500:]
    print(json.dumps(record), flush=True)
    return record


def benchmark_plan(max_model_len):
    plan = [(f"latency-in{n}", 1, 3, n, 256) for n in (1024, 8192, 32768) if n + 256 <= max_model_len]
    plan += [(f"throughput-c{c}", c, min(max(8, 3 * c), 384), 1024, 256) for c in (1, 8, 32, 64, 128)]
    if 16384 + 256 <= max_model_len:
        plan += [("long16k-c4", 4, 8, 16384, 256), ("long16k-c16", 16, 32, 16384, 256)]
    return plan


def run_full(args):
    out_dir = Path(args.out).parent
    result = {"suite": "full", "tag": args.tag, "models": call(args.base, "/v1/models")[0]}
    checks = [
        ("ko_capital", "대한민국의 수도는 어디인가요? 도시 이름만 답하세요.", ["서울"]),
        ("en_multiply", "What is 17 multiplied by 23? Reply with only the number.", ["391"]),
        ("ko_fruit", "사과 3개와 배 5개가 있습니다. 배 2개를 먹으면 남은 과일은 모두 몇 개인가요? 숫자만 답하세요.", ["6"]),
        ("en_red_planet", "Which planet is known as the Red Planet? Answer with one word.", ["Mars"]),
        ("ko_translate", "다음 문장을 영어로 번역하세요: 나는 오늘 아침에 커피를 마셨다.", ["coffee"]),
        ("code_is_prime", "Write a Python function named is_prime(n) that returns True when n is prime. "
                          "Return only the code.", ["def is_prime"]),
    ]
    functional = []
    for name, question, expected in checks:
        reply = chat(args.base, [{"role": "user", "content": question}], max_tokens=320)
        functional.append({"name": name, "passed": all(e in reply["content"] for e in expected),
                           "content": reply["content"][:400], "seconds": reply["seconds"],
                           "finish_reason": reply["finish_reason"]})
    result["functional"] = functional
    reasoning = chat(args.base, [{"role": "user", "content":
                                  "A train travels 120 km in 1.5 hours. What is its average speed in km/h? "
                                  "Give the final answer as a number."}], max_tokens=4096, thinking=True)
    result["reasoning_mode"] = {"passed": "80" in reasoning["content"], "content": reasoning["content"][:400],
                                "reasoning_chars": len(reasoning["reasoning"]),
                                "reasoning_head": reasoning["reasoning"][:300],
                                "finish_reason": reasoning["finish_reason"], "usage": reasoning["usage"],
                                "seconds": reasoning["seconds"]}
    rng = random.Random(2026)
    arithmetic = []
    for _ in range(20):
        a, b, c = rng.randint(10, 99), rng.randint(2, 9), rng.randint(10, 99)
        reply = chat(args.base, [{"role": "user", "content":
                                  f"Compute {a} * {b} + {c}. Reply with only the final integer."}], max_tokens=64)
        numbers = re.findall(r"-?\d+", reply["content"].replace(",", ""))
        arithmetic.append({"question": f"{a} * {b} + {c}", "expected": a * b + c, "content": reply["content"][:120],
                           "finish_reason": reply["finish_reason"],
                           "correct": bool(numbers) and int(numbers[-1]) == a * b + c})
    result["arithmetic"] = {"correct": sum(x["correct"] for x in arithmetic), "total": len(arithmetic),
                            "items": arithmetic}
    question = [{"role": "user", "content": "Describe the Han River in Seoul in three sentences."}]
    first, second = chat(args.base, question, max_tokens=160), chat(args.base, question, max_tokens=160)
    result["determinism"] = {"identical": first["content"] == second["content"], "sample": first["content"][:300]}
    needles = []
    for target in (1400, 8000, 30000, 60000):
        if target + 300 > args.max_model_len:
            continue
        for depth in (0.1, 0.5, 0.9):
            code = f"BLUE-{rng.randint(1000, 9999)}"
            text = needle_prompt(rng, int(target / 1.22), code, depth)
            began = time.time()
            try:
                reply = chat(args.base, [{"role": "user", "content": text + "\n\nWhat is the secret code mentioned "
                                          "above? Answer with the code only."}], max_tokens=24)
                needles.append({"target_tokens": target, "depth": depth,
                                "prompt_tokens": reply["usage"]["prompt_tokens"], "found": code in reply["content"],
                                "content": reply["content"][:80], "seconds": round(time.time() - began, 2)})
            except Exception as exc:
                needles.append({"target_tokens": target, "depth": depth, "error": str(exc)[:300]})
    result["needle"] = needles
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Write a short story about a lighthouse keeper."}],
            "max_tokens": 256, "temperature": 0.0, "stream": True, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    began, first_token, pieces = time.time(), None, 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"]
            if delta.get("content"):
                pieces += 1
                first_token = first_token or time.time()
    total = time.time() - began
    result["streaming_single_user"] = {
        "ttft_seconds": round(first_token - began, 3) if first_token else None,
        "chunks": pieces, "total_seconds": round(total, 2),
        "decode_chunks_per_second": round((pieces - 1) / (total - (first_token - began)), 2)
        if first_token and pieces > 1 else None}
    result["benchmarks"] = [bench(args, *spec, out_dir, seed=2027 + index)
                            for index, spec in enumerate(benchmark_plan(args.max_model_len))]
    found = [n for n in needles if "found" in n]
    result["summary"] = {
        "functional_passed": sum(f["passed"] for f in functional), "functional_total": len(functional),
        "reasoning_passed": result["reasoning_mode"]["passed"],
        "arithmetic": f"{result['arithmetic']['correct']}/{result['arithmetic']['total']}",
        "deterministic": result["determinism"]["identical"],
        "needle_found": f"{sum(n['found'] for n in found)}/{len(needles)}",
        "speed": {b["name"]: {k: (round(b[k], 1) if isinstance(b.get(k), float) else b.get(k))
                              for k in ("output_throughput", "median_ttft_ms", "median_tpot_ms")}
                  for b in result["benchmarks"]}}
    return result


def run_compare(args):
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    summary = {}
    for name, left in a["prompts"].items():
        right = b["prompts"][name]
        pairs = [(x, y) for x, y in zip(left["chosen_logprob"], right["chosen_logprob"])
                 if x is not None and y is not None]
        diffs = [abs(x - y) for x, y in pairs]
        agree = [x == y for x, y in zip(left["top1"], right["top1"]) if x is not None and y is not None]
        summary[name] = {"positions": len(pairs), "max_abs_diff": max(diffs) if diffs else None,
                         "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                         "top1_agreement": sum(agree) / len(agree) if agree else None,
                         "generated_equal": left["generated"] == right["generated"]}
    return {"a": a["tag"], "b": b["tag"], "per_prompt": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["smoke", "full", "compare"])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--tag", default="")
    parser.add_argument("--prompts")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--out", required=True)
    parser.add_argument("--a")
    parser.add_argument("--b")
    args = parser.parse_args()
    runner = {"smoke": run_smoke, "full": run_full, "compare": run_compare}[args.suite]
    result = runner(args)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result.get("summary", result.get("determinism", "")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
