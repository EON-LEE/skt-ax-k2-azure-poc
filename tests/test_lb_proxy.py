import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aml" / "src" / "bench"))
import lb_proxy  # noqa: E402
sys.path.pop(0)


class FakeBackend:
    """A tiny HTTP/1.1 server shaped like vLLM: /health, /metrics, /reset_prefix_cache and streamed completions.
    role "prefill" answers kv_transfer_params requests like a NixlConnector kv_producer (JSON, no stream)."""

    def __init__(self, name, healthy=True, delay=0.2, chunks=4, role=None, prefill_status=200, transfer=True):
        self.name, self.healthy, self.delay, self.chunks = name, healthy, delay, chunks
        self.role, self.prefill_status, self.transfer = role, prefill_status, transfer
        self.served, self.resets, self.in_flight, self.peak = 0, 0, 0, 0
        self.bodies = []

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def handle(self, reader, writer):
        start_line, headers = await lb_proxy.read_head(reader)
        body = await lb_proxy.read_body(reader, headers)
        method, path = start_line.split(" ")[:2]

        async def plain(status, body):
            writer.write(f"HTTP/1.1 {status} X\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()

        payload = json.loads(body) if body.startswith(b"{") else {}
        if path == "/health":
            await plain(200 if self.healthy else 503, b"")
        elif path == "/metrics":
            await plain(200, b'# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{engine="0",model_name="axk2"} 1.0\n')
        elif path == "/reset_prefix_cache":
            self.resets += 1
            await plain(200, b"")
        elif self.role == "prefill" and (payload.get("kv_transfer_params") or payload.get("prompt_logprobs") or path == "/tokenize"):
            self.served += 1
            self.bodies.append(payload)
            answer = {"choices": [{"text": "x"}], "kv_transfer_params": None}
            if payload.get("kv_transfer_params") and self.transfer:
                answer["kv_transfer_params"] = {"do_remote_prefill": True, "do_remote_decode": False, "remote_block_ids": [[1, 2]],
                                                "remote_engine_id": "engine-p", "remote_request_id": "req-1",
                                                "remote_host": "node0", "remote_port": 5600, "tp_size": 8}
            await plain(self.prefill_status, json.dumps(answer).encode())
        elif payload.get("stream") is False:
            self.served += 1
            self.bodies.append(payload)
            await plain(200, json.dumps({"choices": [{"text": f"from {self.name}"}]}).encode())
        else:
            self.served += 1
            self.bodies.append(payload)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
            for index in range(self.chunks):
                data = f"data: {json.dumps({'backend': self.name, 'index': index})}\n\n".encode()
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()
                await asyncio.sleep(self.delay)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            self.in_flight -= 1
        writer.close()


async def fetch(port, method, path, body=b"", stamp=False):
    """(status, body, seconds to the first body byte) of one request through the proxy."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    began = time.monotonic()
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    await writer.drain()
    status, _ = await lb_proxy.read_head(reader)
    first = await reader.read(1)
    first_at = time.monotonic() - began
    rest = await reader.read()
    writer.close()
    return int(status.split(" ")[1]), first + rest, first_at


class ProxyTests(unittest.TestCase):
    def run_with(self, backends, scenario, make=None):
        async def main():
            for backend in backends:
                await backend.start()
            balancer = (make or (lambda urls: lb_proxy.Balancer(urls, health_interval=0.1)))([b.url for b in backends])
            await asyncio.gather(*(balancer.check(b) for b in balancer.backends))
            server = await asyncio.start_server(balancer.handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                return await scenario(port, balancer)
            finally:
                server.close()
                for backend in backends:
                    backend.server.close()
        return asyncio.run(main())

    def test_least_outstanding_routing_and_streaming(self):
        backends = [FakeBackend("a"), FakeBackend("b")]

        async def scenario(port, balancer):
            results = await asyncio.gather(*(fetch(port, "POST", "/v1/completions", b"{}") for _ in range(4)))
            stats = json.loads((await fetch(port, "GET", "/lb/stats"))[1])
            return results, stats

        results, stats = self.run_with(backends, scenario)
        self.assertEqual([b.served for b in backends], [2, 2])
        self.assertEqual([b.peak for b in backends], [2, 2], "concurrent requests spread over both backends")
        for status, body, first_at in results:
            self.assertEqual(status, 200)
            self.assertEqual(body.count(b"data: "), 4)
            self.assertLess(first_at, 0.5, "the first event arrives before the backend finishes (0.8 s)")
        self.assertEqual([b["routed"] for b in stats["backends"]], [2, 2])
        self.assertEqual([b["in_flight"] for b in stats["backends"]], [0, 0])

    def test_metrics_reset_and_health(self):
        backends = [FakeBackend("a"), FakeBackend("b")]

        async def scenario(port, balancer):
            metrics = (await fetch(port, "GET", "/metrics"))[1].decode()
            reset = await fetch(port, "POST", "/reset_prefix_cache")
            healthy = (await fetch(port, "GET", "/health"))[0]
            backends[1].healthy = False
            unhealthy = (await fetch(port, "GET", "/health"))[0]
            routed = [await fetch(port, "POST", "/v1/completions", b"{}") for _ in range(2)]
            return metrics, reset, healthy, unhealthy, routed

        metrics, reset, healthy, unhealthy, routed = self.run_with(backends, scenario)
        self.assertIn('vllm:num_requests_running{replica="0",engine="0",model_name="axk2"} 1.0', metrics)
        self.assertIn('vllm:num_requests_running{replica="1",engine="0",model_name="axk2"} 1.0', metrics)
        self.assertEqual(metrics.count("# TYPE"), 1)
        self.assertEqual(reset[0], 200)
        self.assertEqual([b.resets for b in backends], [1, 1])
        self.assertEqual((healthy, unhealthy), (200, 503))
        self.assertEqual([b.served for b in backends], [2, 0], "an unhealthy backend gets no requests")
        self.assertTrue(all(status == 200 for status, _, _ in routed))

    def test_label_metrics(self):
        text = lb_proxy.label_metrics('# HELP x y\nvllm:a{engine="0"} 1\nvllm:b 2\n', 3)
        self.assertEqual(text.splitlines(), ["# HELP x y", 'vllm:a{replica="3",engine="0"} 1', 'vllm:b{replica="3"} 2'])
        self.assertNotIn("#", lb_proxy.label_metrics("# HELP x y\nvllm:b 2", 1, keep_comments=False))


class DisaggregatedTests(unittest.TestCase):
    REQUEST = {"model": "axk2", "prompt": [1, 2, 3], "max_tokens": 64, "temperature": 0.6, "stream": True,
               "stream_options": {"include_usage": True}, "ignore_eos": True, "min_tokens": 8}

    def run_pd(self, prefill, decode, scenario):
        make = lambda urls: lb_proxy.Disaggregated(urls[0], urls[1], health_interval=0.1)
        return ProxyTests.run_with(self, [prefill, decode], scenario, make)

    def test_prefill_then_decode_with_the_returned_kv_transfer_params(self):
        prefill, decode = FakeBackend("p", role="prefill"), FakeBackend("d", delay=0.2)

        async def scenario(port, balancer):
            streamed = await fetch(port, "POST", "/v1/completions", json.dumps(self.REQUEST).encode())
            logprobs = await fetch(port, "POST", "/v1/completions", b'{"prompt": [1, 2], "prompt_logprobs": 20, "max_tokens": 1}')
            tokenized = await fetch(port, "POST", "/tokenize", b'{"prompt": "x"}')
            metrics = (await fetch(port, "GET", "/metrics"))[1].decode()
            reset = await fetch(port, "POST", "/reset_prefix_cache")
            stats = json.loads((await fetch(port, "GET", "/lb/stats"))[1])
            return streamed, logprobs, tokenized, metrics, reset, stats

        streamed, logprobs, tokenized, metrics, reset, stats = self.run_pd(prefill, decode, scenario)
        first = prefill.bodies[0]
        self.assertEqual((first["max_tokens"], first["stream"]), (1, False))
        self.assertNotIn("stream_options", first)
        self.assertNotIn("min_tokens", first)
        self.assertEqual(first["kv_transfer_params"]["do_remote_decode"], True)
        self.assertEqual(first["kv_transfer_params"]["do_remote_prefill"], False)
        self.assertEqual((first["prompt"], first["ignore_eos"]), ([1, 2, 3], True))
        second = decode.bodies[0]
        self.assertEqual({k: v for k, v in second.items() if k != "kv_transfer_params"}, self.REQUEST)
        self.assertEqual(second["kv_transfer_params"]["remote_block_ids"], [[1, 2]])
        self.assertTrue(second["kv_transfer_params"]["do_remote_prefill"])
        self.assertEqual(streamed[0], 200)
        self.assertEqual(streamed[1].count(b"data: "), 4)
        self.assertIn(b'"backend": "d"', streamed[1])
        self.assertLess(streamed[2], 0.5, "the decode stream is passed through as it arrives (0.8 s in total)")
        self.assertEqual((logprobs[0], tokenized[0]), (200, 200))
        self.assertEqual([b.get("prompt_logprobs") for b in prefill.bodies[1:]], [20, None])
        self.assertEqual(decode.served, 1, "prompt_logprobs and /tokenize stay on the prefill server")
        self.assertIn('vllm:num_requests_running{replica="prefill",engine="0",model_name="axk2"} 1.0', metrics)
        self.assertIn('vllm:num_requests_running{replica="decode",engine="0",model_name="axk2"} 1.0', metrics)
        self.assertEqual((reset[0], prefill.resets, decode.resets), (200, 1, 1))
        self.assertEqual({k: stats["pd"][k] for k in ("disaggregated", "prefill_only", "prefill_failed")}, {
            "disaggregated": 1, "prefill_only": 2, "prefill_failed": 0})
        self.assertEqual([b["name"] for b in stats["backends"]], ["prefill", "decode"])

    def test_a_failed_prefill_never_reaches_decode_and_missing_params_are_counted(self):
        failing, decode = FakeBackend("p", role="prefill", prefill_status=500), FakeBackend("d", delay=0.01)

        async def failed(port, balancer):
            status = (await fetch(port, "POST", "/v1/completions", json.dumps(self.REQUEST).encode()))[0]
            return status, json.loads((await fetch(port, "GET", "/lb/stats"))[1])

        status, stats = self.run_pd(failing, decode, failed)
        self.assertEqual((status, decode.served, stats["pd"]["prefill_failed"]), (500, 0, 1))
        silent, decode = FakeBackend("p", role="prefill", transfer=False), FakeBackend("d", delay=0.01)
        status, stats = self.run_pd(silent, decode, failed)
        self.assertEqual((status, decode.served, stats["pd"]["missing_kv_transfer_params"]), (200, 1, 1))
        self.assertNotIn("kv_transfer_params", decode.bodies[0])

    def test_smoke_request_checks_the_answer_and_the_counters(self):
        cases = ((dict(role="prefill"), True, "ok"), (dict(role="prefill", transfer=False), False, "kv_transfer_params"),
                 (dict(role="prefill", prefill_status=500), False, "HTTPError"))
        for options, ok, reason in cases:
            prefill, decode = FakeBackend("p", **options), FakeBackend("d", delay=0.01)

            async def scenario(port, balancer):
                return await asyncio.to_thread(lb_proxy.smoke, f"http://127.0.0.1:{port}", 10)

            with self.subTest(reason=reason):
                result = self.run_pd(prefill, decode, scenario)
                self.assertEqual(result[0], ok, result)
                self.assertIn(reason, result[1])
                if ok:
                    self.assertEqual((prefill.bodies[0]["max_tokens"], decode.bodies[0]["max_tokens"]), (1, 16))
        with mock.patch.object(lb_proxy, "smoke", return_value=(False, "no answer")), mock.patch("builtins.print") as shown:
            self.assertEqual(lb_proxy.main(["--smoke", "http://127.0.0.1:1"]), 1)
        self.assertEqual(json.loads(shown.call_args[0][0]), {"ok": False, "reason": "no answer"})

    def test_command_line_needs_backends_or_both_roles(self):
        for argv in (["--prefill", "http://a:1"], ["--decode", "http://a:1"],
                     ["--backend", "http://a:1", "--prefill", "http://a:1", "--decode", "http://b:1"], []):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    mock.patch("sys.stderr"):
                lb_proxy.main(argv)


if __name__ == "__main__":
    unittest.main()
