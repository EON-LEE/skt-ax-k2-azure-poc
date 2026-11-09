import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aml" / "src" / "bench"))
import lb_proxy  # noqa: E402
sys.path.pop(0)


class FakeBackend:
    """A tiny HTTP/1.1 server shaped like vLLM: /health, /metrics, /reset_prefix_cache and streamed completions."""

    def __init__(self, name, healthy=True, delay=0.2, chunks=4):
        self.name, self.healthy, self.delay, self.chunks = name, healthy, delay, chunks
        self.served, self.resets, self.in_flight, self.peak = 0, 0, 0, 0

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def handle(self, reader, writer):
        start_line, headers = await lb_proxy.read_head(reader)
        await lb_proxy.read_body(reader, headers)
        method, path = start_line.split(" ")[:2]

        async def plain(status, body):
            writer.write(f"HTTP/1.1 {status} X\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()

        if path == "/health":
            await plain(200 if self.healthy else 503, b"")
        elif path == "/metrics":
            await plain(200, b'# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{engine="0",model_name="axk2"} 1.0\n')
        elif path == "/reset_prefix_cache":
            self.resets += 1
            await plain(200, b"")
        else:
            self.served += 1
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
    def run_with(self, backends, scenario):
        async def main():
            for backend in backends:
                await backend.start()
            balancer = lb_proxy.Balancer([b.url for b in backends], health_interval=0.1)
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


if __name__ == "__main__":
    unittest.main()
