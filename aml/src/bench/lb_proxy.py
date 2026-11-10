"""HTTP proxies of the benchmark protocol, standard library only.

Load balancer for D x 2 (two independent vLLM servers):
  python3 lb_proxy.py --listen 127.0.0.1:8100 --backend http://127.0.0.1:8000 --backend http://<node1>:8000
Each request goes to the healthy backend with the fewest requests in flight (ties in round-robin order); the
response, including server-sent events, is passed through chunk by chunk as it arrives.

Prefill/decode disaggregation for PD (vLLM 0.23 NixlConnector, the protocol of vLLM's
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py):
  python3 lb_proxy.py --listen 127.0.0.1:8100 --prefill http://127.0.0.1:8000 --decode http://<node1>:8000
A completion first goes to the prefill server with max_tokens 1, no streaming and kv_transfer_params
{"do_remote_decode": true, ...}; the kv_transfer_params of its answer then go with the unchanged request to the
decode server, which reads the KV cache from the prefill server and streams the answer back through the proxy.
Requests with prompt_logprobs (P1 parity) and every other path go to the prefill server alone.

  python3 lb_proxy.py --smoke http://127.0.0.1:8100
One short completion (16 tokens) through a running proxy, then its /lb/stats: exit 0 only when the answer came
back and the proxy counted no failure and no prefill answer without kv_transfer_params (the PD smoke test).

Both proxies answer themselves:
  GET  /health              200 when every backend answers its /health (503 otherwise)
  GET  /lb/stats            requests routed, in flight and failed per backend (and the two-step counts of PD)
  GET  /metrics             every backend's /metrics, each sample labelled replica="<backend index or role>"
  POST /reset_prefix_cache  sent to every backend; 200 only if every backend accepted it
HTTP/1.1 with one request per connection (Connection: close on both sides).
"""
import argparse
import asyncio
import itertools
import json
import re
import sys
import time
import urllib.request
import uuid
from urllib.parse import urlsplit

HOP_HEADERS = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "content-length", "host", "upgrade",
               "te", "trailer"}
SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?(\s.*)$")
COMPLETIONS = ("/v1/completions", "/v1/chat/completions")


class Backend:
    def __init__(self, index, url, name=None):
        parts = urlsplit(url)
        self.index, self.url, self.host, self.port = index, url.rstrip("/"), parts.hostname, parts.port or 80
        self.name = str(index) if name is None else name
        self.in_flight = self.routed = self.failed = 0
        self.healthy = False

    def stats(self):
        return {"backend": self.index, "name": self.name, "url": self.url, "healthy": self.healthy,
                "in_flight": self.in_flight, "routed": self.routed, "failed": self.failed}


def label_metrics(text, replica, keep_comments=True):
    """Adds replica="<i>" to every sample line of a Prometheus page."""
    lines = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            if keep_comments:
                lines.append(line)
            continue
        found = SAMPLE.match(line)
        if not found:
            continue
        name, labels, rest = found.groups()
        labels = f'{{replica="{replica}",{labels[1:]}' if labels and labels != "{}" else f'{{replica="{replica}"}}'
        lines.append(f"{name}{labels}{rest}")
    return "\n".join(lines)


async def read_head(reader):
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = [tuple(part.strip() for part in line.split(":", 1)) for line in lines[1:] if ":" in line]
    return lines[0], headers


async def read_body(reader, headers):
    values = {name.lower(): value for name, value in headers}
    if values.get("transfer-encoding", "").lower() == "chunked":
        body = b""
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                return body
            body += await reader.readexactly(size)
            await reader.readexactly(2)
    length = int(values.get("content-length", "0") or 0)
    return await reader.readexactly(length) if length else b""


async def request(backend, method, path, body=b"", headers=(), timeout=10.0):
    """(status, body) of a small request to one backend (health checks, /metrics, fan-out)."""
    async def go():
        reader, writer = await asyncio.open_connection(backend.host, backend.port)
        try:
            lines = [f"{method} {path} HTTP/1.1", f"Host: {backend.host}:{backend.port}", "Connection: close",
                     f"Content-Length: {len(body)}", *(f"{n}: {v}" for n, v in headers)]
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
            await writer.drain()
            status, response_headers = await read_head(reader)
            values = {n.lower() for n, _ in response_headers}
            data = await read_body(reader, response_headers) if values & {"content-length", "transfer-encoding"} \
                else await reader.read()
            return int(status.split(" ")[1]), data
        finally:
            writer.close()
    return await asyncio.wait_for(go(), timeout)


class Balancer:
    policy = "least outstanding requests, ties round-robin"

    def __init__(self, urls, health_interval=2.0, names=None):
        self.backends = [Backend(index, url, (names or [None] * len(urls))[index]) for index, url in enumerate(urls)]
        self.turn = itertools.count()
        self.health_interval = health_interval
        self.started = time.time()

    def pick(self):
        """The healthy backend with the fewest requests in flight; ties go round-robin."""
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            return None
        start, count = next(self.turn), len(self.backends)
        return min(healthy, key=lambda b: (b.in_flight, (b.index - start) % count))

    async def check(self, backend):
        try:
            status, _ = await request(backend, "GET", "/health", timeout=5.0)
            backend.healthy = status == 200
        except Exception:
            backend.healthy = False

    async def health_loop(self):
        while True:
            await asyncio.gather(*(self.check(b) for b in self.backends))
            await asyncio.sleep(self.health_interval)

    async def respond(self, writer, status, body, content_type="application/json"):
        reason = {200: "OK", 404: "Not Found", 502: "Bad Gateway", 503: "Service Unavailable"}.get(status, "Status")
        data = body if isinstance(body, bytes) else body.encode()
        writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len(data)}\r\n"
                     f"Connection: close\r\n\r\n".encode("latin-1") + data)
        await writer.drain()

    async def handle(self, reader, writer):
        try:
            try:
                start_line, headers = await read_head(reader)
                body = await read_body(reader, headers)
            except (asyncio.IncompleteReadError, ValueError):
                return
            method, target = start_line.split(" ")[:2]
            path = target.split("?", 1)[0]
            if method == "GET" and path == "/health":
                await asyncio.gather(*(self.check(b) for b in self.backends))
                ok = all(b.healthy for b in self.backends)
                await self.respond(writer, 200 if ok else 503, json.dumps({"healthy": ok}))
            elif method == "GET" and path == "/lb/stats":
                await self.respond(writer, 200, json.dumps({"policy": self.policy,
                                                            "uptime_seconds": round(time.time() - self.started, 1),
                                                            "backends": [b.stats() for b in self.backends],
                                                            **self.extra_stats()}))
            elif method == "GET" and path == "/metrics":
                pages = await asyncio.gather(*(request(b, "GET", "/metrics") for b in self.backends),
                                             return_exceptions=True)
                parts = [label_metrics(page[1].decode(), b.name, keep_comments=b.index == 0)
                         for b, page in zip(self.backends, pages) if not isinstance(page, Exception) and page[0] == 200]
                status = 200 if len(parts) == len(self.backends) else 502
                await self.respond(writer, status, "\n".join(parts) + "\n", "text/plain; version=0.0.4")
            elif method == "POST" and path == "/reset_prefix_cache":
                results = await asyncio.gather(*(request(b, "POST", target, timeout=120.0) for b in self.backends),
                                               return_exceptions=True)
                codes = [r[0] if not isinstance(r, Exception) else None for r in results]
                await self.respond(writer, 200 if all(c == 200 for c in codes) else 502, json.dumps({"backends": codes}))
            else:
                await self.forward(method, target, headers, body, writer)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    def extra_stats(self):
        return {}

    async def forward(self, method, target, headers, body, writer):
        backend = self.pick()
        if backend is None:
            await self.respond(writer, 503, json.dumps({"error": "no healthy backend"}))
            return
        await self.relay(backend, method, target, headers, body, writer)

    async def relay(self, backend, method, target, headers, body, writer):
        """Sends one request to a backend and passes its response through chunk by chunk."""
        backend.in_flight += 1
        backend.routed += 1
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(backend.host, backend.port)
        except OSError:
            backend.in_flight -= 1
            backend.failed += 1
            backend.healthy = False
            await self.respond(writer, 502, json.dumps({"error": f"backend {backend.name} unreachable"}))
            return
        try:
            kept = [(n, v) for n, v in headers if n.lower() not in HOP_HEADERS]
            lines = [f"{method} {target} HTTP/1.1", f"Host: {backend.host}:{backend.port}", *(f"{n}: {v}" for n, v in kept),
                     f"Content-Length: {len(body)}", "Connection: close"]
            upstream_writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
            await upstream_writer.drain()
            status, response_headers = await read_head(upstream_reader)
            kept = [(n, v) for n, v in response_headers if n.lower() not in {"connection", "keep-alive"}]
            writer.write(("\r\n".join([status, *(f"{n}: {v}" for n, v in kept), "Connection: close"]) + "\r\n\r\n")
                         .encode("latin-1"))
            await writer.drain()
            while True:
                chunk = await upstream_reader.read(65536)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
            if not status.split(" ")[1].startswith(("2", "3")):
                backend.failed += 1
        except (OSError, asyncio.IncompleteReadError):
            backend.failed += 1
        finally:
            backend.in_flight -= 1
            upstream_writer.close()


class Disaggregated(Balancer):
    """Prefill/decode disaggregation: backends[0] prefills (kv_producer), backends[1] decodes (kv_consumer)."""
    policy = "prefill then decode (vLLM NixlConnector kv_transfer_params)"

    def __init__(self, prefill, decode, health_interval=2.0, prefill_timeout=3600.0):
        super().__init__([prefill, decode], health_interval, names=["prefill", "decode"])
        self.prefill_timeout = prefill_timeout
        self.counts = {"disaggregated": 0, "prefill_only": 0, "prefill_failed": 0, "missing_kv_transfer_params": 0}
        self.prefill_seconds = 0.0

    def extra_stats(self):
        done = self.counts["disaggregated"]
        return {"pd": {**self.counts, "prefill_seconds_mean": round(self.prefill_seconds / done, 4) if done else None}}

    async def forward(self, method, target, headers, body, writer):
        prefill, decode = self.backends
        try:
            payload = json.loads(body) if method == "POST" and target.split("?", 1)[0] in COMPLETIONS else None
        except ValueError:
            payload = None
        if not isinstance(payload, dict) or payload.get("prompt_logprobs") is not None:
            self.counts["prefill_only"] += 1
            await self.relay(prefill, method, target, headers, body, writer)
            return
        first = {k: v for k, v in payload.items() if k not in ("stream_options", "min_tokens", "min_completion_tokens")}
        first.update(stream=False, max_tokens=1, kv_transfer_params={
            "do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None, "remote_block_ids": None,
            "remote_host": None, "remote_port": None})
        if "max_completion_tokens" in payload:
            first["max_completion_tokens"] = 1
        kept = [(n, v) for n, v in headers if n.lower() not in HOP_HEADERS and n.lower() != "x-request-id"]
        kept.append(("X-Request-Id", next((v for n, v in headers if n.lower() == "x-request-id"), uuid.uuid4().hex)))
        began = time.time()
        prefill.in_flight += 1
        prefill.routed += 1
        try:
            status, data = await request(prefill, "POST", target, json.dumps(first).encode(), kept, self.prefill_timeout)
        except Exception as exc:
            status, data = None, json.dumps({"error": f"prefill failed: {type(exc).__name__}"}).encode()
        finally:
            prefill.in_flight -= 1
        if status != 200:
            prefill.failed += 1
            self.counts["prefill_failed"] += 1
            await self.respond(writer, status or 502, data)
            return
        self.prefill_seconds += time.time() - began
        self.counts["disaggregated"] += 1
        try:
            params = json.loads(data).get("kv_transfer_params")
        except (ValueError, AttributeError):
            params = None
        if params:
            payload["kv_transfer_params"] = params
        else:
            self.counts["missing_kv_transfer_params"] += 1
        await self.relay(decode, method, target, kept, json.dumps(payload).encode(), writer)


def smoke(url, timeout=180.0, model="axk2"):
    """(ok, reason) of one short greedy completion through a running proxy and of the proxy's counters after it."""
    body = json.dumps({"model": model, "prompt": "The capital of Korea is", "max_tokens": 16, "temperature": 0,
                       "stream": False}).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read())
        with urllib.request.urlopen(url.rstrip("/") + "/lb/stats", timeout=30) as response:
            stats = json.loads(response.read())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]
    pd = stats.get("pd") or {}
    if not answer.get("choices"):
        return False, "the answer has no choices"
    if pd.get("missing_kv_transfer_params"):
        return False, f"{pd['missing_kv_transfer_params']} prefill answers carried no kv_transfer_params"
    failed = {b.get("name", b.get("backend")): b["failed"] for b in stats.get("backends", []) if b.get("failed")}
    if pd.get("prefill_failed") or failed:
        return False, f"the proxy counted failed requests: {failed or pd.get('prefill_failed')}"
    return True, "ok"


async def serve(listen, urls=None, ready=None, prefill=None, decode=None):
    host, port = listen.rsplit(":", 1)
    balancer = Disaggregated(prefill, decode) if prefill else Balancer(urls)
    await asyncio.gather(*(balancer.check(b) for b in balancer.backends))
    server = await asyncio.start_server(balancer.handle, host, int(port))
    health = asyncio.ensure_future(balancer.health_loop())
    if ready is not None:
        ready(balancer, server)
    try:
        async with server:
            await server.serve_forever()
    finally:
        health.cancel()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:8100")
    parser.add_argument("--backend", action="append", help="http://host:port (repeat): least outstanding requests")
    parser.add_argument("--prefill", help="http://host:port of the prefill server (with --decode)")
    parser.add_argument("--decode", help="http://host:port of the decode server (with --prefill)")
    parser.add_argument("--smoke", help="http://host:port of a running proxy: one short request, then its stats")
    parser.add_argument("--smoke-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.smoke:
        ok, reason = smoke(args.smoke, args.smoke_timeout)
        print(json.dumps({"ok": ok, "reason": reason}))
        return 0 if ok else 1
    if bool(args.prefill) != bool(args.decode) or bool(args.prefill) == bool(args.backend):
        parser.error("give --backend (repeatable) or both --prefill and --decode")
    try:
        asyncio.run(serve(args.listen, args.backend, prefill=args.prefill, decode=args.decode))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
