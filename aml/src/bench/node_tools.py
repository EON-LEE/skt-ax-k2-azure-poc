"""Node helpers of the benchmark protocol (protocol.sh): standard library plus Ray.

  python3 node_tools.py pull <view> <outputs> <head ip>
      Copies every other node's env.json, GPU telemetry and InfiniBand counters, and the tail of each second-server
      log (vllm-<phase>-replica1.log), into <view>/node<i>/ for bench_stats.py package.
  python3 node_tools.py tail <outputs> <head ip> <log name>
      Prints the tail of <log name> from the other nodes (failure reports of two-server phases).
  python3 node_tools.py facts <out.json> <outputs> <head ip> <log name>
      The KV cache size and DP rank lines of the latest start in <log name> on the other nodes: their engines
      (a second server, an mp-multinode layout's headless DP engines) never reach node0's log.
  python3 node_tools.py preflight <out.json> <head ip> <nixl_preflight.py> [--timeout 120]
      NIXL's VRAM registration (nixl_preflight.py) on node0 and every other node, with the defaults and with
      UCX_MEMTYPE_CACHE=n, each run limited to --timeout seconds. Prints the PD servers' environment: the first
      variant that passes on every node ("default" or "UCX_MEMTYPE_CACHE=n"), else "none: <reason>".
  python3 node_tools.py descs <out.json> <outputs> <head ip> <phase>
      vLLM NixlConnector's debug lines "Registering descs: [(base, length, device, ''), ...]" and UCX's host-typing
      errors from both PD servers' logs of the latest start, with the zero-length descriptors counted.
  python3 node_tools.py reclaim <out.json> <nodes> <head ip> [--port 8000]
      One reclaim of a node set that was not clean before a phase, on node0 and, when the phase uses more than one
      node, on every other node: SIGKILL the leftover serving processes among those that hold a GPU device or
      listen on the port (reclaim_local), then remove the Ray placement groups that are not REMOVED and the bench
      replica actor. The result goes to <out.json>.

Remote work runs as Ray tasks pinned to a node (resources node:<ip>, num_cpus=0), like bench_clean's checks.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

TAIL_BYTES = 16384
TAIL_LINES = 200
PULLED = ("env.json", "gpu-telemetry.csv", "ib-counters.jsonl")
RAY_SYSTEM = ("raylet", "gcs_server", "dashboard", "log_monitor", "monitor.py", "runtime_env", "ray_client_server")
GPU_DEVICE = re.compile(r"/dev/nvidia(\d+|-uvm)")
START_MARK = "=== axk2 replica start"  # written by bench_replica before every launch of node1's process
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
FACT_LINES = re.compile(r"GPU KV cache size|Maximum concurrency for|is assigned as DP rank|Auto-fit max_model_len")
PREFLIGHT_VARIANTS = (("default", {}), ("UCX_MEMTYPE_CACHE=n", {"UCX_MEMTYPE_CACHE": "n"}))
DESC = re.compile(r"\((\d+), (\d+), (\d+), '[^']*'\)")
MP_CHILD = ("multiprocessing.spawn", "--multiprocessing-fork")


def tail_text(path, max_bytes=TAIL_BYTES, max_lines=TAIL_LINES):
    """The last max_lines lines within the last max_bytes bytes of a file."""
    with open(path, "rb") as handle:
        handle.seek(max(0, os.path.getsize(path) - max_bytes))
        data = handle.read().decode(errors="replace")
    return "\n".join(data.splitlines()[-max_lines:])


def collect(outputs):
    """The files of this node that the results package needs (runs on the node; node0's folder is skipped)."""
    files = {}
    for folder in sorted(glob.glob(os.path.join(outputs, "node*"))):
        node = os.path.basename(folder)
        if node == "node0":
            continue
        for name in PULLED:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    files[f"{node}/{name}"] = handle.read()
        for path in sorted(glob.glob(os.path.join(folder, "vllm-*-replica1.log"))):
            files[f"{node}/tail-{os.path.basename(path)}"] = tail_text(path).encode()
    return files


def node_tail(outputs, name):
    """The tail of an output file of this node (runs on the node)."""
    parts = []
    for folder in sorted(glob.glob(os.path.join(outputs, "node*"))):
        path = os.path.join(folder, name)
        if os.path.basename(folder) != "node0" and os.path.exists(path):
            parts.append(f"--- {os.path.basename(folder)}/{name}, last {TAIL_LINES} lines ---\n{tail_text(path)}")
    return "\n".join(parts)


def node_facts(outputs, name, max_lines=24):
    """{node: {kv_tokens, lines}} from <name> after its last start mark (runs on the node; node0 is skipped)."""
    facts = {}
    for folder in sorted(glob.glob(os.path.join(outputs, "node*"))):
        path = os.path.join(folder, name)
        if os.path.basename(folder) == "node0" or not os.path.exists(path):
            continue
        kv, lines = [], []
        with open(path, errors="replace") as handle:
            for line in handle:
                if line.startswith(START_MARK):
                    kv, lines = [], []
                    continue
                found = KV_TOKENS.search(line)
                if found:
                    kv.append(int(found.group(1).replace(",", "")))
                if FACT_LINES.search(line) and len(lines) < max_lines:
                    lines.append(re.sub(r"^.*?\[[\w./-]+:\d+\] ", "", line.strip())[:240])
        facts[os.path.basename(folder)] = {"kv_tokens": kv, "lines": lines}
    return facts


def facts(out, outputs, head, name):
    found = {}
    for value in on_other_nodes(connect(), node_facts, head, outputs, name).values():
        found.update(value)
    result = {"nodes": found, "kv_tokens": [tokens for node in found.values() for tokens in node["kv_tokens"]]}
    with open(out, "w") as handle:
        json.dump(result, handle)
    return result


def run_preflight(script, variants, timeout=120):
    """{variant: nixl_preflight.py's JSON summary} on this node, each run limited to `timeout` seconds."""
    out = {}
    for name, extra in variants:
        env = {**os.environ, **extra, "CUDA_VISIBLE_DEVICES": "0"}  # Ray hides every GPU from a GPU-less task
        try:
            run = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env, timeout=timeout)
            lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
            out[name] = json.loads(lines[-1]) if lines else {"ok": False, "reason": f"no summary (exit {run.returncode})"}
        except subprocess.TimeoutExpired:
            out[name] = {"ok": False, "reason": f"no answer within {timeout} s"}
        except (OSError, ValueError) as exc:
            out[name] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def preflight(out, head, script, timeout=120):
    """The preflight on node0 and every other node; the first variant that passes on every node (two at least)."""
    result = {"node0": run_preflight(script, PREFLIGHT_VARIANTS, timeout)}
    try:
        found = on_other_nodes(connect(), run_preflight, head, script, PREFLIGHT_VARIANTS, timeout, timeout=3 * timeout)
        for index, (_, value) in enumerate(sorted(found.items()), start=1):
            result[f"node{index}"] = value
    except Exception as exc:
        result["ray_error"] = f"{type(exc).__name__}: {exc}"[:300]
    nodes = sorted(key for key in result if key.startswith("node"))
    passing = [name for name, _ in PREFLIGHT_VARIANTS
               if len(nodes) > 1 and all(result[node].get(name, {}).get("ok") for node in nodes)]
    result["choice"] = passing[0] if passing else None
    if not passing:
        result["reason"] = ("; ".join(f"{node} {name}: {result[node].get(name, {}).get('reason') or 'failed'}"
                                      for node in nodes for name, _ in PREFLIGHT_VARIANTS
                                      if not result[node].get(name, {}).get("ok"))
                            if len(nodes) > 1 else f"node1 not reached: {result.get('ray_error', 'no other node')}")[:400]
    with open(out, "w") as handle:
        json.dump(result, handle)
    return result


def registering_descs(path, max_lines=8, max_chars=1200):
    """The debug lines "Registering descs: [...]" and UCX's "detected as host" errors of a log after its last start
    mark: each with its descriptor count, its zero-length descriptors and its first max_chars characters."""
    lines = []
    if os.path.exists(path):
        with open(path, errors="replace") as handle:
            for line in handle:
                if line.startswith(START_MARK):
                    lines = []
                elif "Registering descs" in line or "detected as host" in line:
                    lines.append(line.strip())
    out = []
    for line in lines[:max_lines]:
        found = DESC.findall(line)
        out.append({"kind": "registering" if "Registering descs" in line else "host", "descs": len(found),
                    "zero_length": sum(1 for desc in found if desc[1] == "0"),
                    "text": re.sub(r"^.*?\[[\w./-]+:\d+\] ", "", line)[:max_chars]})
    return out


def node_descs(outputs, name):
    """{node: registering_descs} of <name> on this node (runs on the node; node0 is skipped)."""
    return {os.path.basename(folder): registering_descs(os.path.join(folder, name))
            for folder in sorted(glob.glob(os.path.join(outputs, "node*"))) if os.path.basename(folder) != "node0"}


def descs(out, outputs, head, phase):
    result = {"node0": registering_descs(os.path.join(outputs, "node0", f"vllm-{phase}.log"))}
    try:
        for value in on_other_nodes(connect(), node_descs, head, outputs, f"vllm-{phase}-replica1.log").values():
            result.update(value)
    except Exception as exc:
        result["ray_error"] = f"{type(exc).__name__}: {exc}"[:300]
    lines = [line for key, value in result.items() if key.startswith("node") for line in value]
    registering = [line for line in lines if line["kind"] == "registering"]
    result["summary"] = (f"{len(registering)} 'Registering descs' lines, {sum(l['descs'] for l in registering)} "
                         f"descriptors, {sum(l['zero_length'] for l in registering)} of zero length")
    with open(out, "w") as handle:
        json.dump(result, handle)
    return result


def write_files(view, files):
    for name, data in files.items():
        target = os.path.join(view, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target + ".tmp", "wb") as handle:
            handle.write(data)
        os.replace(target + ".tmp", target)


def connect():
    import ray
    if not ray.is_initialized():
        ray.init(address="auto", logging_level="ERROR")
    return ray


def other_nodes(ray, head):
    return sorted({n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]} - {head})


def on_other_nodes(ray, function, head, *args, timeout=150):
    """{node address: function(*args)}, run as a Ray task pinned to each other node (num_cpus=0)."""
    remote = ray.remote(num_cpus=0)(function)
    refs = {ip: remote.options(resources={f"node:{ip}": 0.001}).remote(*args) for ip in other_nodes(ray, head)}
    return {ip: ray.get(ref, timeout=timeout) for ip, ref in refs.items()}


def pull(view, outputs, head):
    count = 0
    for files in on_other_nodes(connect(), collect, head, outputs).values():
        write_files(view, files)
        count += len(files)
    return count


# ---------------------------------------------------------------------------------------------------------------
# Reclaim

def compute_pids():
    """nvidia-smi's compute-app PIDs. Inside a container they can be host-namespace numbers that name a different
    process here, so they are recorded for diagnosis and never used to choose what to kill."""
    try:
        run = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({int(value) for value in run.stdout.split() if value.strip().isdigit()})


def cmdline(pid, proc="/proc"):
    try:
        with open(os.path.join(proc, str(pid), "cmdline"), "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def parent_pid(pid, proc="/proc"):
    try:
        with open(os.path.join(proc, str(pid), "stat")) as handle:
            return int(handle.read().rsplit(")", 1)[1].split()[1])  # "pid (comm) state ppid ..."
    except (OSError, ValueError, IndexError):
        return None


def ancestors(pid, proc="/proc"):
    chain, seen = [], {pid}
    while True:
        pid = parent_pid(pid, proc)
        if not pid or pid in seen:
            return chain
        chain.append(pid)
        seen.add(pid)


def pids_with_fd(match, proc="/proc"):
    """PIDs in this (container's) /proc with an open descriptor whose link target satisfies match."""
    pids = set()
    for folder in glob.glob(os.path.join(proc, "[0-9]*", "fd")):
        try:
            descriptors = os.listdir(folder)
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                link = os.readlink(os.path.join(folder, descriptor))
            except OSError:  # closed meanwhile (the listing's own descriptor, for one)
                continue
            if match(link):
                pids.add(int(os.path.basename(os.path.dirname(folder))))
                break
    return sorted(pids)


def gpu_holders(proc="/proc"):
    """PIDs holding /dev/nvidia<N> or /dev/nvidia-uvm open (not /dev/nvidiactl alone)."""
    return pids_with_fd(lambda link: GPU_DEVICE.fullmatch(link) is not None, proc)


def listener_pids(port, proc="/proc"):
    """PIDs holding a listening TCP socket (IPv4 or IPv6) on the port: /proc/net/tcp* inodes -> /proc/<pid>/fd."""
    inodes = set()
    for table in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc, "net", table)) as handle:
                lines = handle.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return pids_with_fd(lambda link: link.startswith("socket:[") and link[8:-1] in inodes, proc) if inodes else []


def verdict(pid, line, protected, proc="/proc"):
    """(kill, reason) for a candidate, from its command line and ancestry alone."""
    lower = line.lower()
    if pid in protected:
        return False, "this reclaim or its caller"
    if "nvidia-smi" in lower:
        return False, "nvidia-smi"
    if any(name in line for name in RAY_SYSTEM):
        return False, "Ray system process"
    if "vllm" in lower:
        return True, "vllm process"
    if line.startswith("ray::"):
        return True, "Ray worker or actor"
    if any(marker in line for marker in MP_CHILD) and any("vllm" in cmdline(p, proc).lower() for p in ancestors(pid, proc)):
        return True, "multiprocessing child of a vllm process"
    return False, "unknown GPU process"


def reclaim_local(port=8000, proc="/proc"):
    """SIGKILLs this node's leftover serving processes, chosen in this container's own process view. Candidates
    hold /dev/nvidia<N> or /dev/nvidia-uvm open or listen on the port; one dies only when its command line names
    vllm, starts with ray:: (Ray workers and actors) or is a multiprocessing child of a vllm process. nvidia-smi,
    Ray's own processes, this process and its ancestors are never killed; any other candidate is recorded as spared,
    so the clean-state re-check still fails."""
    protected = {os.getpid(), *ancestors(os.getpid(), proc)}
    candidates = {pid: "holds a GPU device" for pid in gpu_holders(proc)}
    for pid in listener_pids(port, proc):
        candidates.setdefault(pid, f"listens on port {port}")
    killed, spared = [], []
    for pid, why in sorted(candidates.items()):
        line = cmdline(pid, proc)
        kill, reason = verdict(pid, line, protected, proc)
        if reason == "unknown GPU process" and why != "holds a GPU device":
            reason = f"unknown process on port {port}"
        entry = {"pid": pid, "why": why, "reason": reason, "command": line[:160]}
        if not kill:
            spared.append(entry)
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(entry)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            spared.append({**entry, "reason": f"{reason}: {exc}"})
    if killed:
        time.sleep(2)
    return {"killed": killed, "spared": spared, "nvidia_smi_pids": compute_pids()}


def reclaim_ray(ray):
    """Removes the placement groups that are not REMOVED and the bench replica actor (namespace axk2-bench)."""
    from ray._raylet import PlacementGroupID
    from ray.util.placement_group import PlacementGroup
    removed, errors = [], []
    for group, info in ray.util.placement_group_table().items():
        if info.get("state") != "REMOVED":
            try:
                ray.util.remove_placement_group(PlacementGroup(PlacementGroupID(bytes.fromhex(group))))
                removed.append(group[:12])
            except Exception as exc:
                errors.append(f"{group[:12]}: {type(exc).__name__}")
    try:
        ray.kill(ray.get_actor("replica", namespace="axk2-bench"))
        removed.append("actor replica")
    except Exception:
        pass
    return {"removed": removed, "errors": errors}


def reclaim(out, nodes, head, port=8000):
    began = time.time()
    result = {"node0": reclaim_local(port)}
    try:
        ray = connect()
        result["ray"] = reclaim_ray(ray)
        if nodes > 1:
            for index, (_, value) in enumerate(sorted(on_other_nodes(ray, reclaim_local, head, port).items()), start=1):
                result[f"node{index}"] = value
    except Exception as exc:
        result["ray_error"] = f"{type(exc).__name__}: {exc}"[:300]
    result["seconds"] = round(time.time() - began, 1)
    result["summary"] = (f"killed {sum(len(v.get('killed', [])) for k, v in result.items() if k.startswith('node'))} "
                         f"processes, removed {len((result.get('ray') or {}).get('removed', []))} Ray objects")
    with open(out, "w") as handle:
        json.dump(result, handle)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="node_tools.py")
    commands = parser.add_subparsers(dest="command", required=True)
    pulled = commands.add_parser("pull", help="copy the other nodes' telemetry and log tails into a view")
    pulled.add_argument("view")
    pulled.add_argument("outputs")
    pulled.add_argument("head")
    tail = commands.add_parser("tail", help="print a log tail from the other nodes")
    tail.add_argument("outputs")
    tail.add_argument("head")
    tail.add_argument("name")
    found = commands.add_parser("facts", help="KV cache sizes and DP rank lines of a log on the other nodes")
    found.add_argument("out")
    found.add_argument("outputs")
    found.add_argument("head")
    found.add_argument("name")
    checked = commands.add_parser("preflight", help="NIXL's VRAM registration on every node; the PD servers' environment")
    checked.add_argument("out")
    checked.add_argument("head")
    checked.add_argument("script")
    checked.add_argument("--timeout", type=int, default=120)
    registered = commands.add_parser("descs", help="the PD servers' 'Registering descs' lines and zero-length descriptors")
    registered.add_argument("out")
    registered.add_argument("outputs")
    registered.add_argument("head")
    registered.add_argument("phase")
    reclaimed = commands.add_parser("reclaim", help="kill leftover GPU processes, placement groups and port listeners")
    reclaimed.add_argument("out")
    reclaimed.add_argument("nodes", type=int)
    reclaimed.add_argument("head")
    reclaimed.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "pull":
        print(json.dumps({"files": pull(args.view, args.outputs, args.head)}))
    elif args.command == "tail":
        print("\n".join(on_other_nodes(connect(), node_tail, args.head, args.outputs, args.name).values()))
    elif args.command == "preflight":
        result = preflight(args.out, args.head, args.script, args.timeout)
        print(result["choice"] or f"none: {result['reason']}")
    elif args.command == "descs":
        print(descs(args.out, args.outputs, args.head, args.phase)["summary"])
    elif args.command == "facts":
        kv = facts(args.out, args.outputs, args.head, args.name)["kv_tokens"]
        print(f"{len(kv)} engines on the other nodes, {sum(kv):,} KV tokens")
    else:
        print(reclaim(args.out, args.nodes, args.head, args.port)["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
