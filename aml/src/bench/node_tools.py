"""Node helpers of the benchmark protocol (protocol.sh): standard library plus Ray.

  python3 node_tools.py pull <view> <outputs> <head ip>
      Copies every other node's env.json, GPU telemetry and InfiniBand counters, and the tail of each second-server
      log (vllm-<phase>-replica1.log), into <view>/node<i>/ for bench_stats.py package.
  python3 node_tools.py tail <outputs> <head ip> <log name>
      Prints the tail of <log name> from the other nodes (failure reports of two-server phases).
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
    else:
        print(reclaim(args.out, args.nodes, args.head, args.port)["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
