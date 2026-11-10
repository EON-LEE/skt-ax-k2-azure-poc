import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aml" / "src" / "bench"))
import node_tools  # noqa: E402
sys.path.pop(0)

LINUX = sys.platform.startswith("linux")


class FakeRay(types.ModuleType):
    """Enough of Ray for node_tools: alive nodes, tasks pinned with resources (run at once), placement groups."""

    def __init__(self, addresses, groups=None):
        super().__init__("ray")
        self.addresses, self.pinned, self.removed, self.killed = addresses, [], [], []
        self.util = types.ModuleType("ray.util")
        self.util.placement_group_table = lambda: dict(groups or {})
        self.util.remove_placement_group = lambda group: self.removed.append(group.id)
        placement = types.ModuleType("ray.util.placement_group")
        placement.PlacementGroup = lambda group_id: types.SimpleNamespace(id=group_id.hex())
        raylet = types.ModuleType("ray._raylet")
        raylet.PlacementGroupID = lambda raw: raw
        self.modules = {"ray": self, "ray.util": self.util, "ray.util.placement_group": placement, "ray._raylet": raylet}

    def is_initialized(self):
        return True

    def nodes(self):
        return [{"NodeManagerAddress": address, "Alive": True} for address in self.addresses]

    def remote(self, **options):
        assert options == {"num_cpus": 0}

        def wrap(function):
            ray = self

            class Task:
                def options(self, resources):
                    ray.pinned.append(resources)
                    return types.SimpleNamespace(remote=lambda *args: function(*args))
            return Task()
        return wrap

    def get(self, value, timeout=None):
        return value

    def get_actor(self, name, namespace=None):
        return f"{namespace}/{name}"

    def kill(self, actor):
        self.killed.append(actor)


def outputs_tree(folder):
    outputs = Path(folder, "outputs")
    for node in ("node0", "node1"):
        (outputs / node).mkdir(parents=True)
        (outputs / node / "gpu-telemetry.csv").write_text(f"{node} telemetry\n")
    (outputs / "node1" / "env.json").write_text('{"hostname": "h1"}')
    (outputs / "node1" / "ib-counters.jsonl").write_text('{"t": 1}\n')
    (outputs / "node1" / "vllm-scr-PD-replica1.log").write_text("".join(f"line {i} " + "x" * 90 + "\n" for i in range(600)))
    return outputs


class PullTests(unittest.TestCase):
    def test_tail_keeps_the_last_lines_within_the_byte_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "log")
            path.write_text("".join(f"line {i} " + "x" * 90 + "\n" for i in range(600)))
            tail = node_tools.tail_text(str(path))
        lines = tail.splitlines()
        self.assertLessEqual(len(tail.encode()), node_tools.TAIL_BYTES)
        self.assertLessEqual(len(lines), node_tools.TAIL_LINES)
        self.assertTrue(lines[-1].startswith("line 599 "))

    def test_pull_copies_the_other_nodes_files_and_log_tails(self):
        ray = FakeRay(["10.0.0.1", "10.0.0.2"])
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, ray.modules):
            outputs, view = outputs_tree(folder), Path(folder, "view")
            with redirect_stdout(StringIO()) as shown:
                self.assertEqual(node_tools.main(["pull", str(view), str(outputs), "10.0.0.1"]), 0)
            copied = sorted(str(p.relative_to(view)).replace(os.sep, "/") for p in view.rglob("*") if p.is_file())
            tail = (view / "node1" / "tail-vllm-scr-PD-replica1.log").read_text()
            with redirect_stdout(StringIO()) as printed:
                node_tools.main(["tail", str(outputs), "10.0.0.1", "vllm-scr-PD-replica1.log"])
        self.assertEqual(json.loads(shown.getvalue()), {"files": 4})
        self.assertEqual(copied, ["node1/env.json", "node1/gpu-telemetry.csv", "node1/ib-counters.jsonl",
                                  "node1/tail-vllm-scr-PD-replica1.log"], "node0's own folder is never copied")
        self.assertLessEqual(len(tail.splitlines()), node_tools.TAIL_LINES)
        self.assertEqual(ray.pinned[0], {"node:10.0.0.2": 0.001}, "a task pinned to node1, not to the head")
        self.assertIn("--- node1/vllm-scr-PD-replica1.log, last 200 lines ---", printed.getvalue())


FAKE = 3_999_100  # PIDs of the fake /proc trees below; the real os.getpid() must not collide with them


def fake_proc(folder, processes, listeners=()):
    """A fake /proc: processes {pid: (command line, parent pid, [descriptor link targets])}; listeners
    [(table, port, socket inode)] in LISTEN state."""
    proc = Path(folder, "proc")
    for pid, (line, ppid, links) in processes.items():
        (proc / str(pid) / "fd").mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(line.replace(" ", "\0").encode() + b"\0")
        (proc / str(pid) / "stat").write_text(f"{pid} (two words) S {ppid} {pid} {pid} 0 -1\n")
        for index, link in enumerate(links):
            os.symlink(link, proc / str(pid) / "fd" / str(index))
    (proc / "net").mkdir(parents=True, exist_ok=True)
    for table, zero in (("tcp", "0" * 8), ("tcp6", "0" * 32)):
        rows = ["  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"]
        rows += [f"   {i}: {zero}:{port:04X} {zero}:0000 0A 00000000:00000000 00:00000000 00000000  1000  0 {inode} 1 0"
                 for i, (name, port, inode) in enumerate(l for l in listeners if l[0] == table)]
        (proc / "net" / table).write_text("\n".join(rows) + "\n")
    return proc


class ReclaimTests(unittest.TestCase):
    @unittest.skipUnless(LINUX, "needs symlinks")
    def test_only_serving_processes_in_the_containers_own_view_are_killed(self):
        me, P = os.getpid(), lambda n: FAKE + n
        self.assertNotIn(me, range(FAKE, FAKE + 20))
        spawn = "python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7) --multiprocessing-fork"
        processes = {
            me: ("ray::reclaim_local", P(11), ["/dev/nvidia-uvm"]),  # this reclaim, as a Ray task on node1
            P(11): ("timeout 300 python3 node_tools.py reclaim /out/reclaim-vllm.json 2 head", 1, ["/dev/nvidia0"]),
            P(1): ("VLLM::Worker_TP0", P(6), ["/dev/nvidiactl", "/dev/nvidia0"]),  # setproctitle
            P(2): ("ray::DPEngineCoreActor.run", P(8), ["pipe:[1]", "/dev/nvidia3"]),
            P(3): ("nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -lms 1000", 1, ["/dev/nvidia0"]),
            P(4): ("python3 /opt/runtime/logger.py", 1, ["pipe:[2]", "/dev/null"]),  # nvidia-smi's host PID maps here
            P(5): (spawn, P(6), ["/dev/nvidia-uvm"]),
            P(6): ("vllm serve /model --served-model-name axk2 --port 8000", 1, ["socket:[77777]"]),
            P(7): (spawn, P(4), ["/dev/nvidia1"]),  # a multiprocessing child of something else
            P(8): ("raylet --raylet_socket_name=/tmp/ray/session/sockets/raylet", 1, ["/dev/nvidia2"]),
            P(9): ("/usr/bin/python3 /opt/runtime/agent.py", 1, ["/dev/nvidia4"]),
            P(10): ("python3 -m http.server 8000", 1, ["socket:[88888]"]),
            P(12): ("python3 /opt/monitor/nvml_probe.py", 1, ["/dev/nvidiactl", "/dev/nvidia-uvm-tools"]),
        }
        with tempfile.TemporaryDirectory() as folder:
            proc = fake_proc(folder, processes, [("tcp", 8000, "77777"), ("tcp6", 8000, "88888"), ("tcp", 8100, "99999")])
            with mock.patch.object(node_tools, "compute_pids", return_value=[55, P(4)]), \
                    mock.patch.object(node_tools.os, "kill") as kill, mock.patch.object(node_tools.time, "sleep"):
                result = node_tools.reclaim_local(8000, str(proc))
        killed = {entry["pid"]: entry["reason"] for entry in result["killed"]}
        spared = {entry["pid"]: entry["reason"] for entry in result["spared"]}
        self.assertEqual([call.args for call in kill.call_args_list], [(pid, signal.SIGKILL) for pid in sorted(killed)])
        self.assertEqual(killed, {P(1): "vllm process", P(2): "Ray worker or actor",
                                  P(5): "multiprocessing child of a vllm process", P(6): "vllm process"})
        self.assertEqual(spared, {me: "this reclaim or its caller", P(11): "this reclaim or its caller", P(3): "nvidia-smi",
                                  P(7): "unknown GPU process", P(8): "Ray system process", P(9): "unknown GPU process",
                                  P(10): "unknown process on port 8000"})
        self.assertNotIn(P(4), {**killed, **spared}, "nvidia-smi's (host-namespace) PID never selects a process")
        self.assertNotIn(P(12), {**killed, **spared}, "/dev/nvidiactl and nvidia-uvm-tools are not GPU devices")
        self.assertEqual(result["nvidia_smi_pids"], [55, P(4)], "recorded for diagnosis only")

    @unittest.skipUnless(LINUX, "needs /proc")
    def test_a_vllm_listener_dies_and_a_ray_system_listener_survives(self):
        script = ("import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
                  "print(s.getsockname()[1], flush=True); time.sleep(300)")
        start = lambda title: subprocess.Popen(["bash", "-c", f'exec -a "{title}" "{sys.executable}" -c "{script}"'],
                                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        server, raylet = start("VLLM::APIServer"), start("raylet")
        try:
            ports = [int(process.stdout.readline()) for process in (server, raylet)]
            # real signals, but only for listeners on these two ports: no real GPU process is ever a candidate here
            with mock.patch.object(node_tools, "gpu_holders", return_value=[]), \
                    mock.patch.object(node_tools, "compute_pids", return_value=[]):
                first, second = node_tools.reclaim_local(ports[0]), node_tools.reclaim_local(ports[1])
            self.assertEqual(server.wait(10), -9)
            self.assertIsNone(raylet.poll(), "raylet keeps running")
            self.assertEqual([(k["pid"], k["reason"]) for k in first["killed"]], [(server.pid, "vllm process")])
            self.assertEqual([(s["pid"], s["reason"]) for s in second["spared"]], [(raylet.pid, "Ray system process")])
        finally:
            for process in (server, raylet):
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()

    @unittest.skipUnless(LINUX, "needs /proc")
    def test_listener_on_the_port_is_found(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            self.assertIn(os.getpid(), node_tools.listener_pids(port))
        self.assertEqual(node_tools.listener_pids(port), [])

    @unittest.skipUnless(LINUX, "needs /proc")
    def test_the_caller_is_never_a_target(self):
        with mock.patch.object(node_tools, "gpu_holders", return_value=[]), \
                mock.patch.object(node_tools, "compute_pids", return_value=[]), \
                mock.patch.object(node_tools, "listener_pids", return_value=[os.getpid()]), \
                mock.patch.object(node_tools.os, "kill") as kill:
            result = node_tools.reclaim_local()
        kill.assert_not_called()
        self.assertEqual((result["spared"][0]["why"], result["spared"][0]["reason"]),
                         ("listens on port 8000", "this reclaim or its caller"))

    def test_reclaim_removes_placement_groups_and_reaches_the_other_nodes(self):
        groups = {"ab" * 14: {"state": "CREATED"}, "cd" * 14: {"state": "REMOVED"}, "ef" * 14: {"state": "PENDING"}}
        ray = FakeRay(["10.0.0.1", "10.0.0.2"], groups)
        local = {"killed": [{"pid": 7, "why": "GPU compute process", "command": "python3 vllm worker"}], "spared": []}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, ray.modules), \
                mock.patch.object(node_tools, "reclaim_local", return_value=local):
            out = Path(folder, "reclaim.json")
            with redirect_stdout(StringIO()) as shown:
                node_tools.main(["reclaim", str(out), "2", "10.0.0.1"])
            result, removed = json.loads(out.read_text()), sorted(ray.removed)
            one_node = node_tools.reclaim(str(Path(folder, "one.json")), 1, "10.0.0.1")
        self.assertEqual(removed, ["ab" * 14, "ef" * 14])
        self.assertEqual(ray.killed, ["axk2-bench/replica", "axk2-bench/replica"])
        self.assertEqual(sorted(k for k in result if k.startswith("node")), ["node0", "node1"])
        self.assertEqual(result["summary"], "killed 2 processes, removed 3 Ray objects")
        self.assertEqual(shown.getvalue().strip(), result["summary"])
        self.assertNotIn("node1", one_node, "a one-node phase leaves the other node alone")

    def test_reclaim_without_ray_still_cleans_node0(self):
        local = {"killed": [], "spared": []}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"ray": None}), \
                mock.patch.object(node_tools, "reclaim_local", return_value=local):
            result = node_tools.reclaim(str(Path(folder, "r.json")), 2, "10.0.0.1")
        self.assertEqual(result["node0"], local)
        self.assertIn("ray_error", result)


if __name__ == "__main__":
    unittest.main()
