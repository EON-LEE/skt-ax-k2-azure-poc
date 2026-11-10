import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "aml" / "src"
sys.path[:0] = [str(ROOT / "aml"), str(SRC / "bench"), str(SRC)]
import render_job  # noqa: E402
import slo_suites  # noqa: E402
del sys.path[:3]

ENTRY = (SRC / "entry.sh").read_text(encoding="utf-8")
PROTOCOL_SH = (SRC / "bench" / "protocol.sh").read_text(encoding="utf-8")
PROTOCOL = slo_suites.load_protocol()
SCREEN_TEMPLATES = ("serving-screen-fp8-nd96.yml", "serving-screen-nvfp4-nd96.yml", "serving-screen-nvfp4-2node-nd96.yml")
NEW_TEMPLATES = ("slo-round-nd96.yml", "serving-diag-nd96.yml", "slo-nvfp4-2replica-nd96.yml", "collect-slo-outputs.yml",
                 *SCREEN_TEMPLATES)
POSIX_BASH = shutil.which("bash") and sys.platform != "win32"


def template(name):
    return (ROOT / "aml" / "jobs" / name).read_text(encoding="utf-8")


def env_value(text, name):
    found = re.search(rf'^  {name}: "?([^"\n]*)"?$', text, re.M)
    return found.group(1) if found else None


def download_snippet(text):
    """The python -c program of a git-pinned template's command, folded the way YAML folds `>-`."""
    command = re.sub(r"\n\s+", " ", text.split("command: >-", 1)[1].split("\nlimits:", 1)[0])
    return re.search(r'python3 -c "(.*?)"\s+&& break', command).group(1)


FAKE_MLFLOW = """import json, os


def set_tags(tags):  # the job's mlflow, as far as report.py needs it
    with open(os.path.join(os.path.dirname(__file__), "tags.jsonl"), "a") as out:
        out.write(json.dumps(sorted(tags)) + "\\n")
"""


def protocol_functions(folder, extra_stubs=""):
    """bash that defines protocol.sh's functions with entry.sh's helpers stubbed; outputs go to `folder`. PKGS holds
    a fake mlflow, so report.py only counts as sent when AXK2_REPORT_PKGS reaches it."""
    (folder / "bin").mkdir(exist_ok=True)
    (folder / "pkgs").mkdir(exist_ok=True)
    (folder / "pkgs" / "mlflow.py").write_text(FAKE_MLFLOW)
    (folder / "functions.sh").write_text(PROTOCOL_SH[PROTOCOL_SH.index("bench_model() {"):])
    return f"""
        set -u; SRC={SRC}; BENCH={SRC}/bench; OUT={folder}; ROUND=1; FULL_NATIVE={folder}; FULL_HEALTH_TIMEOUT=5
        PKGS={folder}/pkgs; PATH={folder}/bin:$PATH; SERVER_PID=""; NODE_IP=127.0.0.1; BENCH_STOP_SETTLE=0
        BENCH_PACKAGE_DIR={folder}/package-view
        . {folder}/functions.sh
        log() {{ :; }}; report() {{ :; }}; server_facts() {{ :; }}; parallel_facts() {{ :; }}; failure_tail() {{ :; }}
        bench_clean() {{ return 0; }}; nvidia-smi() {{ :; }}
        phase_result() {{ echo "$1 $2 ${{4:-}}" >> {folder}/phases; }}
        wait_healthy() {{ sleep 1; kill -0 "$SERVER_PID" 2>/dev/null; }}
        {extra_stubs}
    """


class EntryHookTests(unittest.TestCase):
    def test_entry_sources_the_protocol_before_the_probe_and_dispatches_to_it(self):
        hook = '[ -f "$SRC/bench/protocol.sh" ] && . "$SRC/bench/protocol.sh"'
        self.assertIn(hook, ENTRY)
        self.assertLess(ENTRY.index("die() {"), ENTRY.index(hook))
        self.assertLess(ENTRY.index(hook), ENTRY.index("read -r IFACE NODE_IP HEAD_IP"))
        self.assertIn('elif [ -n "${BENCH_PLAN:-}" ]; then protocol_phase "$name"', ENTRY)
        self.assertIn('NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}', ENTRY)
        self.assertNotIn("VLLM_SERVER_DEV_MODE", ENTRY, "dev mode only for the protocol's servers")

    def test_kernel_tests_include_the_dsa_regression_test_when_present(self):
        block = ENTRY.split('if [ "$RUN_KERNEL_TESTS" = "1" ]; then', 1)[1].split("\nfi\n", 1)[0]
        self.assertIn('$(ls "$SRC/bench/test_dsa_port_regression.py" 2>/dev/null)', block)
        self.assertIn("test_mqa_logits_triton.py", block)

    def test_protocol_phases_never_fall_back_to_eager(self):
        code = [line for line in PROTOCOL_SH.splitlines() if "enforce-eager" in line
                and not line.lstrip().startswith("#") and "no --enforce-eager retry" not in line]
        self.assertEqual(code, [])
        self.assertIsNone(re.search(r"^\s*(?:if !?\s*|elif !?\s*)?serve\s", PROTOCOL_SH, re.M),
                          "protocol phases must not use entry.sh's serve(), which retries eagerly")
        self.assertIn('setsid vllm serve "$model" "${SPEC_ARGS[@]}"', PROTOCOL_SH)
        self.assertIn('kill -INT -- "-$SERVER_PID"', PROTOCOL_SH)
        self.assertNotIn("export VLLM_SERVER_DEV_MODE", PROTOCOL_SH)
        self.assertIn('env "${SPEC_ENV[@]}"', PROTOCOL_SH)
        self.assertIn("never runs over TCP", PROTOCOL_SH)
        self.assertIn('bench_nccl "$began"', PROTOCOL_SH)
        self.assertIn(f"--repo {PROTOCOL['checkpoints']['nvfp4']['repo']}", PROTOCOL_SH)

    def test_review_fixes_nccl_path_smi_probe_and_replica_timeout(self):
        # NCCL logs: one path that exists on every node, since vLLM's Ray DP backend carries node0's NCCL_*
        # variables into node1's engine actors; bench_nccl scans that folder and the exit trap keeps the files.
        self.assertIn('NCCL_DEBUG_FILE="$BENCH_NCCL_DIR/%h.%p.log"', PROTOCOL_SH)
        self.assertNotRegex(PROTOCOL_SH, r'NCCL_DEBUG_FILE="\$OUT')
        self.assertIn('python3 - "$BENCH_NCCL_DIR" "$1" "$NODE_IP"', PROTOCOL_SH)
        self.assertIn('cp -p "$BENCH_NCCL_DIR"/*.log "$OUT/nccl/"', PROTOCOL_SH)
        # nvidia-smi -c is --compute-mode, not a sample count: the field-name probe must not use it.
        probe = [line for line in PROTOCOL_SH.splitlines() if "nvidia-smi --query-gpu" in line and "/dev/null 2>&1 ||" in line]
        self.assertEqual(len(probe), 1)
        self.assertNotIn(" -c ", probe[0])
        # The second D replica waits as long as node0's server does.
        replica = PROTOCOL_SH[PROTOCOL_SH.index("bench_replica start"):].split("| tail -n 1", 1)[0]
        self.assertIn('"VLLM_ENGINE_READY_TIMEOUT_S=$FULL_HEALTH_TIMEOUT"', replica)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_shell_syntax(self):
        for path in (SRC / "entry.sh", SRC / "bench" / "protocol.sh"):
            with self.subTest(path=path.name):
                self.assertNotIn(b"\r\n", path.read_bytes())
                subprocess.run(["bash", "-n", str(path)], check=True)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_other_plans_skip_the_protocol(self):
        script = (f'set -u; SRC={SRC}; PHASE_PLAN="demo"; OUT=/nonexistent; '
                  f'. {SRC}/bench/protocol.sh; echo "plan=${{BENCH_PLAN:-none}} phases=$PHASE_PLAN"')
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "plan=none phases=demo")
        self.assertIn('*" scr-"*', PROTOCOL_SH.split("esac", 1)[0], "screening tokens activate the protocol")
        self.assertIn('--layouts "${ROUND_LAYOUTS:-}"', PROTOCOL_SH)
        self.assertIn('--record "$OUT/protocol-plan.json"', PROTOCOL_SH)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_a_failed_start_is_not_retried_and_tcp_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            stubs = protocol_functions(folder)
            fake = folder / "bin" / "vllm"
            fake.write_text(f'#!/bin/bash\necho "$* DEV=$VLLM_SERVER_DEV_MODE" >> {folder}/vllm-calls\nexit 1\n')
            fake.chmod(0o755)
            run = lambda body: subprocess.run(["bash", "-c", stubs + body], capture_output=True, text=True)
            first = run('IB_MODE=ib; protocol_phase slo-A; echo "rc=$?"')
            calls = (folder / "vllm-calls").read_text().splitlines()
            second = run('IB_MODE=tcp; protocol_phase slo-C2; echo "rc=$?"')
            phases = (folder / "phases").read_text().splitlines()
            start = json.loads((folder / "start-slo-A.json").read_text())
            packaged = (folder / "package-view" / "reported.json").exists()
        self.assertIn("rc=1", first.stdout, first.stderr)
        self.assertEqual(len(calls), 1, "one start, no --enforce-eager retry")
        self.assertIn("--max-model-len 131072", calls[0])
        self.assertIn("--api-server-count 1", calls[0])
        self.assertNotIn("enforce-eager", calls[0])
        self.assertTrue(calls[0].endswith("DEV=1"))
        self.assertIn("rc=1", second.stdout, second.stderr)
        self.assertTrue(phases[0].startswith("slo-A failed the server never became healthy"), phases)
        self.assertTrue(phases[1].startswith("slo-C2 failed the IB probe did not choose InfiniBand"), phases)
        self.assertEqual((start["started"], start["length_fallback"], start["requested_max_model_len"]), (False, None, 131072))
        self.assertTrue(packaged, "the phase reported its package (best effort) and still returned 1")

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_screening_relaunches_once_with_auto_fit_when_the_kv_cache_is_short(self):
        short = ("ValueError: To serve at least one request with the model's max seq len (131072), (9.6 GiB KV cache is "
                 "needed, which is larger than the available KV cache memory (5.1 GiB). Based on the available memory, "
                 "the estimated maximum model length is 69632. Try increasing `gpu_memory_utilization`")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            stubs = protocol_functions(folder, "IB_MODE=ib")
            fake = folder / "bin" / "vllm"
            fake.write_text(f'#!/bin/bash\necho "$*" >> {folder}/vllm-calls\n'
                            f'case "$*" in *"--max-model-len 131072"*) echo "{short}" ;; esac\nexit 1\n')
            fake.chmod(0o755)
            run = lambda body: subprocess.run(["bash", "-c", stubs + body], capture_output=True, text=True)
            result = run('protocol_phase scr-C16; echo "rc=$?"')
            calls = (folder / "vllm-calls").read_text().splitlines()
            start = json.loads((folder / "start-scr-C16.json").read_text())
            kept = (folder / "vllm-scr-C16-short.log").read_text()
            (folder / "vllm-calls").unlink()
            other = run('protocol_phase slo-C2; echo "rc=$?"')
            slo_calls = (folder / "vllm-calls").read_text().splitlines()
        self.assertIn("rc=1", result.stdout, result.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertIn("--max-model-len 131072", calls[0])
        self.assertIn("--max-model-len -1", calls[1])
        self.assertEqual(start["length_fallback"], {"requested": 131072, "estimated": 69632})
        self.assertIn("estimated maximum model length is 69632", kept)
        self.assertIn("rc=1", other.stdout)
        self.assertEqual(len(slo_calls), 1, "rounds never relaunch: only the screening has the length fallback")

    def pd_stubs(self, folder, smoke='{"ok": true, "reason": "ok"}', replica_ip="10.1.2.3"):
        """protocol.sh's functions with the second server, the proxy, the smoke test and the client stubbed."""
        return protocol_functions(folder, f"""
            bench_model() {{ echo {folder}; }}
            bench_replica() {{ echo "$*" >> {folder}/replica; [ "$1" = start ] && echo {replica_ip}; return 0; }}
            bench_wait_url() {{ echo "$1" >> {folder}/urls; return 0; }}
            bench_effective() {{ echo '{{"problems": []}}'; }}
            report() {{ echo "$*" >> {folder}/reports; }}
            timeout() {{ echo "$1" >> {folder}/limits; shift; "$@"; }}
            python3() {{ if [ "${{1:-}}" = "$BENCH/lb_proxy.py" ] && [ "${{2:-}}" = --smoke ]; then echo '{smoke}';
                         elif [ "${{1:-}}" = "$BENCH/lb_proxy.py" ]; then echo "$*" > {folder}/proxy; sleep 2;
                         elif [ "${{1:-}}" = "$BENCH/slo_suites.py" ] && [ "${{2:-}}" = run ]; then echo "$*" > {folder}/client;
                         elif [ "${{1:-}}" = "$BENCH/node_tools.py" ] && [ "${{2:-}}" = tail ]; then echo "NODE1 TAIL: NIXL error";
                         else command python3 "$@"; fi; }}
            wait_healthy() {{ echo 7 > {folder}/startup-$1.seconds; }}
        """)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_pd_starts_the_decode_server_on_node1_and_the_pd_proxy(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            stubs = self.pd_stubs(folder)
            fake = folder / "bin" / "vllm"
            fake.write_text(f'#!/bin/bash\necho "$* HOST=$VLLM_NIXL_SIDE_CHANNEL_HOST" >> {folder}/vllm-calls\n')
            fake.chmod(0o755)
            result = subprocess.run(["bash", "-c", stubs + 'IB_MODE=ib; protocol_phase scr-PD; echo "rc=$?"'],
                                    capture_output=True, text=True, timeout=120)
            call = (folder / "vllm-calls").read_text()
            replica = (folder / "replica").read_text().splitlines()
            proxy = (folder / "proxy").read_text()
            client = (folder / "client").read_text()
            limits = (folder / "limits").read_text().split()
            start = json.loads((folder / "start-scr-PD.json").read_text())
        self.assertIn("rc=0", result.stdout, result.stderr)
        self.assertIn('"kv_role": "kv_producer"', call)
        self.assertIn("HOST=127.0.0.1", call, "the prefill server announces node0's address")
        started = replica[0]
        self.assertTrue(started.startswith("start scr-PD-replica1"))
        self.assertIn('\\"kv_role\\": \\"kv_consumer\\"', started)
        self.assertIn('"VLLM_NIXL_SIDE_CHANNEL_HOST": "@node_ip"', started)
        self.assertIn('"VLLM_ENGINE_READY_TIMEOUT_S": "5"', started)
        self.assertEqual(replica[-1], "stop")
        self.assertIn("--prefill http://127.0.0.1:8000 --decode http://10.1.2.3:8000", proxy)
        self.assertIn("--base http://127.0.0.1:8100", client)
        self.assertIn("--suites screen", client)
        self.assertIn("200", limits, "the smoke request is bounded")
        self.assertIn("5400", limits, "the client of an experimental phase runs under timeout")
        self.assertEqual((start["proxy"], start["experimental"], start["seconds_to_healthy"]), ("pd", True, 7))

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_a_failed_pd_smoke_skips_the_suites_and_a_dead_peer_reports_its_log(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            fake = folder / "bin"
            fake.mkdir()
            (fake / "vllm").write_text("#!/bin/bash\n")
            (fake / "vllm").chmod(0o755)
            smoke = subprocess.run(["bash", "-c", self.pd_stubs(folder, '{"ok": false, "reason": "1 prefill answers carried no kv_transfer_params"}')
                                    + 'IB_MODE=ib; protocol_phase scr-PD; echo "rc=$?"'], capture_output=True, text=True, timeout=120)
            phases = (folder / "phases").read_text().splitlines()
            ran_client = (folder / "client").exists()
            smoke_reports = (folder / "reports").read_text()
            for name in ("phases", "reports"):
                (folder / name).unlink()
            dead = subprocess.run(["bash", "-c", self.pd_stubs(folder, replica_ip="") + 'protocol_phase scr-PD; echo "rc=$?"'],
                                  capture_output=True, text=True, timeout=120)
            reports = (folder / "reports").read_text()
            failure = reports[reports.index("node0.scr-PD.failure"):]
            dead_phase = (folder / "phases").read_text()
        smoke_failure = smoke_reports[smoke_reports.index("node0.scr-PD.failure"):]
        self.assertIn("PD smoke failed: 1 prefill answers carried no kv_transfer_params", smoke_failure)
        self.assertIn("--- node1: vllm-scr-PD-replica1.log ---\nNODE1 TAIL: NIXL error", smoke_failure)
        self.assertIn("rc=1", smoke.stdout, smoke.stderr)
        self.assertTrue(phases[0].startswith("scr-PD failed experimental phase: PD smoke failed:"), phases)
        self.assertIn("no kv_transfer_params", phases[0])
        self.assertFalse(ran_client, "no suites after a failed smoke test")
        self.assertIn("rc=1", dead.stdout, dead.stderr)
        self.assertIn("experimental phase: the second server could not be started on node1", dead_phase)
        self.assertIn("--- node1: vllm-scr-PD-replica1.log ---\nNODE1 TAIL: NIXL error", failure)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_a_dirty_node_is_reclaimed_once_and_the_event_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reclaim = f"""
                timeout() {{ shift; "$@"; }}
                report() {{ echo "AXK2_REPORT $1 $(cat "${{3:-}}" 2>/dev/null)"; echo "AXK2_REPORT_WARN mlflow unavailable: no mlflow"; }}
                python3() {{ if [ "${{1:-}}" = "$BENCH/node_tools.py" ] && [ "${{2:-}}" = reclaim ]; then
                               echo "$*" >> {folder}/reclaims; echo '{{"summary": "killed 2 processes"}}' > "$3";
                               echo "killed 2 processes, removed 1 Ray objects";
                             else command python3 "$@"; fi; }}
                bench_clean() {{ [ "$2" != before ] && return 0; echo x >> {folder}/checks;
                                 [ "$(wc -l < {folder}/checks)" -gt "$DIRTY_CHECKS" ]; }}
            """
            fake = folder / "bin"
            fake.mkdir()
            (fake / "vllm").write_text("#!/bin/bash\nexit 1\n")
            (fake / "vllm").chmod(0o755)
            run = lambda dirty: subprocess.run(
                ["bash", "-c", protocol_functions(folder, reclaim) + f'IB_MODE=ib; DIRTY_CHECKS={dirty}; rm -f {folder}/checks; '
                 'protocol_phase slo-A; echo "rc=$?"'], capture_output=True, text=True, timeout=120)
            once, twice = run(1), run(2)
            phases = (folder / "phases").read_text().splitlines()
            reclaims = (folder / "reclaims").read_text().splitlines()
            vllm_ran = (folder / "vllm-slo-A.log").exists()
        self.assertIn("rc=1", once.stdout)
        self.assertTrue(phases[0].startswith("slo-A failed the server never became healthy"), phases)
        self.assertIn("[reclaimed before the start: killed 2 processes, removed 1 Ray objects]", phases[0])
        self.assertTrue(phases[1].startswith("slo-A failed a node was not clean before the start"), phases)
        self.assertIn("[reclaimed before the start: killed 2 processes", phases[1])
        self.assertEqual(len(reclaims), 2, "one reclaim per phase, never a loop")
        self.assertFalse(any("AXK2_REPORT" in line for line in phases), "report.py's output stays out of the note")
        self.assertIn(f"reclaim {folder}/reclaim-slo-A.json 2 127.0.0.1", reclaims[0])
        self.assertTrue(vllm_ran)

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_package_is_reported_from_the_job_best_effort(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            out = folder / "outputs" / "node0"
            (out / "verify").mkdir(parents=True)
            (out / "phases.jsonl").write_text(json.dumps({"phase": "scr-A", "status": "passed"}) + "\n")
            stubs = protocol_functions(folder, f"OUT={out}; NNODES=1")
            run = lambda body: subprocess.run(["bash", "-c", stubs + body], capture_output=True, text=True, timeout=300)
            unsent = run('PKGS=/nonexistent; bench_package scr-A; echo "rc=$?"')
            nothing_recorded = not (folder / "package-view" / "reported.json").exists()
            phase = run('false; bench_package scr-A; echo "rc=$?"')
            state = json.loads((folder / "package-view" / "reported.json").read_text())
            tags = (folder / "pkgs" / "tags.jsonl").read_text().splitlines()
            final = run('bench_package exit; echo "rc=$?"')
            other = run('RANK=1; bench_package exit; echo "rc=$?"; ls ' + str(folder))
            log = (out / "package-report.log").read_text()
        self.assertIn("rc=0", unsent.stdout)
        self.assertTrue(nothing_recorded, "without mlflow, report.py's warning means nothing was sent")
        summaries = [line for line in log.splitlines() if '"keys"' in line]
        self.assertIn('"slo.package": {"chunks": 1, "sent": false}', summaries[0])
        self.assertIn("AXK2_REPORT_WARN mlflow unavailable", log.split(summaries[0])[0])
        self.assertIn("rc=0", phase.stdout)
        self.assertEqual(sorted(state), ["slo.package"])
        self.assertIn('"slo.package.000"', tags[0], "the job's mlflow received the chunks")
        self.assertEqual(len(summaries), 3, "unsent, after the phase and on exit")
        self.assertIn('"slo.package": {"chunks": 1, "sent": true}', log.splitlines()[-1], "exit sends every key again")
        self.assertIn("rc=0", final.stdout)
        self.assertIn("rc=0", other.stdout)
        trap = next(line for line in PROTOCOL_SH.splitlines() if line.startswith("trap "))
        self.assertIn("bench_package exit", trap)
        body = PROTOCOL_SH[PROTOCOL_SH.index("bench_package() {"):].split("\n}\n", 1)[0]
        self.assertEqual(body.count("timeout "), 2, "the Ray pull and the package step are both bounded")
        self.assertIn('AXK2_REPORT_PKGS="${PKGS:-}"', body)
        self.assertTrue(body.rstrip().endswith("return 0"))


class TemplateTests(unittest.TestCase):
    PIN = {"AXK2_SRC_GIT": "owner/repo@" + "a" * 40,
           "AXK2_SRC_MANIFEST": "entry.sh:" + "b" * 64 + ",bench/slo_suites.py:" + "c" * 64}

    def test_new_templates_render_and_pin_the_protocol(self):
        fp8, nvfp4 = PROTOCOL["checkpoints"]["fp8"]["revision"], PROTOCOL["checkpoints"]["nvfp4"]["revision"]
        for name in NEW_TEMPLATES:
            with self.subTest(template=name):
                text = template(name)
                self.assertIn("os.makedirs(os.path.dirname('/tmp/axk2src/' + n), exist_ok=True)", text)
                rendered = render_job.render(text, "cGF5bG9hZA==", self.PIN)
                self.assertIn(self.PIN["AXK2_SRC_MANIFEST"], rendered)
                self.assertIsNone(re.search(r"__AXK2_[A-Z0-9_]+__", rendered))
                self.assertLess(len(rendered), 8_000, "AML caps the total docker argument size")
                if "instance_count: 2" in text:
                    self.assertIn('IB_PROBE: "1"', text)
                    self.assertNotIn("NCCL_IB_DISABLE", text)
                    self.assertIn("test -f /tmp/axk2src/bench/protocol.sh", text)
                    plan = re.search(r'PHASE_PLAN: "([^"]+)"', text).group(1).split()
                    round_ = int(re.search(r'ROUND: "(\d+)"', text).group(1))
                    self.assertTrue(slo_suites.expand_plan(PROTOCOL, plan, round_ or 1))
        round_job = template("slo-round-nd96.yml")
        self.assertIn('PHASE_PLAN: "slo-round"', round_job)
        self.assertIn(f'NVFP4_IN: "hf:{nvfp4}"', round_job)
        self.assertIn(f"entry.sh none hf:{fp8}", round_job)
        self.assertGreaterEqual(int(re.search(r"timeout: (\d+)", round_job).group(1)), 36000)
        self.assertIn('RUN_KERNEL_TESTS: "1"', template("serving-diag-nd96.yml"))
        replicas = template("slo-nvfp4-2replica-nd96.yml")
        self.assertIn("FULL_REPO: skt/A.X-K2-NVFP4", replicas)
        self.assertIn(f"entry.sh none hf:{nvfp4}", replicas)
        self.assertIn("bench_stats.py package", template("collect-slo-outputs.yml"))

    def test_screening_templates_follow_the_protocol(self):
        jobs = {job["template"].rsplit("/", 1)[1]: job for job in PROTOCOL["screen"]["jobs"].values()}
        self.assertEqual(sorted(jobs), sorted(SCREEN_TEMPLATES))
        screened = set()
        for name in SCREEN_TEMPLATES:
            with self.subTest(template=name):
                text, job = template(name), jobs[name]
                plan = env_value(text, "PHASE_PLAN").split()
                self.assertEqual(plan, job["plan"])
                self.assertEqual(plan[0], next(r for r in PROTOCOL["screen"]["references"] if r in plan),
                                 "the parity reference is the job's first launch")
                self.assertEqual(env_value(text, "ROUND"), "0")
                self.assertIn(f"instance_count: {job['nodes']}", text)
                self.assertNotIn("cpu-lp", text)
                self.assertIn("compute: azureml:a100-nd96-lp", text)
                self.assertIn("os.makedirs(os.path.dirname('/tmp/axk2src/' + n), exist_ok=True)", text)
                self.assertRegex(text, r"Expected about [\d.]+ h")
                timeout = int(re.search(r"timeout: (\d+)", text).group(1))
                hours = float(re.search(r"Expected about ([\d.]+) h", text).group(1))
                self.assertGreaterEqual(timeout, 1.5 * hours * 3600, "a margin over the expected duration")
                nvfp4 = env_value(text, "FULL_REPO") == PROTOCOL["checkpoints"]["nvfp4"]["repo"]
                revision = PROTOCOL["checkpoints"]["nvfp4" if nvfp4 else "fp8"]["revision"]
                self.assertIn(f"entry.sh none hf:{revision}", text)
                for phase in plan:
                    spec = slo_suites.phase_spec(PROTOCOL, phase, 0)
                    self.assertEqual(spec["checkpoint"], "nvfp4" if nvfp4 else "fp8", phase)
                    self.assertLessEqual(spec["nodes"] * spec["replicas"], job["nodes"], phase)
                    self.assertEqual(spec["suites"], ["screen"], phase)
                screened.update(plan)
        self.assertEqual(env_value(template("serving-screen-fp8-nd96.yml"), "RUN_KERNEL_TESTS"), "1")
        self.assertEqual(screened, {p for p in PROTOCOL["phases"] if p.startswith("scr-")})
        self.assertEqual(env_value(template("slo-round-nd96.yml"), "ROUND_LAYOUTS"), ",".join(PROTOCOL["rounds"]["layouts"]))
        self.assertIn("FALLBACK ONLY", template("collect-slo-outputs.yml"))

    def test_layouts_expected_to_fail_or_relaunch_run_last(self):
        estimates = PROTOCOL["estimates"]["layouts"]
        risky = lambda spec: bool(spec["experimental"] or PROTOCOL["layouts"][spec["layout"]].get("expected")
                                  or estimates.get(spec["layout"], {}).get("expected"))
        for job in PROTOCOL["screen"]["jobs"].values():
            with self.subTest(job=job["template"]):
                flags = [risky(slo_suites.phase_spec(PROTOCOL, phase, 0)) for phase in job["plan"]]
                self.assertTrue(flags[-1], "every job ends with the launches expected to fail or to relaunch")
                self.assertEqual(flags, sorted(flags), "no healthy launch after a risky one")
        self.assertEqual(PROTOCOL["screen"]["jobs"]["fp8"]["plan"][-2:], ["scr-C16", "scr-B0dcp"])
        self.assertEqual(PROTOCOL["screen"]["jobs"]["nvfp4"]["plan"][-1], "scr-D8")

    def test_manifest_lists_bench_files_for_new_templates_only(self):
        def fake_run(command, capture_output=True, check=True):
            git = command[3:]
            outputs = {"rev-parse": b"f" * 40 + b"\n", "status": b"", "branch": b"  origin/main\n",
                       "remote": b"https://github.com/owner/repo.git\n"}
            stdout = (outputs[git[0]] if git[0] in outputs else b"blob of " + git[1].encode())
            return mock.Mock(stdout=stdout)
        with mock.patch.object(render_job.subprocess, "run", side_effect=fake_run):
            repo, manifest = render_job.git_pin()
        names = [item.split(":", 1)[0] for item in manifest.split(",")]
        self.assertEqual(repo, "owner/repo@" + "f" * 40)
        self.assertIn("entry.sh", names)
        for name in ("bench/slo_suites.py", "bench/protocol.sh", "bench/slo_protocol.json", "bench/bench_stats.py",
                     "bench/lb_proxy.py", "bench/parity_corpus.json"):
            self.assertIn(name, names)
        expected = hashlib.sha256(b"blob of " + ("f" * 40 + ":aml/src/bench/slo_suites.py").encode()).hexdigest()
        self.assertIn(f"bench/slo_suites.py:{expected}", manifest)
        old = render_job.manifest_for(template("ep-compare-fp8-nd96.yml"), manifest)
        self.assertNotIn("bench/", old)
        self.assertIn("entry.sh:", old)
        self.assertEqual(render_job.manifest_for(template("slo-round-nd96.yml"), manifest), manifest)

    def test_download_commands_write_the_bench_folder(self):
        names = [p.name for p in render_job.src_files()] + [f"bench/{p.name}" for p in render_job.bench_files()]
        blobs = {name: (SRC / name).read_bytes().replace(b"\r\n", b"\n") for name in names}
        manifest = ",".join(f"{n}:{hashlib.sha256(b).hexdigest()}" for n, b in blobs.items())

        class Response:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

        opener = lambda url, timeout=0: Response(blobs[url.split("/aml/src/", 1)[1]])
        for name, manifest_used, works in (("slo-round-nd96.yml", manifest, True),
                                           ("ep-compare-fp8-nd96.yml", manifest, False),
                                           ("ep-compare-fp8-nd96.yml", render_job.manifest_for(
                                               template("ep-compare-fp8-nd96.yml"), manifest), True)):
            with self.subTest(template=name, bench=("bench/" in manifest_used)), tempfile.TemporaryDirectory() as folder:
                code = download_snippet(template(name)).replace("/tmp/axk2src/", Path(folder).as_posix() + "/")
                env = {"AXK2_SRC_GIT": "owner/repo@" + "a" * 40, "AXK2_SRC_MANIFEST": manifest_used}
                with mock.patch.dict(os.environ, env), mock.patch.object(urllib.request, "urlopen", opener), \
                        warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)  # the one-liner leaves files to the GC
                    if works:
                        exec(compile(code, name, "exec"), {})
                        gc.collect()
                        self.assertTrue(Path(folder, "entry.sh").is_file())
                        self.assertEqual(Path(folder, "bench", "protocol.sh").is_file(), "bench/" in manifest_used)
                    else:
                        with self.assertRaises(OSError):
                            exec(compile(code, name, "exec"), {})
                        gc.collect()

    def test_the_bundle_carries_no_benchmark_files(self):
        import base64
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(render_job.payload())), mode="r:gz") as tar:
            names = tar.getnames()
        self.assertFalse([n for n in names if n.startswith("bench") or "slo_" in n or "parity" in n], names)
        self.assertIn("entry.sh", names)


class PublicFileTests(unittest.TestCase):
    FILES = [ROOT / "docs" / "BENCHMARK_PROTOCOL.md", *sorted((SRC / "bench").glob("*.*")),
             *(ROOT / "aml" / "jobs" / name for name in NEW_TEMPLATES)]

    def test_no_identifiers_hosts_or_personal_names(self):
        # Account and domain names of whoever runs the suite must not leak into the public files.
        local = [re.escape(v) for v in {os.environ.get(k, "") for k in ("USERNAME", "USER", "USERDOMAIN")} if len(v) >= 4]
        patterns = (r"\b(?!0\.0\.0\.0\b|127\.0\.0\.1\b)\d{1,3}(?:\.\d{1,3}){3}\b",
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    r"[\w.+-]+@[\w-]+\.[\w.]+", r"azurewebsites\.net|azureml\.ms|\brg-[a-z0-9]|-mlw\b",
                    r"[A-Za-z]:\\Users") + ((r"(?i)" + "|".join(local),) if local else ())
        for path in self.FILES:
            if path.suffix not in {".md", ".py", ".sh", ".json", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                with self.subTest(file=path.name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text))

    def test_protocol_doc_tables_and_values(self):
        doc = (ROOT / "docs" / "BENCHMARK_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("구현 완료, GPU 실행 전", doc)
        table = []
        for line in doc.splitlines() + [""]:
            if line.startswith("|"):
                table.append(line.count("|"))
            elif table:
                self.assertEqual(len(set(table)), 1, f"a table has rows with different cell counts: {table}")
                table = []
        prices = PROTOCOL["prices_usd_per_node_hour"]
        self.assertIn(f"{prices['on_demand']}달러", doc)
        self.assertIn(f"{prices['spot']}달러", doc)
        self.assertIn(f"{PROTOCOL['server']['max_model_len']['performance']:,}", doc)
        for phase in ("slo-A", "slo-C2", "slo-D", "slo-A2", "slo-C2s", "dx1-D", "dx2-D", "diag-B0dcp"):
            self.assertIn(f"`{phase}`", doc)
        for phase in (p for p in PROTOCOL["phases"] if p.startswith("scr-")):
            self.assertIn(f"`{phase}`", doc)
        for name in SCREEN_TEMPLATES:
            self.assertIn(f"`{name}`", doc)
        for needle in ("`slo.package`", "`ROUND_LAYOUTS`", "bench_stats.py select", "fetch_results.py show", "대체 경로",
                       "공용 네트워크 접근이 꺼져", "CPU 클러스터가 없고", "약 195달러", "약 705달러", "약 900달러"):
            self.assertIn(needle, doc)


if __name__ == "__main__":
    unittest.main()
