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
NEW_TEMPLATES = ("slo-round-nd96.yml", "serving-diag-nd96.yml", "slo-nvfp4-2replica-nd96.yml", "collect-slo-outputs.yml")
POSIX_BASH = shutil.which("bash") and sys.platform != "win32"


def template(name):
    return (ROOT / "aml" / "jobs" / name).read_text(encoding="utf-8")


def download_snippet(text):
    """The python -c program of a git-pinned template's command, folded the way YAML folds `>-`."""
    command = re.sub(r"\n\s+", " ", text.split("command: >-", 1)[1].split("\nlimits:", 1)[0])
    return re.search(r'python3 -c "(.*?)"\s+&& break', command).group(1)


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

    @unittest.skipUnless(POSIX_BASH, "needs a POSIX bash")
    def test_a_failed_start_is_not_retried_and_tcp_is_refused(self):
        functions = PROTOCOL_SH[PROTOCOL_SH.index("bench_model() {"):]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "bin").mkdir()
            fake = folder / "bin" / "vllm"
            fake.write_text(f'#!/bin/bash\necho "$* DEV=$VLLM_SERVER_DEV_MODE" >> {folder}/vllm-calls\nexit 1\n')
            fake.chmod(0o755)
            (folder / "functions.sh").write_text(functions)
            stubs = f"""
                set -u; SRC={SRC}; BENCH={SRC}/bench; OUT={folder}; ROUND=1; FULL_NATIVE={folder}; FULL_HEALTH_TIMEOUT=5
                PKGS=/nonexistent; PATH={folder}/bin:$PATH; SERVER_PID=""; NODE_IP=127.0.0.1; BENCH_STOP_SETTLE=0
                . {folder}/functions.sh
                log() {{ :; }}; report() {{ :; }}; server_facts() {{ :; }}; parallel_facts() {{ :; }}; failure_tail() {{ :; }}
                bench_clean() {{ return 0; }}; nvidia-smi() {{ :; }}
                phase_result() {{ echo "$1 $2 $4" >> {folder}/phases; }}
                wait_healthy() {{ sleep 1; kill -0 "$SERVER_PID" 2>/dev/null; }}
            """
            run = lambda body: subprocess.run(["bash", "-c", stubs + body], capture_output=True, text=True)
            first = run('IB_MODE=ib; protocol_phase slo-A; echo "rc=$?"')
            calls = (folder / "vllm-calls").read_text().splitlines()
            second = run('IB_MODE=tcp; protocol_phase slo-C2; echo "rc=$?"')
            phases = (folder / "phases").read_text().splitlines()
        self.assertIn("rc=1", first.stdout, first.stderr)
        self.assertEqual(len(calls), 1, "one start, no --enforce-eager retry")
        self.assertIn("--max-model-len 131072", calls[0])
        self.assertIn("--api-server-count 1", calls[0])
        self.assertNotIn("enforce-eager", calls[0])
        self.assertTrue(calls[0].endswith("DEV=1"))
        self.assertIn("rc=1", second.stdout, second.stderr)
        self.assertTrue(phases[0].startswith("slo-A failed the server never became healthy"), phases)
        self.assertTrue(phases[1].startswith("slo-C2 failed the IB probe did not choose InfiniBand"), phases)


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


if __name__ == "__main__":
    unittest.main()
