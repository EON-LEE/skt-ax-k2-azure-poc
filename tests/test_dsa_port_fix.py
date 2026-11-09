"""Our fix on top of vLLM PR #38476 in the DSA port (aml/src/dsa_port.json): the Triton paged MQA-logits decode
kernel must stay inside its logits row when the row width is not a multiple of the 64-token block.

(a) Network-free: the patched_files entry is well formed, patches the PR-new file, and the overlay writes the PR
    files before it patches them.
(b) With the PR file (raw.githubusercontent.com at the pinned commit, or DSA_PORT_PR_DIR pointing at a local copy
    of the PR tree): the diff applies with apply_overlay and gives after_sha256. Skips when offline.
(c) With torch and triton as well: the Triton interpreter runs aml/src/bench/test_dsa_port_regression.py's harness on
    CPU against the PR kernel and the patched kernel. The PR kernel writes outside its rows when the width is not
    a multiple of 64; the patched kernel never does and matches a float64 reference; in-row results of the two
    are bit-identical, and so is the whole buffer for multiples of 64. No GPU needed.
"""
import hashlib
import http.client
import importlib.util
import json
import linecache
import os
import re
import subprocess
import sys
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "aml" / "src"
PATH = "vllm/v1/attention/ops/mqa_logits_triton.py"
PORT = json.loads((SRC / "dsa_port.json").read_text(encoding="utf-8"))
ENTRY = PORT["patched_files"].get(PATH, {})
VLLM_IMPORT = "from vllm.triton_utils import tl, triton"

sys.path.insert(0, str(SRC))
try:
    import apply_overlay
finally:
    sys.path.pop(0)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_hunks(diff):
    """[(old_start, old_len, new_start, new_len, body lines)] of a unified diff."""
    hunks = []
    for line in diff.splitlines():
        header = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if header:
            a, b, c, d = header.groups()
            hunks.append((int(a), int(b or 1), int(c), int(d or 1), []))
        elif hunks and not line.startswith("\\"):
            hunks[-1][4].append(line)
    return hunks


_PR_CACHE = {}


def pr_file():
    """The PR's mqa_logits_triton.py at the pinned commit, or None when it cannot be read."""
    if "data" not in _PR_CACHE:
        local = os.environ.get("DSA_PORT_PR_DIR")
        try:
            if local:
                data = (Path(local) / PATH).read_bytes()
            else:
                url = apply_overlay.PR_RAW.format(sha=PORT["pr_commit"], path=PATH)
                with urllib.request.urlopen(url, timeout=20) as response:
                    data = response.read()
        except (OSError, http.client.HTTPException):
            data = None
        _PR_CACHE["data"] = data
    return _PR_CACHE["data"]


class MemPath:
    """Just enough of pathlib.Path for apply_dsa_port, backed by a dict instead of the disk."""

    def __init__(self, files, parts=()):
        self.files, self.parts = files, parts

    def __truediv__(self, other):
        return MemPath(self.files, self.parts + tuple(str(other).split("/")))

    @property
    def parent(self):
        return MemPath(self.files, self.parts[:-1])

    def exists(self):
        return "/".join(self.parts) in self.files

    def mkdir(self, parents=False, exist_ok=False):
        pass

    def read_bytes(self):
        return self.files["/".join(self.parts)]

    def write_bytes(self, data):
        self.files["/".join(self.parts)] = data


def run_dsa_port(port, sources):
    """apply_overlay.apply_dsa_port on an in-memory site, with `sources` standing in for the PR downloads."""
    site, tests = {}, {}
    with mock.patch.object(apply_overlay, "json", types.SimpleNamespace(loads=lambda _: port)), \
            mock.patch.object(apply_overlay, "fetch", lambda sha, path, template=None: sources[path]):
        result = apply_overlay.apply_dsa_port(MemPath(site), MemPath(tests))
    return result, site


class DsaPortFixEntryTests(unittest.TestCase):
    def test_entry_patches_the_pr_file(self):
        new = {item["path"]: item for item in PORT["new_files"]}
        self.assertEqual(new[PATH]["source"], "pr")
        self.assertEqual(ENTRY["before_sha256"], new[PATH]["pr_sha256"])
        for key in ("before_sha256", "after_sha256"):
            self.assertRegex(ENTRY[key], r"^[0-9a-f]{64}$")
        self.assertNotEqual(ENTRY["before_sha256"], ENTRY["after_sha256"])
        self.assertIn("Our fix on top of the unmerged PR", PORT["note"])

    def test_diff_is_well_formed(self):
        diff = ENTRY["diff"]
        self.assertTrue(diff.startswith(f"diff --git a/{PATH} b/{PATH}\n"))
        self.assertIn(f"\n--- a/{PATH}\n+++ b/{PATH}\n@@ ", diff)
        hunks = parse_hunks(diff)
        self.assertEqual(len(hunks), 3)
        shift = 0
        for old_start, old_len, new_start, new_len, body in hunks:
            self.assertTrue(all(line[:1] in (" ", "-", "+") for line in body), body)
            self.assertEqual(sum(line[:1] in (" ", "-") for line in body), old_len)
            self.assertEqual(sum(line[:1] in (" ", "+") for line in body), new_len)
            self.assertEqual(new_start, old_start + shift)
            shift += new_len - old_len
        added = [line[1:].strip() for *_, body in hunks for line in body if line.startswith("+")]
        removed = [line[1:].strip() for *_, body in hunks for line in body if line.startswith("-")]
        self.assertEqual(removed, ["mask=mask_n,"])
        self.assertIn("mask=mask_n & (k_offset < max_model_len),", added)
        self.assertEqual(added.count("max_model_len,"), 2, "kernel parameter and launch argument")

    def test_overlay_writes_pr_files_before_patching_them(self):
        before, after = b"a\nb\nc\n", b"a\nB\nc\n"
        port = {"upstream_pr": "pr", "pr_commit": "0" * 40,
                "new_files": [{"path": "vllm/x.py", "source": "pr", "pr_sha256": sha256(before)}],
                "patched_files": {"vllm/x.py": {"before_sha256": sha256(before), "after_sha256": sha256(after),
                                                "diff": "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"}}}
        result, site = run_dsa_port(port, {"vllm/x.py": before})
        self.assertEqual(site, {"vllm/x.py": after})
        self.assertEqual((result["new_files"], result["patched_files"]), (1, 1))


class DsaPortFixApplyTests(unittest.TestCase):
    def setUp(self):
        self.pr = pr_file()
        if self.pr is None:
            self.skipTest("PR file unreachable (offline); set DSA_PORT_PR_DIR to a local copy of the PR tree")

    def test_diff_applies_to_the_pr_file(self):
        self.assertEqual(sha256(self.pr), ENTRY["before_sha256"])
        patched = apply_overlay.apply_unified_diff(self.pr.decode(), ENTRY["diff"]).encode()
        self.assertEqual(sha256(patched), ENTRY["after_sha256"])

    def test_overlay_chain_for_the_file(self):
        item = next(item for item in PORT["new_files"] if item["path"] == PATH)
        port = {**PORT, "new_files": [item], "patched_files": {PATH: ENTRY}}
        result, site = run_dsa_port(port, {PATH: self.pr})
        self.assertEqual(sha256(site[PATH]), ENTRY["after_sha256"])
        self.assertEqual(result["pr_commit"], PORT["pr_commit"])


def load_kernel_module(tag, text):
    """Imports a copy of mqa_logits_triton.py with plain triton in place of vllm.triton_utils."""
    assert text.count(VLLM_IMPORT) == 1
    text = text.replace(VLLM_IMPORT, "import triton\nimport triton.language as tl")
    name = f"_dsa_port_fix_mqa_logits_{tag}"
    filename = f"<{name}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)  # for inspect / triton.jit
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(text, filename, "exec"), module.__dict__)
    kernel = module._fp8_paged_mqa_logits_kernel
    kernel.configs = kernel.configs[:1]  # the interpreter ignores num_warps/num_stages: nothing to tune
    return module


def interpreter_results(pr):
    """Runs the GPU test's harness on CPU against the PR kernel and the patched kernel.

    Needs TRITON_INTERPRET=1 before triton is first imported, so the test calls it in a fresh interpreter."""
    import torch
    import triton.language as tl
    from triton.runtime import interpreter

    spec = importlib.util.spec_from_file_location("_dsa_port_regression", SRC / "bench" / "test_dsa_port_regression.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    original = load_kernel_module("pr", pr.decode())
    patched = load_kernel_module("patched", apply_overlay.apply_unified_diff(pr.decode(), ENTRY["diff"]))
    create_dot = interpreter.InterpreterBuilder.create_dot

    def bf16_dot(builder, a, b, *rest):
        # triton 3.9's interpreter multiplies bf16 operands as their raw uint16 storage; widen them to fp32
        # (exact) so the values can be checked. Store addresses do not depend on this.
        def widen(x):
            if not x.dtype.is_bf16():
                return x
            return interpreter.TensorHandle((x.data.astype("uint32") << 16).view("float32"), tl.float32)
        return create_dot(builder, widen(a), widen(b), *rest)

    results = []
    with mock.patch.object(interpreter.InterpreterBuilder, "create_dot", bf16_dot):
        for case in harness.CASES:
            for pad in (0, harness.B):
                old, new = (harness.run(module, case, pad, "cpu") for module in (original, patched))
                try:
                    harness.check(*new, case)
                    failure = None
                except AssertionError as error:
                    failure = str(error)[:1000]
                old_bits, new_bits, inside = old[2].view(torch.int32), new[2].view(torch.int32), new[3]
                results.append({
                    "case": case, "pad": pad, "block": harness.B, "patched_check": failure,
                    "interpreted": type(patched._fp8_paged_mqa_logits_kernel.fn).__name__,
                    "pr_stray_writes": harness.stray_writes(*old[2:]),
                    "rows_equal": torch.equal(old_bits[inside], new_bits[inside]),
                    "buffer_equal": torch.equal(old_bits, new_bits)})
    return results


CHILD = """import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_dsa_port_fix", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("RESULT " + json.dumps(module.interpreter_results(sys.stdin.buffer.read())))
"""


class DsaPortFixInterpreterTests(unittest.TestCase):
    def test_pr_kernel_spills_and_patched_kernel_stays_in_row(self):
        if not all(importlib.util.find_spec(name) for name in ("torch", "triton")):
            self.skipTest("needs torch and triton")
        pr = pr_file()
        if pr is None:
            self.skipTest("PR file unreachable (offline); set DSA_PORT_PR_DIR")
        if sha256(pr) != ENTRY["before_sha256"]:
            self.skipTest("PR file hash mismatch (DsaPortFixApplyTests reports it)")
        child = subprocess.run([sys.executable, "-c", CHILD, __file__], input=pr, capture_output=True,
                               timeout=1800, env={**os.environ, "TRITON_INTERPRET": "1"})
        lines = [line for line in child.stdout.decode().splitlines() if line.startswith("RESULT ")]
        self.assertTrue(lines, child.stderr.decode(errors="replace")[-4000:])
        results = json.loads(lines[-1][len("RESULT "):])
        self.assertEqual(len(results), 8)
        self.assertTrue(any(r["case"][2] % r["block"] for r in results), "a width that is not a multiple of 64")
        for r in results:
            (lens, next_n, width, _), block = r["case"], r["block"]
            spill = [max(0, -(-n // block) * block - width) for n in lens for _ in range(next_n)]
            with self.subTest(case=r["case"], pad=r["pad"]):
                self.assertEqual(r["interpreted"], "InterpretedFunction")
                self.assertGreaterEqual(len(lens), 2)
                self.assertIsNone(r["patched_check"])
                # Unpadded, a spill lands in the next row (which overwrites it later on the interpreter's
                # row-by-row schedule) unless it comes from the last row; padded, every spill is visible.
                self.assertEqual(r["pr_stray_writes"], sum(spill) if r["pad"] else spill[-1])
                self.assertTrue(r["rows_equal"])
                self.assertEqual(r["buffer_equal"], r["pr_stray_writes"] == 0)


if __name__ == "__main__":
    unittest.main()


