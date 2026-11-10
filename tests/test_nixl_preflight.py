import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aml" / "src" / "bench" / "nixl_preflight.py"

FAKE_TORCH = """import itertools

_addresses = itertools.count(1 << 40, 1 << 32)
uint8 = "uint8"


class _Tensor:
    def __init__(self, size):
        self.nbytes, self._address = size, next(_addresses)

    def data_ptr(self):
        return self._address


class cuda:
    @staticmethod
    def set_device(index):
        pass

    @staticmethod
    def current_device():
        return 0


def empty(size, dtype=None, device=None):
    return _Tensor(size)
"""

# Like UCX in the screening: a registration after the first one misses the memory-type cache unless it is off,
# and a zero-length descriptor is always typed as host memory.
FAKE_NIXL = """import os


class nixl_agent_config:
    def __init__(self, num_threads=1):
        pass


class nixl_agent:
    def __init__(self, name, config):
        self.registered = 0

    def get_reg_descs(self, descs, kind):
        assert kind == "VRAM"
        return list(descs)

    def register_memory(self, descs):
        (base, length, device, _), = descs
        if length == 0 or (self.registered and os.environ.get("UCX_MEMTYPE_CACHE") != "n"):
            raise RuntimeError("NIXL_ERR_BACKEND: VRAM memory is detected as host by UCX")
        self.registered += 1
        return descs

    def deregister_memory(self, handle):
        pass
"""


class PreflightTests(unittest.TestCase):
    def run_preflight(self, fakes, **env):
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("UCX_", "NIXL_"))}
        run = subprocess.run([sys.executable, str(SCRIPT), "--gpu", "0"], capture_output=True, text=True, timeout=120,
                             env={**clean, "PYTHONPATH": str(fakes), "API_TOKEN": "do-not-print", **env})
        return run.returncode, run.stdout, json.loads(run.stdout.strip().splitlines()[-1])

    def fakes(self, folder):
        (Path(folder) / "torch").mkdir()
        (Path(folder) / "torch" / "__init__.py").write_text(FAKE_TORCH)
        (Path(folder) / "nixl").mkdir()
        (Path(folder) / "nixl" / "__init__.py").write_text("")
        (Path(folder) / "nixl" / "_api.py").write_text(FAKE_NIXL)
        return Path(folder)

    def test_a_memtype_cache_miss_fails_by_default_and_passes_with_the_cache_off(self):
        with tempfile.TemporaryDirectory() as folder:
            fakes = self.fakes(folder)
            code, stdout, default = self.run_preflight(fakes, UCX_NET_DEVICES="all")
            fixed_code, _, fixed = self.run_preflight(fakes, UCX_MEMTYPE_CACHE="n")
        self.assertEqual(len(stdout.strip().splitlines()), 1, "one JSON line")
        self.assertEqual((code, default["ok"]), (1, False))
        self.assertTrue(default["registrations"]["alloc-before-first-registration"]["ok"])
        self.assertFalse(default["registrations"]["alloc-after-first-registration"]["ok"])
        self.assertTrue(default["reason"].startswith("VRAM registration failed (alloc-after-first-registration): "
                                                     "RuntimeError: NIXL_ERR_BACKEND: VRAM memory is detected as host"))
        self.assertIn("libuct_cuda) was not loaded", default["reason"], "the fakes map no UCX library")
        self.assertEqual((fixed_code, fixed["ok"], fixed["reason"]), (0, True, ""))
        self.assertFalse(fixed["registrations"]["zero-length-control"]["ok"], "the control is typed as host memory")
        self.assertEqual(fixed["registrations"]["zero-length-control"]["bytes"], 0)
        self.assertEqual(set(default["ld_cache"]), {"libcuda.so.1", "libnvidia-ml.so.1", "libcudart.so.12"})
        self.assertEqual(set(default["mapped"]), {"libucs", "libuct_cuda", "libnvidia-ml", "libcuda"})

    def test_only_ucx_and_nixl_variables_are_printed(self):
        with tempfile.TemporaryDirectory() as folder:
            _, stdout, result = self.run_preflight(self.fakes(folder), UCX_NET_DEVICES="all", NIXL_PLUGIN_DIR="/x")
        self.assertNotIn("do-not-print", stdout)
        self.assertNotIn("PYTHONPATH", stdout)
        self.assertEqual(result["env"], {"NIXL_PLUGIN_DIR": "/x", "UCX_MODULE_LOG_LEVEL": "warn", "UCX_NET_DEVICES": "all",
                                         "UCX_RCACHE_MAX_UNRELEASED": "1024"})

    def test_a_node_without_nixl_reports_why_the_preflight_could_not_run(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "torch").mkdir()
            (Path(folder) / "torch" / "__init__.py").write_text(FAKE_TORCH)
            code, _, result = self.run_preflight(Path(folder))
        self.assertEqual((code, result["ok"]), (1, False))
        self.assertTrue(result["reason"].startswith("the preflight could not run: ModuleNotFoundError"), result["reason"])


if __name__ == "__main__":
    unittest.main()
