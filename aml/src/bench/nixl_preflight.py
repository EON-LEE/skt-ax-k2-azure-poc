"""NIXL/UCX VRAM-registration preflight of a PD phase (vllm/vllm-openai:v0.23.0-cu129, nixl-cu12 1.2.0 with its
bundled UCX). protocol.sh runs it on node0 and node1 before the PD servers start, with the job's UCX environment and
with the overrides of node_tools.PREFLIGHT_VARIANTS (UCX_TLS=all and UCX_NET_DEVICES=all, then also
UCX_MEMTYPE_CACHE=n). The screening's PD launch failed in register_kv_caches with NIXL_ERR_BACKEND, "VRAM memory is
detected as host by UCX" (the job runtime's UCX_TLS=tcp leaves out cuda_copy); this checks what that depends on:

  - the loader cache has the CUDA libraries UCX's CUDA module needs (libcuda, libnvidia-ml, libcudart);
  - UCX's CUDA transport (nixl_cu12.libs/ucx/libuct_cuda) is mapped once a NIXL agent exists;
  - VRAM allocated before the first registration and VRAM allocated after it both register as VRAM (the second
    one goes through UCX's memory-type cache, which UCX_MEMTYPE_CACHE=n turns off);
  - a zero-length VRAM descriptor, the control: UCX types it as host memory, with the same error text.

    python3 nixl_preflight.py [--gpu 0]

One GPU. Prints one JSON line, the last line of stdout (UCX may print its own lines before it), and exits 0 when
both VRAM registrations work. Of the environment it prints only the names and values of UCX_* and NIXL_* variables.
"""
import argparse
import json
import os
import subprocess
import sys

LIBRARIES = ("libcuda.so.1", "libnvidia-ml.so.1", "libcudart.so.12")
MAPPED = {"libucs": "nixl_cu12.libs/libucs-", "libuct_cuda": "nixl_cu12.libs/ucx/libuct_cuda",
          "libnvidia-ml": "libnvidia-ml.so", "libcuda": "libcuda.so"}
REQUIRED = ("alloc-before-first-registration", "alloc-after-first-registration")


def registrations(gpu, result):
    """The three VRAM registrations on one GPU; records which libraries are mapped once the agent exists."""
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    torch.cuda.set_device(gpu)
    device = torch.cuda.current_device()
    before = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")  # like vLLM's weights and KV cache
    agent = nixl_agent("axk2-preflight", nixl_agent_config(num_threads=4))
    with open("/proc/self/maps") as handle:
        maps = handle.read()
    result["mapped"] = {name: marker in maps for name, marker in MAPPED.items()}

    def register(tensor, length=None):
        size = tensor.nbytes if length is None else length
        try:
            handle = agent.register_memory(agent.get_reg_descs([(tensor.data_ptr(), size, device, "")], "VRAM"))
            agent.deregister_memory(handle)
            return {"ok": True, "bytes": size}
        except Exception as exc:  # nixlBackendError
            return {"ok": False, "bytes": size, "error": f"{type(exc).__name__}: {exc}"[:200]}

    found = {"alloc-before-first-registration": register(before)}
    after = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")  # a fresh allocation once UCX's cache exists
    found["alloc-after-first-registration"] = register(after)
    found["zero-length-control"] = register(before, 0)  # expected to fail: typed as host memory
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ.setdefault("UCX_MODULE_LOG_LEVEL", "warn")  # a UCX module that fails to load says so
    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")  # what vLLM's NixlConnector sets
    result = {"ok": False, "reason": "", "gpu": args.gpu,
              "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("UCX_", "NIXL_"))}}
    try:
        cache = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        cache = ""
    result["ld_cache"] = {library: library in cache for library in LIBRARIES}
    try:
        result["registrations"] = registrations(args.gpu, result)
    except Exception as exc:  # no GPU, torch or nixl here: the preflight itself could not run
        result["reason"] = f"the preflight could not run: {type(exc).__name__}: {exc}"[:300]
    else:
        failed = [name for name in REQUIRED if not result["registrations"][name]["ok"]]
        result["ok"] = not failed
        if failed:
            hint = "" if result["mapped"]["libuct_cuda"] else "; UCX's CUDA transport (libuct_cuda) was not loaded"
            result["reason"] = (f"VRAM registration failed ({', '.join(failed)}): "
                                f"{result['registrations'][failed[0]]['error']}{hint}")[:300]
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
