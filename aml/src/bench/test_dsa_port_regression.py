"""Regression test for our fix on top of vLLM PR #38476 (dsa_port.json): the paged MQA-logits decode kernel
must not write past its logits row when the width (the batch max_seq_len in eager steps) is not a multiple of
64. Runs on the GPU node after the overlay; tests/test_dsa_port_fix.py runs it on CPU (Triton interpreter)."""
import unittest

try:
    import torch
except ImportError:
    torch = None

B = 64
# (context lengths, next_n, logits width, clean_logits); unpatched, the first spills into row 1, the second
# past the tensor; 5056 is a multiple of 64 like FULL CUDA graph widths.
CASES = [((5000, 4000), 1, 5000, False), ((4000, 5000), 1, 5000, False), ((5000, 3000), 2, 5000, False),
         ((5000, 4000), 1, 5056, True)]


class GuardedTorch:
    """`torch` for the module: each logits row is followed by `pad` NaN columns, the last by a NaN tail."""

    def __init__(self, pad):
        self.pad = pad

    def __getattr__(self, name):
        return getattr(torch, name)

    def full(self, shape, fill, dtype=None, device=None):
        rows, width = shape
        pitch = width + self.pad
        self.buf = torch.full((rows * pitch + B,), float("nan"), dtype=dtype, device=device)
        self.inside = torch.zeros(self.buf.shape, dtype=torch.bool)
        self.inside[: rows * pitch].view(rows, pitch)[:, :width] = True
        return self.buf[: rows * pitch].view(rows, pitch)[:, :width].fill_(fill)

    def empty(self, shape, dtype=None, device=None):
        return self.full(shape, float("nan"), dtype, device)


def run(mod, case, pad, device, heads=64, dim=128):
    """Calls the wrapper the way sparse_attn_indexer.py does; returns CPU tensors."""
    lens, next_n, width, clean = case
    g = torch.Generator().manual_seed(0)
    nblk = -(-max(lens) // B) + 2
    kv = torch.randn(len(lens) * nblk, B, dim, generator=g)
    scale = kv.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    k8 = (kv / scale).to(torch.float8_e4m3fn)
    cache = torch.cat([k8.view(torch.uint8).flatten(1), scale.flatten(1).view(torch.uint8)], 1)
    q = torch.randn(len(lens), next_n, heads, dim, generator=g).to(torch.float8_e4m3fn)
    w = torch.randn(len(lens) * next_n, heads, generator=g)
    table = torch.randperm(len(lens) * nblk, generator=g).to(torch.int32).view(len(lens), nblk)
    guarded, real = GuardedTorch(pad), mod.torch
    mod.torch = guarded
    try:
        out = mod.fp8_paged_mqa_logits_triton(
            q.to(device), cache.view(-1, B, 1, dim + 4).to(device), w.to(device),
            torch.tensor(lens, dtype=torch.int32, device=device), table.to(device),
            max_model_len=width, clean_logits=clean)
    finally:
        mod.torch = real
    return (q, k8, scale, w, table), out.cpu(), guarded.buf.cpu(), guarded.inside


def stray_writes(buf, inside):
    return int((~torch.isnan(buf) & ~inside).sum())


def check(args, out, buf, inside, case):
    """Nothing written outside the logits rows; in-row values match a float64 reference."""
    lens, next_n, width, clean = case
    assert stray_writes(buf, inside) == 0, f"{case}: {stray_writes(buf, inside)} floats outside the rows"
    q, k8, scale, w, table = args
    for t in range(out.shape[0]):
        i, j = divmod(t, next_n)
        n = min(width, -(-lens[i] // B) * B)  # columns of the row's active blocks
        k = k8[table[i].long()].double().flatten(0, 1)[:n]
        s = scale[table[i].long()].double().flatten()[:n]
        ref = ((q[i, j].double() @ k.T * s).relu() * w[t].double()[:, None]).sum(0)
        pos = torch.arange(n)
        ref = ref.masked_fill((pos >= lens[i]) | (pos > lens[i] - next_n + j), float("-inf"))
        torch.testing.assert_close(out[t, :n].double(), ref, rtol=1e-3, atol=1e-2)
        tail = out[t, n:]
        assert (tail.isneginf() if clean else tail.isnan()).all(), f"{case}: row {t} written past its blocks"


def test_paged_logits_stay_in_row():
    if torch is None or not torch.cuda.is_available():
        raise unittest.SkipTest("needs a CUDA GPU")
    from vllm.v1.attention.ops import mqa_logits_triton as mod
    for case in CASES:
        for pad in (0, B):
            check(*run(mod, case, pad, "cuda"), case)
