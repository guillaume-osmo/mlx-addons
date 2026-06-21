#!/usr/bin/env python3
"""Three-way linalg benchmark: CPU vs Torch+metal-linalg vs MLX-addons.

The main comparison is batched symmetric ``eigh`` because all three stacks have
a meaningful implementation:

- CPU: ``torch.linalg.eigh`` on CPU/Accelerate.
- Torch MPS: ``metal_linalg.eigh`` / custom Metal Jacobi kernels.
- MLX: ``mlx_addons.linalg.eigh_small_batch`` / custom MLX Metal kernel for
  ``n <= 32`` and MLX CPU-stream fallback above that.

Run from the repo root:

    PYTHONPATH=src python benchmarks/bench_linalg_threeway.py

If ``metal-linalg`` is not importable, pass a checkout path:

    PYTHONPATH=src python benchmarks/bench_linalg_threeway.py \
      --metal-linalg-path /tmp/metal-linalg-inspect
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from mlx_addons.linalg import JACOBI_MAX_N, eigh_small_batch


@dataclass(frozen=True)
class Timing:
    ms: float
    route: str
    max_abs_err: float | None = None


def _maybe_add_metal_linalg_path(path: str | None) -> None:
    if not path:
        default = Path("/tmp/metal-linalg-inspect")
        path = str(default) if default.exists() else None
    if path:
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _import_metal_linalg(path: str | None):
    _maybe_add_metal_linalg_path(path)
    try:
        return importlib.import_module("metal_linalg")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] metal_linalg unavailable: {exc}")
        return None


def _make_symmetric(batch: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((batch, n, n)).astype(np.float32)
    return 0.5 * (raw + np.swapaxes(raw, -1, -2))


def _sync_torch(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _eval_mlx(out):
    if isinstance(out, tuple):
        mx.eval(*out)
    else:
        mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def _median_ms(fn, *, kind: str, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        out = fn()
        if kind == "mlx":
            _eval_mlx(out)
        elif kind == "torch_mps":
            _sync_torch("mps")
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        if kind == "mlx":
            _eval_mlx(out)
        elif kind == "torch_mps":
            _sync_torch("mps")
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _sample_eigh_value_error(values, ref: np.ndarray, sample: int = 16) -> float:
    got = np.asarray(values)
    n = min(sample, got.shape[0])
    return float(np.max(np.abs(got[:n] - ref[:n])))


def bench_eigh_case(
    A_np: np.ndarray,
    metal_linalg,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Timing]:
    batch, n, _ = A_np.shape

    # CPU baseline. This is the same CPU framework used by metal-linalg's own
    # benchmark harness, and on macOS routes through Apple's CPU linear algebra.
    A_t_cpu = torch.from_numpy(A_np)
    w_cpu, _ = torch.linalg.eigh(A_t_cpu)
    cpu_ms = _median_ms(lambda: torch.linalg.eigh(A_t_cpu), kind="torch_cpu", warmup=warmup, repeat=repeat)
    w_ref = w_cpu.numpy()

    out: dict[str, Timing] = {
        "cpu_torch": Timing(cpu_ms, "torch.cpu", 0.0),
    }

    if metal_linalg is not None and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"):
        A_t_mps = A_t_cpu.to("mps")
        w_t, _ = metal_linalg.eigh(A_t_mps)
        _sync_torch("mps")
        torch_ms = _median_ms(lambda: metal_linalg.eigh(A_t_mps), kind="torch_mps", warmup=warmup, repeat=repeat)
        route = "metal-linalg.gpu" if n <= 64 else "metal-linalg.cpu-fallback"
        out["torch_mps_metal_linalg"] = Timing(torch_ms, route, _sample_eigh_value_error(w_t.cpu().numpy(), w_ref))

    A_mx = mx.array(A_np)
    w_mx, _ = eigh_small_batch(A_mx)
    _eval_mlx((w_mx,))
    mlx_ms = _median_ms(lambda: eigh_small_batch(A_mx), kind="mlx", warmup=warmup, repeat=repeat)
    route = "mlx-addons.gpu" if n <= JACOBI_MAX_N else "mlx.cpu-fallback"
    out["mlx_addons"] = Timing(mlx_ms, route, _sample_eigh_value_error(np.asarray(w_mx), w_ref))
    return out


def run_eigh(args, metal_linalg) -> None:
    cases = [(int(n), int(b)) for n, b in (part.split(":") for part in args.eigh_cases)]
    print("\n=== batched symmetric eigh, data resident ===")
    print(f"{'n':>4} {'batch':>7} {'method':>24} {'route':>24} {'ms':>10} {'vs_cpu':>9} {'max|dw|':>11}")
    print("-" * 98)
    for n, batch in cases:
        A_np = _make_symmetric(batch, n, seed=n + batch)
        result = bench_eigh_case(A_np, metal_linalg, warmup=args.warmup, repeat=args.repeat)
        cpu_ms = result["cpu_torch"].ms
        for name in ("cpu_torch", "torch_mps_metal_linalg", "mlx_addons"):
            if name not in result:
                continue
            t = result[name]
            speed = cpu_ms / t.ms if t.ms > 0 else float("nan")
            err = "" if t.max_abs_err is None else f"{t.max_abs_err:11.2e}"
            print(f"{n:4d} {batch:7d} {name:>24} {t.route:>24} {t.ms:10.2f} {speed:8.2f}x {err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metal-linalg-path",
        default=None,
        help="Path to a metal-linalg checkout. Defaults to /tmp/metal-linalg-inspect if present.",
    )
    parser.add_argument(
        "--eigh-cases",
        nargs="+",
        default=["3:16384", "8:16384", "16:8192", "32:2048", "48:2048", "64:1024"],
        help="Cases as n:batch.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    metal_linalg = _import_metal_linalg(args.metal_linalg_path)
    print(f"torch {torch.__version__} | mps={torch.backends.mps.is_available()} compile_shader={hasattr(torch.mps, 'compile_shader')}")
    print(f"mlx {mx.__version__} | default_device={mx.default_device()}")
    print(f"mlx-addons eigh GPU limit: n <= {JACOBI_MAX_N}")
    if metal_linalg is not None:
        from metal_linalg.kernels import BATCH_N_MAX

        print(f"metal-linalg eigh GPU limit: n <= {BATCH_N_MAX}")
    run_eigh(args, metal_linalg)


if __name__ == "__main__":
    main()
