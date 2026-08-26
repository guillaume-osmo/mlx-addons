"""KDE "stylized facts of returns" — MLX (Apple GPU) port of the cuML demo.

Reproduces https://github.com/will-hill/GPU-Quant-Finance 03-kde-stylized-facts,
swapping NVIDIA cuML for ``mlx_addons.neighbors.KernelDensity`` on the Apple GPU.

Three outputs, all written next to this file under ``images/``:

1. ``mlx_intraday_density_ratio.png`` — empirical return density (KDE + bootstrap
   band) vs. the matched normal, with the empirical/normal ratio panel. The
   stylized fact: real intraday returns are leptokurtic with heavy tails.
2. ``mlx_kde_benchmark.png`` — CPU (sklearn) vs Apple-GPU (MLX) bootstrap-fit time
   across sample sizes, with per-size speedup.
3. Console: a peak-memory comparison of the naive full ``(grid x samples)`` kernel
   matrix vs. the tiled MLX reduction — the "minimize RAM" headline.

Data: reads ``intraday_returns_5m.parquet`` (columns ``ticker, ret``) if present
next to this file or under ``../data``; otherwise synthesizes a heavy-tailed pool
so the demo is self-contained.

Run::

    python benchmarks/bench_kde_stylized_facts.py            # quick (~1 min)
    python benchmarks/bench_kde_stylized_facts.py --full     # notebook-scale
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlx_addons.neighbors import KernelDensity, bootstrap_kde  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG = HERE / "images"
GREEN, BLUE, RED, AMBER = "#8ad07a", "#5eaeff", "#ff5e5e", "#ffb84d"  # MLX-teal / blue / red / amber

# Fixed evaluation grid: +/- 2% in 5-minute return space (matches the notebook).
GRID = np.linspace(-0.02, 0.02, 5000).reshape(-1, 1).astype(np.float32)
KDE_BANDWIDTH = 0.0002


# --------------------------------------------------------------------- data
def load_returns() -> np.ndarray:
    """Pooled 5-minute returns as a 1-D float32 array (real file or synthetic)."""
    for cand in ("intraday_returns_5m.parquet", "../data/intraday_returns_5m.parquet"):
        p = HERE / cand
        if p.exists():
            import pandas as pd

            df = pd.read_parquet(p)
            print(f"loaded {len(df):,} real returns from {p}")
            return df["ret"].to_numpy(np.float32)
    return synth_returns()


def synth_returns(n_tickers: int = 100, bars_per_ticker: int = 6600, seed: int = 0) -> np.ndarray:
    """Heavy-tailed synthetic 5-minute returns (Student-t core, per-ticker vol).

    Leptokurtic with heavy tails by construction, so the empirical-vs-normal story
    holds. Pre-filtered to ``|r| <= 5%`` like the reference pipeline's bad-tick drop.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_tickers):
        dof = rng.uniform(3.5, 6.0)                      # heavier tails at low dof
        vol = rng.uniform(0.0008, 0.0025)                # per-ticker 5-min vol
        r = rng.standard_t(dof, size=bars_per_ticker) * vol
        out.append(r.astype(np.float32))
    pooled = np.concatenate(out)
    pooled = pooled[np.abs(pooled) <= 0.05]              # drop bad ticks, as in the demo
    print(f"synthesized {len(pooled):,} returns from {n_tickers} tickers (heavy-tailed)")
    return pooled


# --------------------------------------------------------- section 3 figure
def figure_density_vs_normal(returns: np.ndarray, n_boot: int) -> dict:
    from scipy.stats import kurtosis, norm

    run = bootstrap_kde(
        returns, GRID, n_samples=min(50_000, len(returns)),
        n_boot=n_boot, bandwidth=KDE_BANDWIDTH, seed=0, query_tile=5000, sample_tile=8192,
    )
    g_pct = GRID.ravel() * 100
    mu, sigma = float(returns.mean()), float(returns.std())
    gaussian = norm.pdf(GRID.ravel(), loc=mu, scale=sigma)
    mean, lo, hi = run["mean"], run["lo"], run["hi"]
    ratio = mean / np.clip(gaussian, 1e-300, None)

    thresh = 0.01
    p_emp = float((np.abs(returns) > thresh).mean())
    p_norm = norm.sf(thresh, mu, sigma) + norm.cdf(-thresh, mu, sigma)
    tail_mult = p_emp / p_norm

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.fill_between(g_pct, lo, hi, color=BLUE, alpha=0.30, label=f"5-95% bootstrap ({n_boot} refits)")
    ax1.plot(g_pct, mean, color=BLUE, lw=2.5, label="Empirical KDE (MLX / Apple GPU)")
    ax1.plot(g_pct, gaussian, color=RED, lw=2, ls="--", label="Normal (matched sigma)")
    ax1.set_yscale("log")
    ax1.set_ylim(0.01, mean.max() * 2)
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", framealpha=0.2)
    ax1.set_title("5-minute returns vs. the textbook normal model", fontsize=14, pad=12)
    ax1.grid(alpha=0.15)

    ax2.axhline(1, color="white", lw=1, alpha=0.4)
    ax2.plot(g_pct, ratio, color=AMBER, lw=2.5)
    ax2.fill_between(g_pct, 1, ratio, where=(ratio > 1), color=AMBER, alpha=0.3, label="more frequent than normal")
    ax2.fill_between(g_pct, 1, ratio, where=(ratio < 1), color=BLUE, alpha=0.3, label="less frequent than normal")
    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 1e4)
    ax2.set_ylabel("Empirical / Normal")
    ax2.set_xlabel("5-minute return (%)")
    ax2.legend(loc="upper center", framealpha=0.2, ncol=2)
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / "mlx_intraday_density_ratio.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    exc_kurt = float(kurtosis(returns, fisher=True))
    print(f"\n  KDE bootstrap band: {run['fit_seconds']:.2f}s on Apple GPU "
          f"({run['n_boot']} fits x {run['n_samples']:,} samples)")
    print(f"  Excess kurtosis: {exc_kurt:.1f}   (normal = 0)")
    print(f"  |r| > 1% moves happen {tail_mult:,.0f}x more often than the matched normal predicts.")
    print(f"  saved {out}")
    return run


# --------------------------------------------------- section 7 benchmark
def _sklearn_bootstrap_seconds(pool: np.ndarray, n_samples: int, n_boot: int) -> float:
    from sklearn.neighbors import KernelDensity as SkKDE

    t0 = time.perf_counter()
    for i in range(n_boot):
        rng = np.random.default_rng(i)
        s = rng.choice(pool, size=n_samples, replace=True).reshape(-1, 1)
        np.exp(SkKDE(bandwidth=KDE_BANDWIDTH, kernel="gaussian").fit(s).score_samples(GRID))
    return time.perf_counter() - t0


def benchmark_sweep(pool: np.ndarray, scales, n_boot_full: int, cpu_boot: int) -> dict:
    print(f"\nBenchmark sweep — CPU (sklearn) timed at {cpu_boot} boots, scaled to "
          f"{n_boot_full}; MLX runs the full {n_boot_full}.")
    cpu_s, gpu_s = {}, {}
    for n in scales:
        n = min(n, len(pool))
        cpu_measured = _sklearn_bootstrap_seconds(pool, n, cpu_boot)
        cpu_s[n] = cpu_measured * (n_boot_full / cpu_boot)               # linear-in-n_boot
        gpu_s[n] = bootstrap_kde(
            pool, GRID, n_samples=n, n_boot=n_boot_full, bandwidth=KDE_BANDWIDTH,
            query_tile=5000, sample_tile=8192,
        )["fit_seconds"]
        print(f"  n={n:>7,}:  CPU {cpu_s[n]:8.2f}s   MLX {gpu_s[n]:7.3f}s   "
              f"-> {cpu_s[n]/gpu_s[n]:6.0f}x")

    labels = [f"{n//1000}K" for n in scales]
    cpu_t = np.array([cpu_s[min(n, len(pool))] for n in scales])
    gpu_t = np.array([gpu_s[min(n, len(pool))] for n in scales])
    speed = cpu_t / gpu_t

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
    x = np.arange(len(scales)); w = 0.38
    ax.bar(x - w / 2, cpu_t, w, color=BLUE, label="CPU (sklearn)")
    ax.bar(x + w / 2, gpu_t, w, color=GREEN, label="Apple GPU (MLX)")
    for i, (c, s) in enumerate(zip(cpu_t, speed)):
        ax.text(i, c, f"{s:.0f}x", ha="center", va="bottom", fontsize=24, fontweight="bold", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\nsamples" for l in labels], fontsize=14)
    ax.set_ylabel("Bootstrap fit time (seconds)", fontsize=14)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.text(0.5, 0.96, "Gaussian KDE bootstrap — sklearn CPU vs MLX Apple GPU",
             ha="center", fontsize=17, fontweight="bold", color=GREEN)
    fig.text(0.5, 0.925, f"100 tickers · {n_boot_full} bootstraps · tiled, low-RAM",
             ha="center", fontsize=12, color="white")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / "mlx_kde_benchmark.png"
    plt.savefig(out, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return {"scales": list(scales), "cpu_s": cpu_t.tolist(), "gpu_s": gpu_t.tolist()}


# --------------------------------------------------------- RAM comparison
def _naive_full_matrix_density(sample: np.ndarray, bw: float) -> mx.array:
    """The naive path a GPU library takes: materialise the full (G, N) kernel matrix."""
    D = mx.array(sample.reshape(-1))
    G = mx.array(GRID.reshape(-1))
    diff = G[:, None] - D[None, :]                       # (G, N) <-- the whole matrix at once
    kmat = mx.exp(-0.5 * (diff / bw) ** 2)
    dens = mx.sum(kmat, axis=1) * (1.0 / (len(D) * bw * math.sqrt(2 * math.pi)))
    mx.eval(dens)
    return dens


def ram_comparison(pool: np.ndarray, n_ram: int) -> None:
    n_ram = min(n_ram, len(pool))
    sample = np.random.default_rng(0).choice(pool, size=n_ram, replace=True).astype(np.float32)
    g = GRID.shape[0]
    print(f"\nPeak-memory comparison — one Gaussian KDE, grid={g:,} x samples={n_ram:,}")
    naive_matrix_mb = g * n_ram * 4 / 1e6
    print(f"  full (grid x samples) matrix would be: {naive_matrix_mb:,.0f} MB of float32")

    mx.clear_cache(); mx.reset_peak_memory()
    dens_tiled = KernelDensity(bandwidth=KDE_BANDWIDTH, query_tile=2048, sample_tile=8192).fit(
        sample.reshape(-1, 1)
    ).eval_density(GRID)
    peak_tiled = mx.get_peak_memory() / 1e6

    # The naive path needs the (G, N) matrix AND its exp() live at once (~2x the matrix).
    # Skip materialising it when that would risk an OOM; report the requirement instead.
    if 2 * naive_matrix_mb > 3000:
        print(f"  naive full-matrix  peak GPU memory: ~{2*naive_matrix_mb:,.0f} MB "
              "(not run — would risk OOM)")
        print(f"  tiled (2048x8192)  peak GPU memory: {peak_tiled:8.1f} MB   "
              f"-> ~{2*naive_matrix_mb/peak_tiled:.0f}x less (peak is flat in N)")
        return

    mx.clear_cache(); mx.reset_peak_memory()
    dens_naive = _naive_full_matrix_density(sample, KDE_BANDWIDTH)
    peak_naive = mx.get_peak_memory() / 1e6
    rel = float(np.max(np.abs(np.asarray(dens_naive) - dens_tiled)) / dens_tiled.max())
    print(f"  naive full-matrix  peak GPU memory: {peak_naive:8.1f} MB")
    print(f"  tiled (2048x8192)  peak GPU memory: {peak_tiled:8.1f} MB   "
          f"-> {peak_naive/peak_tiled:.0f}x less")
    print(f"  (curves identical: max rel diff {rel:.2e})")


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="notebook-scale sweep (slower)")
    ap.add_argument("--ram-n", type=int, default=50_000, help="samples for the RAM comparison")
    args = ap.parse_args()

    if args.full:
        scales, n_boot_full, cpu_boot, fig_boot = [10_000, 50_000, 100_000, 200_000], 1000, 20, 200
    else:
        scales, n_boot_full, cpu_boot, fig_boot = [10_000, 50_000, 100_000], 100, 8, 100

    print("=" * 74)
    print("  KDE stylized facts — MLX (Apple GPU) port of the cuML demo")
    print("=" * 74)
    pool = load_returns()

    figure_density_vs_normal(pool, n_boot=fig_boot)
    benchmark_sweep(pool, scales, n_boot_full=n_boot_full, cpu_boot=cpu_boot)
    ram_comparison(pool, n_ram=args.ram_n)
    print("\ndone.")


if __name__ == "__main__":
    main()
