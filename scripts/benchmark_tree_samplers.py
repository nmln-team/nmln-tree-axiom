"""Scaling microbenchmark for the tree-block samplers (MLJ revision, Reviewer 2).

Times a single spanning-tree block update as a function of domain size n, which
is the quantity that governs whether the approach reaches large structures. The
NMLN model is not involved: we feed the samplers synthetic edge-weight matrices,
so the measurement isolates sampler cost from training/RDKit/MLflow overhead.

Two weight regimes are measured, because Wilson's expected runtime depends on
the *weighted* cover time of the graph and is therefore weight-sensitive, while
matrix-tree and Colbourn are not:

  uniform   w_ij = 1                     (best case for Wilson)
  lognormal w_ij = exp(sigma * N(0,1))   (heterogeneous, closer to learned weights)

Each (sampler, n, regime) cell is repeated --reps times; we report mean and
standard deviation. A sampler is dropped from all larger n once its mean time
per sample exceeds --time-limit seconds, and that is recorded explicitly as a
timeout rather than a missing value.

Usage
-----
    python scripts/benchmark_tree_samplers.py \
        --sizes 10,20,50,100,200,500,1000 \
        --reps 5 --time-limit 60 --out benchmark_results/tree_samplers.csv

CPU-only run (skips Colbourn's GPU path):
    python scripts/benchmark_tree_samplers.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time

import numpy as np
import torch

from nmln.sampler import (
    sampler as matrix_tree_sampler,
    wilson_method_fast,
)

SAMPLERS = ("wilson", "matrix_tree")
REGIMES = ("uniform", "lognormal")


def make_weights(n: int, regime: str, rng: np.random.Generator) -> np.ndarray:
    """Symmetric, non-negative, zero-diagonal weight matrix on n nodes."""
    if regime == "uniform":
        w = np.ones((n, n), dtype=np.float64)
    elif regime == "lognormal":
        w = np.exp(rng.normal(0.0, 1.0, size=(n, n)))
    else:
        raise ValueError(f"unknown regime: {regime}")
    w = np.triu(w, k=1)
    w = w + w.T
    return np.ascontiguousarray(w)


def draw_once(sampler_name: str, w: np.ndarray, device: str):
    """Draw one spanning tree and return its edge list."""
    if sampler_name == "wilson":
        edges, _, _ = wilson_method_fast(w, r=0)
        return edges
    if sampler_name == "matrix_tree":
        return matrix_tree_sampler(w, r=0)
    raise ValueError(f"unknown sampler: {sampler_name}")


# At small n a single draw finishes far below the clock resolution, so we repeat
# the draw until at least MIN_BLOCK_S has elapsed and divide by the count.
MIN_BLOCK_S = 0.05
MAX_INNER = 10_000


def time_one(sampler_name: str, w: np.ndarray, device: str) -> tuple[float, int]:
    """Return (seconds per sample, n_edges of the last draw)."""
    inner = 0
    edges = None
    start = time.perf_counter()
    while True:
        edges = draw_once(sampler_name, w, device)
        inner += 1
        elapsed = time.perf_counter() - start
        if elapsed >= MIN_BLOCK_S or inner >= MAX_INNER:
            break
    return elapsed / inner, len(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="10,20,50,100,200,500,1000",
        help="comma-separated domain sizes n",
    )
    parser.add_argument("--reps", type=int, default=5, help="repetitions per cell")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="seconds per sample above which a sampler is declared timed out "
        "and skipped for all larger n",
    )
    parser.add_argument(
        "--samplers",
        default=",".join(SAMPLERS),
        help=f"comma-separated subset of {SAMPLERS}",
    )
    parser.add_argument(
        "--regimes",
        default=",".join(REGIMES),
        help=f"comma-separated subset of {REGIMES}",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device for the Colbourn sampler",
    )
    parser.add_argument(
        "--out",
        default="benchmark_results/tree_samplers.csv",
        help="output CSV path (a .json sidecar with run metadata is also written)",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Warm up numba's JIT so compilation time does not land in the first cell.
    if "wilson" in samplers:
        wilson_method_fast(make_weights(8, "uniform", np.random.default_rng(0)), r=0)

    rows: list[dict] = []
    # A sampler that times out in one regime may still be viable in another,
    # so the timeout flag is tracked per (sampler, regime).
    timed_out: set[tuple[str, str]] = set()

    for regime in regimes:
        for n in sizes:
            for name in samplers:
                if (name, regime) in timed_out:
                    rows.append(
                        dict(
                            sampler=name,
                            regime=regime,
                            n=n,
                            reps=0,
                            mean_s="",
                            std_s="",
                            min_s="",
                            max_s="",
                            valid_frac="",
                            status="timeout",
                        )
                    )
                    print(f"{regime:10s} n={n:<5d} {name:12s} skipped (timed out earlier)")
                    continue

                rng = np.random.default_rng(args.seed + n)
                times: list[float] = []
                n_valid = 0
                ok = True
                for _ in range(args.reps):
                    w = make_weights(n, regime, rng)
                    try:
                        elapsed, n_edges = time_one(name, w, args.device)
                    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                        print(f"{regime:10s} n={n:<5d} {name:12s} FAILED: {exc}")
                        rows.append(
                            dict(
                                sampler=name,
                                regime=regime,
                                n=n,
                                reps=0,
                                mean_s="",
                                std_s="",
                                min_s="",
                                max_s="",
                                valid_frac="",
                                status=f"error: {exc}",
                            )
                        )
                        ok = False
                        break
                    if n_edges == n - 1:
                        n_valid += 1
                    else:
                        print(
                            f"  warning: {name} returned {n_edges} edges at n={n} "
                            f"(expected {n - 1}) -- not a spanning tree"
                        )
                    times.append(elapsed)
                    # Stop early if even one sample already blows the budget.
                    if elapsed > args.time_limit:
                        break
                if not ok:
                    continue

                mean = statistics.fmean(times)
                std = statistics.stdev(times) if len(times) > 1 else 0.0
                rows.append(
                    dict(
                        sampler=name,
                        regime=regime,
                        n=n,
                        reps=len(times),
                        mean_s=f"{mean:.6g}",
                        std_s=f"{std:.6g}",
                        min_s=f"{min(times):.6g}",
                        max_s=f"{max(times):.6g}",
                        valid_frac=f"{n_valid / len(times):.3f}",
                        status="ok",
                    )
                )
                print(
                    f"{regime:10s} n={n:<5d} {name:12s} "
                    f"{mean:.6f} +/- {std:.6f} s  (reps={len(times)}, "
                    f"valid trees {n_valid}/{len(times)})"
                )

                if mean > args.time_limit:
                    timed_out.add((name, regime))
                    print(
                        f"  -> {name} exceeds the {args.time_limit}s budget at n={n}; "
                        f"skipping larger n in regime '{regime}'"
                    )

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sampler",
                "regime",
                "n",
                "reps",
                "mean_s",
                "std_s",
                "min_s",
                "max_s",
                "valid_frac",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = dict(
        sizes=sizes,
        samplers=samplers,
        regimes=regimes,
        reps=args.reps,
        time_limit_s=args.time_limit,
        seed=args.seed,
        device=args.device,
        torch_version=torch.__version__,
        cuda_device=(
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
        ),
    )
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWrote {args.out} ({len(rows)} rows) and its .json metadata sidecar.")


if __name__ == "__main__":
    main()
