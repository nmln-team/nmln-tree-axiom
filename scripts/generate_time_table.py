"""Aggregate runtime across seeded repetitions into a LaTeX table and a plot.

MLJ major revision, Reviewer 2: the previous version of this script kept a
single run per (atoms, method) cell -- duplicate runs silently overwrote each
other -- and reported total SLURM job wall time, which includes training,
validation, RDKit sanitisation and MLflow logging rather than sampling alone.
Both problems are fixed here:

  * every matching run is collected, and we report mean +/- std over seeds
    together with the number of repetitions actually found;
  * two timing columns are produced where available -- total job wall time
    (comparable to the old table) and cumulative sampling time, logged by
    `nmln.model.Model.sample` as `val/cumulative_sampling_seconds`.

Runs predating that metric simply have no sampling time; those cells report
"n/a" instead of being silently dropped.

Usage
-----
    python generate_time_table.py --tracking-uri mlruns --out-dir paper_tables
    python generate_time_table.py --metric sampling      # sampling-time table
    python generate_time_table.py --no-plot              # table only
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

import mlflow

METHOD_ORDER = ["original", "wilson", "matrix_tree"]
METHOD_LABELS = {
    "original": "No tree axiom",
    "wilson": "Wilson sampler",
    "matrix_tree": "Matrix-tree sampler",
}


def collect(
    tracking_uri: str, num_chains: int, min_samples: int, name_contains: str
) -> dict:
    """Return runs[atoms][method] = list of per-run timing dicts."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    runs: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for exp in client.search_experiments():
        for run in client.search_runs(exp.experiment_id, max_results=50_000):
            m, p = run.data.metrics, run.data.params
            # Without this filter the older single-seed runs (also 20 chains,
            # also seed 22) are pooled with the new grid and seed 22 is counted
            # twice.
            if name_contains:
                run_name = run.data.tags.get("mlflow.runName", "") or ""
                if name_contains not in run_name and name_contains not in p.get(
                    "name", ""
                ):
                    continue
            if m.get("val/n_generated_samples_inc_invalid", 0) < min_samples:
                continue
            if int(p.get("num_chains", 0)) != num_chains:
                continue
            start, end = run.info.start_time, run.info.end_time
            if not (start and end):
                continue
            tree = p.get("tree_constraint", "?")
            method = p.get("sampler_type", "original") if tree == "True" else "original"
            atoms = int(p.get("max_num_atoms", 0))
            if not atoms:
                continue
            runs[atoms][method].append(
                dict(
                    seed=p.get("seed", "?"),
                    run_id=run.info.run_id,
                    wall_h=(end - start) / 1000 / 3600,
                    sampling_h=(
                        m["val/cumulative_sampling_seconds"] / 3600
                        if "val/cumulative_sampling_seconds" in m
                        else None
                    ),
                )
            )
    return runs


def summarize(values: list[float]) -> tuple[float, float, int]:
    """Return (mean, sample std, n). Std is 0 when a single run is available."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return mean, std, n


def cell_values(entries: list[dict], metric: str) -> list[float]:
    key = "sampling_h" if metric == "sampling" else "wall_h"
    return [e[key] for e in entries if e[key] is not None]


def fmt_cell(mean: float, std: float, n: int) -> str:
    if n == 0:
        return "n/a"
    if n == 1:
        # Be explicit that this cell has no repetitions behind it.
        return f"{mean:.2f} (n=1)"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--num-chains", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=2_000_000)
    parser.add_argument(
        "--metric",
        choices=["wall", "sampling"],
        default="wall",
        help="'wall' = total job time (old behaviour), 'sampling' = sampler only",
    )
    parser.add_argument(
        "--name-contains",
        default="mlj-",
        help="only aggregate runs whose name contains this string; pass an "
        "empty string to include every run (this will pool the old "
        "single-seed runs with the new grid)",
    )
    parser.add_argument("--out-dir", default="paper_tables")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    runs = collect(
        args.tracking_uri, args.num_chains, args.min_samples, args.name_contains
    )
    if not runs:
        print(
            f"No runs matched (tracking_uri={args.tracking_uri}, "
            f"num_chains={args.num_chains}, min_samples={args.min_samples})."
        )
        return

    os.makedirs(args.out_dir, exist_ok=True)
    atoms_sorted = sorted(runs)

    # ---- console table + per-run CSV -------------------------------------
    header = f"{'Atoms':<7}" + "".join(f"{METHOD_LABELS[m]:<26}" for m in METHOD_ORDER)
    print(f"\nMetric: {args.metric} time in hours (mean +/- std over seeds)\n")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for atoms in atoms_sorted:
        line = f"{atoms:<7}"
        for method in METHOD_ORDER:
            vals = cell_values(runs[atoms].get(method, []), args.metric)
            mean, std, n = summarize(vals)
            cell = "n/a" if n == 0 else f"{mean:.2f} +/- {std:.2f} (n={n})"
            line += f"{cell:<26}"
            summary_rows.append(
                dict(
                    atoms=atoms,
                    method=method,
                    metric=args.metric,
                    n_runs=n,
                    mean_h=f"{mean:.4f}" if n else "",
                    std_h=f"{std:.4f}" if n else "",
                )
            )
        print(line)

    per_run_path = os.path.join(args.out_dir, f"runtime_per_run_{args.metric}.csv")
    with open(per_run_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["atoms", "method", "seed", "run_id", "wall_h", "sampling_h"]
        )
        w.writeheader()
        for atoms in atoms_sorted:
            for method in METHOD_ORDER:
                for e in runs[atoms].get(method, []):
                    w.writerow(
                        dict(
                            atoms=atoms,
                            method=method,
                            seed=e["seed"],
                            run_id=e["run_id"],
                            wall_h=f"{e['wall_h']:.4f}",
                            sampling_h=(
                                f"{e['sampling_h']:.4f}"
                                if e["sampling_h"] is not None
                                else ""
                            ),
                        )
                    )

    summary_path = os.path.join(args.out_dir, f"runtime_summary_{args.metric}.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["atoms", "method", "metric", "n_runs", "mean_h", "std_h"]
        )
        w.writeheader()
        w.writerows(summary_rows)

    # ---- LaTeX table -----------------------------------------------------
    what = "sampling" if args.metric == "sampling" else "wall-clock"
    caption = (
        f"Mean {what} time in hours to generate 2M samples, over independent "
        f"seeds ($\\pm$ one standard deviation), with "
        f"{args.num_chains} parallel chains."
    )
    latex = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:runtime_{args.metric}}}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Atoms} & "
        + " & ".join(f"\\textbf{{{METHOD_LABELS[m]}}}" for m in METHOD_ORDER)
        + " \\\\",
        "\\hline",
    ]
    for atoms in atoms_sorted:
        cells = []
        for method in METHOD_ORDER:
            mean, std, n = summarize(cell_values(runs[atoms].get(method, []), args.metric))
            cells.append(fmt_cell(mean, std, n))
        latex.append(f"{atoms} & " + " & ".join(cells) + " \\\\")
    latex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    latex_str = "\n".join(latex)

    latex_path = os.path.join(args.out_dir, f"runtime_table_{args.metric}.tex")
    with open(latex_path, "w") as fh:
        fh.write(latex_str + "\n")
    print("\n--- LaTeX ---\n")
    print(latex_str)

    # ---- runtime-vs-size plot with error bars -----------------------------
    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            markers = {"original": "o", "wilson": "s", "matrix_tree": "^"}
            for method in METHOD_ORDER:
                xs, ys, es = [], [], []
                for atoms in atoms_sorted:
                    mean, std, n = summarize(
                        cell_values(runs[atoms].get(method, []), args.metric)
                    )
                    if n:
                        xs.append(atoms)
                        ys.append(mean)
                        es.append(std)
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=es,
                        marker=markers[method],
                        capsize=3,
                        label=METHOD_LABELS[method],
                    )
            ax.set_xlabel("Number of heavy atoms")
            ax.set_ylabel(f"{what.capitalize()} time (hours)")
            # no in-figure title: the journal wants titles in the caption only
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for ext in ("eps", "pdf", "png"):
                fig.savefig(
                    os.path.join(args.out_dir, f"runtime_vs_atoms_{args.metric}.{ext}"),
                    dpi=200,
                )
            plt.close(fig)
            print(f"\nPlot written to {args.out_dir}/runtime_vs_atoms_{args.metric}.*")
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            print(f"\nPlot skipped ({exc}).")

    print(f"\nPer-run CSV : {per_run_path}")
    print(f"Summary CSV : {summary_path}")
    print(f"LaTeX table : {latex_path}")


if __name__ == "__main__":
    main()
