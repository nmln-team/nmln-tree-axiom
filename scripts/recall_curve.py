"""Recall-vs-samples curves from an MLflow file store, no mlflow import needed.

Reads val/recall/by_gen_count_inc_invalid, whose MLflow `step` is the number of
samples generated so far, so each run already holds its full recall curve. Runs
are grouped by (size, arm), interpolated onto a common sample grid, and averaged
over seeds. Writes a CSV always; writes a figure if matplotlib is importable
(copy the CSV or the mlruns dir to a machine that has it otherwise).

    python scripts/recall_curve.py mergertrees/mlruns --tag 3M --out recall_curve_3M
"""
import argparse, csv, pathlib, re, statistics, collections

ap = argparse.ArgumentParser()
ap.add_argument("mlruns")
ap.add_argument("--tag", default="", help="only runs whose name ends with -<tag>; empty = untagged runs only")
ap.add_argument("--domain", default="mergertrees")
ap.add_argument("--points", type=int, default=60)
ap.add_argument("--out", default="recall_curve")
a = ap.parse_args()

series = collections.defaultdict(list)          # (size, arm) -> list of [(samples, recall), ...]
for f in pathlib.Path(a.mlruns).rglob("metrics/val/recall/by_gen_count_inc_invalid"):
    run = f.parents[3]
    nf = run / "tags" / "mlflow.runName"
    name = nf.read_text().strip() if nf.exists() else run.name
    m = re.search(a.domain + r"(\d+)", name)
    if not m: continue
    tagged = bool(re.search(r"-seed=\d+-(.+)$", name))
    if a.tag and not name.endswith("-" + a.tag): continue
    if not a.tag and tagged: continue
    arm = "no-tree" if "no-tree" in name else "matrix_tree" if "matrix_tree" in name else "wilson"
    pts = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            pts.append((int(parts[2]), float(parts[1])))
    if pts:
        pts.sort()
        series[(int(m.group(1)), arm)].append(pts)

def interp(pts, x):
    if x <= pts[0][0]: return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]

rows = []
for (size, arm), runs in sorted(series.items()):
    xmax = min(p[-1][0] for p in runs)
    grid = [round(xmax * i / a.points) for i in range(1, a.points + 1)]
    for x in grid:
        ys = [interp(p, x) for p in runs]
        rows.append((size, arm, x, statistics.mean(ys), statistics.stdev(ys) if len(ys) > 1 else 0.0, len(ys)))

csv_path = pathlib.Path(a.out + ".csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size", "arm", "samples", "recall_mean", "recall_sd", "seeds"]); w.writerows(rows)
print(f"{len(series)} (size, arm) cells, {sum(len(v) for v in series.values())} runs -> {csv_path}")
for (size, arm), runs in sorted(series.items()):
    print(f"  n={size:<3} {arm:<12} {len(runs)} seeds, final recall {statistics.mean(p[-1][1] for p in runs):.3f} at {min(p[-1][0] for p in runs):,} samples")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not importable here; plot the CSV elsewhere"); raise SystemExit
sizes = sorted({k[0] for k in series})
fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 3.8), squeeze=False)
style = {"no-tree": ("#9A6A4F", "--"), "wilson": ("#0F6E6B", "-"), "matrix_tree": ("#0F6E6B", ":")}
for ax, size in zip(axes[0], sizes):
    for arm in ("no-tree", "wilson", "matrix_tree"):
        r = [row for row in rows if row[0] == size and row[1] == arm]
        if not r: continue
        xs = [row[2] / 1e6 for row in r]; mu = [row[3] for row in r]; sd = [row[4] for row in r]
        c, ls = style[arm]
        ax.plot(xs, mu, ls, color=c, label=f"{ {'no-tree': 'NMLN without tree axiom', 'wilson': 'with tree axiom (Wilson)', 'matrix_tree': 'with tree axiom (matrix-tree)'}[arm]} ({r[0][5]} seeds)")
        ax.fill_between(xs, [max(0.0, m - s) for m, s in zip(mu, sd)], [min(1.0, m + s) for m, s in zip(mu, sd)], color=c, alpha=0.15, lw=0)
    ax.set_ylim(bottom=0); ax.set_title(f"n = {size}"); ax.set_xlabel("samples generated (millions)"); ax.set_ylabel("recall"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=150)
print(f"figure -> {a.out}.pdf / .png")
