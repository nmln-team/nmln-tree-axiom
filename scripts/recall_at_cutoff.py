"""Recall-at-cutoff curves for the molecule domain, aggregated over seeds.

This is the Fig1 protocol of the paper: rank a run's unique generated molecules
by frequency and, for each prefix length t, report the fraction of validation
molecules among the top t. It replaces the single-seed
scripts/mlflow_metrics_rci.py of the archive with a five-seed version.

Needs no RDKit and no mlflow: the model writes canonical SMILES, one per line,
to all_generated_smiles.in-training.txt (invalid samples appear as
DUMMY_INVALID_SMILE), and data/molecules/tree<N>/val_smiles.txt is already
canonical. Stdlib only, so it runs on a cluster login node. matplotlib is used for
the figure when importable; otherwise copy the CSV home and run again on it.

    python scripts/recall_at_cutoff.py --experiments exact_rerun/experiments --out ~/recall_cutoff
    python scripts/recall_at_cutoff.py --from-csv ~/recall_cutoff.csv --out ~/recall_cutoff   # plot only
"""
import argparse, csv, pathlib, re, statistics, collections, math

INVALID = "DUMMY_INVALID_SMILE"
LABELS = {"original": "NMLN without tree axiom", "wilson": "NMLN with tree axiom (Wilson sampler)",
          "matrix_tree": "NMLN with tree axiom (matrix-tree sampler)"}
STYLE = {"original": ("#2166AC", "-"), "wilson": ("#D6604D", "--"), "matrix_tree": ("#4DAC26", "-.")}

ap = argparse.ArgumentParser()
ap.add_argument("--experiments", default="exact_rerun/experiments")
ap.add_argument("--domain", default="molecules", help="molecules, or a FOL domain prefix such as mergertrees / glycans")
ap.add_argument("--tag", default="", help="FOL domains: only runs whose name ends with -<tag>, e.g. 3M")
ap.add_argument("--data", default="data/molecules")
ap.add_argument("--sizes", default="8,9,10,11,12,13,14,15")
ap.add_argument("--budget", type=int, default=2_000_000, help="samples read per run")
ap.add_argument("--points", type=int, default=80, help="x-grid points per panel")
ap.add_argument("--out", default="recall_cutoff")
ap.add_argument("--from-csv", default="", help="skip computation; plot this CSV")
ap.add_argument("--arms", default="original,wilson", help="comma list; the paper figure shows baseline and Wilson, as the original submission did")
ap.add_argument("--style", default="seeds", choices=["seeds", "band"], help="'seeds' = thin per-seed lines + bold mean (default); 'band' = mean +/- sd")
ap.add_argument("--suptitle", action="store_true", help="put a title line inside the figure (off: journal artwork rules)")
ap.add_argument("--ylim", default="auto", help="'auto' = per-panel range (as in the original Fig1), or a number such as 1 for a common top")
a = ap.parse_args()
ARMS = [(k, LABELS[k]) for k in a.arms.split(",")]


def curve(path, val, budget):
    """Return (n_unique_valid, [(t, recall)...]) for one run."""
    cnt = collections.Counter()
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= budget: break
            s = line.strip()
            if s and s != INVALID: cnt[s] += 1
    ranked = [s for s, _ in cnt.most_common()]
    hits, pts = 0, []
    for t, s in enumerate(ranked, start=1):
        if s in val: hits += 1
        pts.append((t, hits / len(val)))
    return len(ranked), pts


def at(pts, t):
    """Recall of the top-t prefix; a step function, constant past the end."""
    if not pts: return 0.0
    if t >= pts[-1][0]: return pts[-1][1]
    return pts[max(0, min(len(pts), t) - 1)][1]


rows, seed_rows = [], []
if a.from_csv:
    with open(a.from_csv) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["size"]), r["arm"], int(r["t"]), float(r["recall_mean"]), float(r["recall_sd"]), int(r["seeds"])))
    sp = pathlib.Path(a.from_csv.replace(".csv", "_seeds.csv"))
    if sp.exists():
        with open(sp) as fh:
            for r in csv.DictReader(fh):
                seed_rows.append((int(r["size"]), r["arm"], r["seed"], int(r["t"]), float(r["recall"])))
else:
    exp = pathlib.Path(a.experiments)
    for n in [int(x) for x in a.sizes.split(",")]:
        if a.domain == "molecules":
            val = set(pathlib.Path(a.data, f"tree{n}", "val_smiles.txt").read_text().split())
        else:
            # a FOL world's key is its sorted fact string, exactly what the model
            # writes per sample for non-molecular domains
            val_entries = ["".join(sorted(l.strip() for l in f.read_text().splitlines() if l.strip()))
                           for f in pathlib.Path(a.data, f"{a.domain}{n}", "val").glob("*.mol")]
            val = set(val_entries)
            if len(val) < len(val_entries):
                print(f"n={n}: validation set has {len(val_entries)} files but {len(val)} distinct worlds; "
                      f"recall below is over DISTINCT worlds (MLflow's logged value counts each file)")
        for arm, _ in ARMS:
            if a.domain == "molecules":
                runs = sorted(exp.glob(f"mlj-{arm}-20chain-{n}atoms-seed*"))
            else:
                pat = {"original": "no-tree", "wilson": "sampler=wilson", "matrix_tree": "sampler=matrix_tree"}[arm]
                runs = sorted(r for r in exp.glob(f"mlj-{a.domain}{n}-{pat}-seed=*")
                              if (r.name.endswith("-" + a.tag) if a.tag else not re.search(r"-seed=\d+-.+$", r.name)))
            curves, finals = [], []
            for rd in runs:
                f = rd / "all_generated_smiles.in-training.txt"
                if not f.exists(): continue
                nu, pts = curve(f, val, a.budget)
                if not pts:          # run still starting, or an empty file
                    print(f"  skipping {rd.name}: no samples yet"); continue
                curves.append(pts); finals.append((rd.name, nu, pts[-1][1]))
            if not curves: continue
            xmax = max(p[-1][0] for p in curves)
            # geometric grid so the early, steep part of the curve is resolved
            grid = sorted({max(1, round(xmax ** (i / (a.points - 1)))) for i in range(a.points)} | {xmax})
            for t in grid:
                ys = [at(p, t) for p in curves]
                rows.append((n, arm, t, statistics.mean(ys), statistics.stdev(ys) if len(ys) > 1 else 0.0, len(ys)))
                for (name, _, _), y in zip(finals, ys):
                    seed_rows.append((n, arm, re.search(r"seed=?(\d+)", name).group(1), t, y))
            print(f"n={n:<3} {arm:<12} {len(curves)} seeds  final recall "
                  f"{statistics.mean(x[2] for x in finals):.3f} ± {statistics.stdev([x[2] for x in finals]) if len(finals) > 1 else 0:.3f}"
                  f"  unique valid {statistics.mean(x[1] for x in finals):,.0f}")
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["size", "arm", "t", "recall_mean", "recall_sd", "seeds"]); w.writerows(rows)
    with open(a.out + "_seeds.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["size", "arm", "seed", "t", "recall"]); w.writerows(seed_rows)
    print(f"-> {a.out}.csv and {a.out}_seeds.csv")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.ticker as mt
except Exception:
    print("matplotlib not importable here; run again with --from-csv on a machine that has it"); raise SystemExit

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.linestyle": "--", "grid.alpha": .4, "lines.linewidth": 1.8})
sizes = sorted({r[0] for r in rows})
ncol = 4; nrow = math.ceil(len(sizes) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
for ax, n in zip(axes.flat, sizes):
    for arm, label in ARMS:
        r = [x for x in rows if x[0] == n and x[1] == arm]
        if not r: continue
        xs = [x[2] for x in r]; mu = [x[3] for x in r]; sd = [x[4] for x in r]
        c, ls = STYLE[arm]
        if a.style == "seeds" and seed_rows:
            for seed in sorted({x[2] for x in seed_rows if x[0] == n and x[1] == arm}):
                sr = [x for x in seed_rows if x[0] == n and x[1] == arm and x[2] == seed]
                ax.plot([x[3] for x in sr], [x[4] for x in sr], ls, color=c, lw=0.7, alpha=0.45)
            ax.plot(xs, mu, ls, color=c, lw=2.2, label=label)
        else:
            ax.plot(xs, mu, ls, color=c, label=label)
            ax.fill_between(xs, [max(0.0, m - s) for m, s in zip(mu, sd)], [min(1.0, m + s) for m, s in zip(mu, sd)], color=c, alpha=.18, lw=0)
    ax.set_xscale("log"); ax.set_title(f"{n} atoms" if a.domain == "molecules" else f"n = {n}", pad=4)
    if a.ylim == "auto":
        top = max([x[3] + x[4] for x in rows if x[0] == n] + [x[4] for x in seed_rows if x[0] == n])
        ax.set_ylim(0, max(top * 1.12, 0.02))
    else:
        ax.set_ylim(0, float(a.ylim) * 1.02)
    ax.set_xlabel("unique valid molecules, by frequency" if a.domain == "molecules" else "unique generated worlds, by frequency"); ax.set_ylabel("recall")
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}" if v < 1e3 else (f"{v/1e3:.0f}k" if v < 1e6 else f"{v/1e6:.1f}M")))
for ax in list(axes.flat)[len(sizes):]: ax.axis("off")
axes.flat[0].legend(loc="upper left", framealpha=.9, fontsize=8)
if a.suptitle:  # off by default: the journal wants no titles inside the artwork
    fig.suptitle("Recall at cutoff, seeds 22–26 (thin) and their mean (bold); 20 chains, 2M samples" if a.style == "seeds" and seed_rows else "Recall at cutoff, mean ± sd over seeds 22–26 (20 chains, 2M samples)", y=1.0)
fig.tight_layout()
for ext in ("pdf", "eps", "png"):
    fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=150)
print(f"-> {a.out}.pdf / .eps / .png")
