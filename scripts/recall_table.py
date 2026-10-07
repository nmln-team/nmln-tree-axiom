"""Recall table from an MLflow file store, stdlib only.

Groups runs by (size, arm), reports mean +/- sd over the seeds that reached
their own sample budget, and lists unfinished runs with their progress. Needs
no mlflow import, so it runs on a cluster login node.

    python scripts/recall_table.py mergertrees/mlruns            # untagged runs only
    python scripts/recall_table.py mergertrees/mlruns --tag 3M   # runs named ...-seed=NN-3M
"""
import sys, pathlib, re, statistics, collections

args = sys.argv[1:]
tag = ""
if "--tag" in args:
    i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
root = pathlib.Path(args[0]).expanduser()


def last(p, idx=1):
    try:
        return float(p.read_text().strip().splitlines()[-1].split()[idx])
    except Exception:
        return None


best = {}
for f in root.rglob("metrics/val/recall/by_gen_count_inc_invalid"):
    run = f.parents[3]
    nf = run / "tags" / "mlflow.runName"
    name = nf.read_text().strip() if nf.exists() else run.name
    tagged = bool(re.search(r"-seed=\d+-.+$", name))
    if tag and not name.endswith("-" + tag):
        continue
    if not tag and tagged:
        continue
    rec = last(f)
    if rec is None:
        continue
    samp = last(run / "metrics" / "val" / "n_generated_samples_inc_invalid") or 0.0
    bp = run / "params" / "max_generated_samples"
    budget = float(bp.read_text().strip()) if bp.exists() else 2_000_000.0
    # a requeued task leaves a stub beside the real run; keep the furthest-along copy
    if name not in best or samp > best[name][1]:
        best[name] = (rec, samp, budget)

done, partial = collections.defaultdict(list), []
for name, (rec, samp, budget) in best.items():
    m = (re.search(r"(?:mergertrees|colortrees|glycans|blocks|taxonomy)(\d+)", name)
         or re.search(r"(\d+)atoms", name))
    if not m:
        continue
    size = int(m.group(1))
    arm = ("no-tree" if ("no-tree" in name or "original" in name) else
           "matrix_tree" if "matrix_tree" in name else
           "wilson" if "wilson" in name else "?")
    if samp >= budget:
        done[(size, arm)].append(rec)
    else:
        partial.append((size, arm, name, rec, samp, budget))

arms = ["no-tree", "wilson", "matrix_tree"]
print(f"\n{root}   tag={tag or '(none)'}\n")
print(f"{'n':>4} " + "".join(f"{a:>24}" for a in arms) + "      ratio")
print("-" * 90)
for size in sorted({k[0] for k in done} | {p[0] for p in partial}):
    row = f"{size:>4} "
    for a in arms:
        v = done.get((size, a), [])
        if not v:
            row += f"{'-':>24}"
        elif len(v) == 1:
            row += f"{v[0]:>18.3f}  (1)"
        else:
            row += f"{statistics.mean(v):>12.3f}±{statistics.stdev(v):.3f} ({len(v)})"
    b, w = done.get((size, "no-tree"), []), done.get((size, "wilson"), [])
    if b and w and statistics.mean(b) > 0:
        row += f"   {statistics.mean(w) / statistics.mean(b):>5.1f}x"
    elif b and w and statistics.mean(w) > 0:
        row += "     inf"
    print(row)

print(f"\nmean ± sd over runs that reached their full sample budget; (k) = seeds counted. "
      f"{sum(len(v) for v in done.values())} complete.")
if partial:
    print(f"\nstill running or cut short ({len(partial)}):")
    for size, arm, name, rec, samp, budget in sorted(partial):
        print(f"  {name:52s} {rec:.3f}  [{100 * samp / budget:3.0f}%]")
