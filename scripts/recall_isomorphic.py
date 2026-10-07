"""Isomorphism-invariant recall for FOL domains, re-scored from saved samples.

The model is permutation-invariant over constants, so it spreads probability
across every renumbering of a world; the data files use one fixed numbering
(for merger trees, 0 is the z=0 root and 1 the main progenitor). Exact-hash
recall credits only that numbering. Canonical SMILES already quotient this out
for molecules; this script does the same for FOL worlds, so the two kinds of
domain are scored alike.

Canonical key = colour refinement (1-WL) run to convergence, seeded with each
constant's unary labels and refined over typed edges. Colour refinement decides
isomorphism exactly on trees, which is what every legal world here is; a
non-tree sample can never share a key with a tree (edge count is part of the
key), so the baseline's illegal samples cannot produce false hits.

Reports, per (size, arm) over seeds: recall over distinct validation worlds
with the raw key (what is in the paper tables now) and with the canonical key,
plus a recall-at-cutoff CSV under the canonical key. Stdlib only.

    python scripts/recall_isomorphic.py --domain mergertrees --tag 3M --sizes 4,6 --out ~/iso_galaxy
"""
import argparse, csv, pathlib, re, statistics, collections, hashlib

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--experiments", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--data-suffix", default="", help="e.g. _k3 for colortrees7_k3")
ap.add_argument("--sizes", required=True)
ap.add_argument("--arms", default="original,wilson,matrix_tree")
ap.add_argument("--budget", type=int, default=0, help="samples read per run; 0 = all")
ap.add_argument("--points", type=int, default=60)
ap.add_argument("--out", default="recall_isomorphic")
ap.add_argument("--every", type=int, default=10_000, help="sample interval for the recall-vs-samples curve")
a = ap.parse_args()
data = pathlib.Path(a.data or f"data/{a.domain}")
exps = pathlib.Path(a.experiments or f"{a.domain}/experiments")
ARMPAT = {"original": "no-tree", "wilson": "sampler=wilson", "matrix_tree": "sampler=matrix_tree"}
FACT = re.compile(r"(\w+)\(([^)]*)\)\.")


def canonical(key: str) -> str:
    """Colour-refinement canonical form of a world given as its sorted fact string,
    returned as a short digest so millions of keys fit in memory."""
    labels = collections.defaultdict(list); adj = collections.defaultdict(list); nodes = set()
    for pred, args in FACT.findall(key):
        xs = args.split(",")
        if len(xs) == 1:
            labels[xs[0]].append(pred); nodes.add(xs[0])
        else:
            u, v = xs; adj[u].append((pred, v)); adj[v].append((pred, u)); nodes.update((u, v))
    col = {v: (tuple(sorted(labels[v])),) for v in nodes}
    n_edges = sum(len(adj[v]) for v in nodes) // 2
    for _ in range(len(nodes)):
        new = {v: (col[v], tuple(sorted((p, col[u]) for p, u in adj[v]))) for v in nodes}
        # relabel to small ints so keys stay short; refinement is what matters
        ids = {c: i for i, c in enumerate(sorted(set(new.values())))}
        nxt = {v: ids[new[v]] for v in nodes}
        if len(set(nxt.values())) == len(set(col.values())):
            col = nxt; break
        col = nxt
    # final key: multiset of (node colour), multiset of typed coloured edges, size
    emap = collections.Counter()
    for v in nodes:
        for p, u in adj[v]:
            if v < u: emap[(p, tuple(sorted((col[v], col[u]))))] += 1
    # colours are only canonical up to the refinement's own ordering, which is
    # itself derived from sorted structure, so the pair below is isomorphism-invariant
    form = (len(nodes), n_edges, tuple(sorted(collections.Counter(col.values()).items())), tuple(sorted(emap.items())))
    return hashlib.sha1(repr(form).encode()).hexdigest()


def is_tree_candidate(key: str, n: int) -> bool:
    """A symmetric binary fact contributes two facts with one comma each; a tree on n
    nodes has n-1 edges. Anything else cannot be isomorphic to a validation world."""
    return key.count(",") == 2 * (n - 1)


def world_keys(d):
    return ["".join(sorted(l.strip() for l in f.read_text().splitlines() if l.strip())) for f in d.glob("*.mol")]


rows, summary, prog_rows = [], [], []
LABEL = {"original": "NMLN without tree axiom", "wilson": "with tree axiom (Wilson)", "matrix_tree": "with tree axiom (matrix-tree)"}
for n in [int(x) for x in a.sizes.split(",")]:
    val_raw = set(world_keys(data / f"{a.domain}{n}{a.data_suffix}" / "val"))
    val_iso = {canonical(k) for k in val_raw}
    print(f"\nn={n}: {len(val_raw)} distinct validation worlds, {len(val_iso)} up to isomorphism")
    for arm in a.arms.split(","):
        runs = sorted(r for r in exps.glob(f"mlj-{a.domain}{n}{a.data_suffix}-{ARMPAT[arm]}-seed=*")
                      if (r.name.endswith("-" + a.tag) if a.tag else not re.search(r"-seed=\d+-.+$", r.name)))
        raw_r, iso_r, curves, progs = [], [], [], []
        for rd in runs:
            f = rd / "all_generated_smiles.in-training.txt"
            if not f.exists(): continue
            cnt = collections.Counter(); memo = {}; hit = set(); prog = []
            with open(f) as fh:
                for i, line in enumerate(fh, start=1):
                    if a.budget and i > a.budget: break
                    s = line.strip()
                    if s:
                        if s not in cnt and is_tree_candidate(s, n):   # first sight: canonicalise once
                            c = canonical(s); memo[s] = c
                            if c in val_iso: hit.add(c)
                        cnt[s] += 1
                    if i % a.every == 0:
                        prog.append((i, len(hit) / len(val_iso)))
            if not cnt: continue
            if prog: progs.append(prog)
            raw_r.append(len(val_raw & set(cnt)) / len(val_raw))
            # non-tree samples keep their raw key: they can never hit a validation
            # tree, so merging their isomorphic copies would change nothing reported
            iso_cnt = collections.Counter()
            for k, c in cnt.items(): iso_cnt[memo.get(k, "NT:" + k)] += c
            del cnt, memo
            iso_r.append(len(val_iso & set(iso_cnt)) / len(val_iso))
            hits, pts = 0, []
            for t, k in enumerate(k for k, _ in iso_cnt.most_common()):
                if k in val_iso: hits += 1
                pts.append((t + 1, hits / len(val_iso)))
            curves.append(pts)
        if not raw_r: continue
        m = lambda v: f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
        print(f"  {arm:<12} {len(raw_r)} seeds   raw-key recall {m(raw_r)}   isomorphism-invariant {m(iso_r)}")
        summary.append((n, arm, len(raw_r), statistics.mean(raw_r), statistics.mean(iso_r)))
        for si, prog in enumerate(progs):
            for x, y in prog: prog_rows.append((n, arm, si, x, y))
        xmax = max(p[-1][0] for p in curves)
        grid = sorted({max(1, round(xmax ** (i / (a.points - 1)))) for i in range(a.points)} | {xmax})
        for t in grid:
            ys = [(p[min(len(p), t) - 1][1] if t < p[-1][0] else p[-1][1]) for p in curves]
            rows.append((n, arm, t, statistics.mean(ys), statistics.stdev(ys) if len(ys) > 1 else 0.0, len(ys)))

with open(a.out + ".csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size", "arm", "t", "recall_mean", "recall_sd", "seeds"]); w.writerows(rows)
with open(a.out + "_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size", "arm", "seeds", "recall_raw_key", "recall_isomorphic"]); w.writerows(summary)
with open(a.out + "_curve.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["size", "arm", "seed_idx", "samples", "recall_isomorphic"]); w.writerows(prog_rows)
print(f"\n-> {a.out}.csv (cutoff), {a.out}_summary.csv, {a.out}_curve.csv (recall vs samples)")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not importable; plot the _curve.csv elsewhere"); raise SystemExit
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.linestyle": "--", "grid.alpha": .4})
STYLE = {"original": ("#2166AC", "-"), "wilson": ("#D6604D", "--"), "matrix_tree": ("#4DAC26", "-.")}
sizes = sorted({r[0] for r in prog_rows})
if sizes:
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.4 * len(sizes), 3.2), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        for arm in a.arms.split(","):
            seeds = sorted({r[2] for r in prog_rows if r[0] == n and r[1] == arm})
            if not seeds: continue
            c, ls = STYLE[arm]; mean = collections.defaultdict(list)
            for si in seeds:
                sr = [r for r in prog_rows if r[0] == n and r[1] == arm and r[2] == si]
                ax.plot([r[3] / 1e6 for r in sr], [r[4] for r in sr], ls, color=c, lw=0.7, alpha=0.45)
                for r in sr: mean[r[3]].append(r[4])
            xs = sorted(mean); ax.plot([x / 1e6 for x in xs], [statistics.mean(mean[x]) for x in xs], ls, color=c, lw=2.2, label=f"{LABEL[arm]} ({len(seeds)} seeds)")
        ax.set_title(f"n = {n}"); ax.set_xlabel("samples generated (millions)"); ax.set_ylabel("recall, up to isomorphism"); ax.set_ylim(0, 1.03)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"{a.out}_curve.{ext}", dpi=150, bbox_inches="tight")
    print(f"-> {a.out}_curve.pdf / .png")
