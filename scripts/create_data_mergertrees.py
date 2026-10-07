"""Build NMLN worlds from TNG SubLink merger trees.

A world is the assembly history of one z=0 galaxy: nodes are subhalos, edges are
progenitor-descendant links, and the whole thing is a rooted tree.

Two deliberate choices, both consequences of the width-2 expressivity result.

1. Non-branching temporal chains are collapsed. A raw SubLink tree has one node
   per snapshot, so a galaxy that simply ages for 40 snapshots contributes 40
   nodes of no structural interest. After collapsing, tree size measures merger
   history rather than simulation output cadence.

2. Epoch ordering is NOT a legality condition. Requiring ranks to descend along
   every root-to-leaf path -- as the taxonomy domain does -- smuggles in "at most
   one later-epoch neighbour", which is a degree constraint and by
   Proposition (expressivity) outside the width-2 class. Here epoch is just
   another unary attribute the model may or may not learn to respect, and the
   oracle reports violations as a diagnostic rather than as illegality. Legality
   is therefore treeness plus one-of-n per attribute group: entirely inside the
   model class.

Usage:
    python scripts/create_data_mergertrees.py --report-only     # go/no-go first
    python scripts/create_data_mergertrees.py --min-nodes 8 --max-nodes 24
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import typer

# SubhaloMassType particle-type indices used by TNG.
PT_GAS, PT_STARS = 0, 4


def load_tree(path: Path) -> dict | None:
    """Read the fields we need out of one SubLink subtree file."""
    want = [
        "SubhaloID", "SnapNum", "DescendantID",
        "FirstProgenitorID", "NextProgenitorID",
        "SubhaloMass", "SubhaloSFR",
    ]
    with h5py.File(path, "r") as f:
        if not all(k in f for k in want):
            return None
        return {k: np.array(f[k]) for k in want}


def progenitor_subtree(t: dict) -> tuple[list[int], list[tuple[int, int]]] | None:
    """Walk FirstProgenitorID / NextProgenitorID from the root.

    Returns (node indices, edges as index pairs). The root is the row whose
    SubhaloID is smallest, which SubLink guarantees is the tree's own root.
    Rooting explicitly avoids the documented TreeID trap, where one stored tree
    can hold several unrelated histories sharing a FoF group.
    """
    ids = t["SubhaloID"]
    index_of = {int(v): i for i, v in enumerate(ids)}
    root = int(np.argmin(ids))

    nodes, edges = [], []
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        nodes.append(u)
        child = int(t["FirstProgenitorID"][u])
        while child != -1:
            ci = index_of.get(child)
            if ci is None or ci in seen:
                break
            seen.add(ci)
            edges.append((u, ci))
            stack.append(ci)
            child = int(t["NextProgenitorID"][ci])
    if len(nodes) < 2:
        return None
    return nodes, edges


def collapse_chains(nodes: list[int], edges: list[tuple[int, int]], root: int,
                    stride: int = 0):
    """Drop nodes with exactly one progenitor: pure ageing, no merger event.

    stride=0 drops every such node, which is what the first datasets did. That
    removes every degree-2 node by construction, so a 6-node tree can only be
    (3,3,1,1,1,1) or (5,1,1,1,1,1) -- and 100% of the real trees came out as the
    former. A single rigid degree signature is the one thing width-2 potentials
    provably cannot express, so the model could only hit it at the combinatorial
    base rate, which is what the measured recall matched almost exactly.

    stride>0 keeps every stride-th node of a non-branching run instead, so how
    long a galaxy ages between mergers survives as structure and the degree
    sequence varies across worlds the way it does for molecules.
    """
    children = {n: [] for n in nodes}
    parent = {}
    for u, v in edges:
        children[u].append(v)
        parent[v] = u

    base = {n for n in nodes if n == root or len(children[n]) != 1}
    keep = set(base)
    if stride > 0:
        for k in base:
            for c in children[k]:
                run = []
                while c not in base:
                    run.append(c)
                    c = children[c][0]
                keep |= {run[i] for i in range(stride - 1, len(run), stride)}

    new_edges = []
    for k in keep:
        for c in children[k]:
            # walk down through single-child nodes to the next retained one
            while c not in keep:
                c = children[c][0]
            new_edges.append((k, c))
    return sorted(keep), new_edges


def attributes(t: dict, idx: int, snap_max: int) -> dict:
    """Attributes chosen for what TNG100-3 actually resolves.

    Stellar mass is zero for 60% of subhalos at this resolution and gas fraction
    piles up at 1.0, so neither can be binned usefully. Total subhalo mass is
    never zero and spans three decades, and star formation is a clean 69/31
    binary. Measured over 150 raw trees before choosing.
    """
    return dict(
        mass=float(t["SubhaloMass"][idx]),
        sfr=float(t["SubhaloSFR"][idx]),
        snap=int(t["SnapNum"][idx]),
        z_frac=int(t["SnapNum"][idx]) / max(snap_max, 1),
    )


def main(
    raw_dir: Path = typer.Option(Path("data/tng_raw")),
    out_root: Path = typer.Option(Path("data/mergertrees")),
    min_nodes: int = typer.Option(8),
    max_nodes: int = typer.Option(24),
    exact_nodes: str = typer.Option("", help="comma-separated node counts to build, e.g. 6,8,10,12. Empty reports the histogram only. Parsing 5000 raw trees takes ~20 minutes, so every requested size is written from a single pass."),
    report_only: bool = typer.Option(False, help="measure size distributions, write nothing"),
    mass_cuts: str = typer.Option(
        "1,10",
        help="two mass cuts in units of 1e10 Msun/h separating low/med/high, e.g. 1,10 "
             "for dwarf / intermediate / massive. Empty falls back to data tertiles, which "
             "force ~33/33/33 and so hand the model a near-maximum-entropy label to learn.",
    ),
    chain_stride: int = typer.Option(
        0,
        help="0 collapses every non-branching node, which forces one rigid degree "
             "sequence. A positive value keeps every stride-th node of a run, so "
             "branch length varies and so does the degree sequence.",
    ),
    val_fraction: float = typer.Option(0.2),
) -> None:
    files = sorted(raw_dir.glob("sublink_*.hdf5"))
    if not files:
        raise SystemExit(f"No SubLink files in {raw_dir}. Run download_tng_trees.py first.")
    print(f"reading {len(files)} raw trees from {raw_dir}")

    raw_sizes, kept = [], []
    all_attrs = []
    for p in files:
        t = load_tree(p)
        if t is None:
            continue
        sub = progenitor_subtree(t)
        if sub is None:
            continue
        nodes, edges = sub
        root = int(np.argmin(t["SubhaloID"]))
        raw_sizes.append(len(nodes))
        cnodes, cedges = collapse_chains(nodes, edges, root, stride=chain_stride)
        snap_max = int(t["SnapNum"].max())
        if min_nodes <= len(cnodes) <= max_nodes:
            attrs = {n: attributes(t, n, snap_max) for n in cnodes}
            all_attrs.extend(attrs.values())
            kept.append((cnodes, cedges, attrs))

    def describe(xs, label):
        if not xs:
            print(f"  {label}: none"); return
        xs = sorted(xs)
        print(f"  {label}: n={len(xs)} min={xs[0]} median={statistics.median(xs):.0f} "
              f"mean={statistics.fmean(xs):.1f} p90={xs[int(.9*len(xs))-1]} max={xs[-1]}")

    print("\n=== size distribution ===")
    describe(raw_sizes, "raw nodes per tree      ")
    describe([len(c) for c, _, _ in kept], f"after collapsing, in [{min_nodes},{max_nodes}]")
    print(f"  trees retained: {len(kept)} of {len(raw_sizes)}")

    hist = Counter(len(c) for c, _, _ in kept)
    if hist:
        print("\n=== trees per exact node count (NMLN needs a fixed domain size) ===")
        for n_ in sorted(hist):
            bar = "#" * min(40, hist[n_])
            print(f"    n={n_:<3} {hist[n_]:>5}  {bar}")
        best = max(hist, key=lambda k: hist[k])
        print(f"  most populated: n={best} with {hist[best]} trees "
              f"-- rerun with --exact-nodes {best}")

    if all_attrs:
        print("\n=== attribute ranges (for choosing bins) ===")
        for k in ("mass", "sfr"):
            v = sorted(a[k] for a in all_attrs)
            qs = [v[int(q * len(v)) - 1] for q in (0.25, 0.5, 0.75)]
            print(f"  {k:8s} min={v[0]:.4g} q25={qs[0]:.4g} med={qs[1]:.4g} "
                  f"q75={qs[2]:.4g} max={v[-1]:.4g}")

    wanted = [int(x) for x in exact_nodes.split(",") if x.strip()]
    if report_only or not wanted:
        print("\nNothing written. NMLN needs every world to have the same number of "
              "constants, so pick counts above and rerun with e.g. --exact-nodes 6,8,10.")
        return
    all_kept = kept
    print(f"\nbuilding sizes {wanted} from {len(all_kept)} collapsed trees")

    # Mass cuts. Tertiles put ~1/3 of nodes in each bin by construction, which is
    # the maximum-entropy labelling and leaves the potential almost nothing to
    # learn: measured at 1.089 nats against a 1.099 ceiling on the tertile build.
    # Fixed physical cuts inherit the real skew of the halo mass function instead.
    mass_v = sorted(a["mass"] for a in all_attrs)
    if mass_cuts.strip():
        m_lo, m_hi = (float(x) for x in mass_cuts.split(","))
        cut_kind = "physical"
    else:
        m_lo, m_hi = mass_v[len(mass_v)//3], mass_v[2*len(mass_v)//3]
        cut_kind = "tertile"

    def facts_for(cnodes, cedges, attrs) -> list[str]:
        order = {n: i for i, n in enumerate(cnodes)}
        out = []
        for u, v in cedges:
            out.append(f"prog({order[u]},{order[v]}).")
            out.append(f"prog({order[v]},{order[u]}).")
        for n in cnodes:
            a = attrs[n]; i = order[n]
            out.append(f"{'lowmass' if a['mass'] < m_lo else 'highmass' if a['mass'] >= m_hi else 'medmass'}({i}).")
            out.append(f"{'starforming' if a['sfr'] > 0 else 'quenched'}({i}).")
            z = a["z_frac"]
            out.append(f"{'early' if z < 1/3 else 'late' if z >= 2/3 else 'mid'}({i}).")
        return sorted(out)

    import random

    for want in wanted:
        sized = [k for k in all_kept if len(k[0]) == want]
        if not sized:
            print(f"  n={want}: no trees, skipped")
            continue
        rng = random.Random(42)
        worlds = [facts_for(*k) for k in sized]
        # Many real histories collapse to the same labelled tree, and recall is
        # exact-hash coverage of the validation set, so duplicates would let a
        # handful of common worlds dominate the score. Keep one copy of each.
        # (The datasets used for the reported runs predate this line; their
        # recall was recomputed over distinct worlds from the saved samples.)
        n_raw = len(worlds)
        worlds = list({tuple(w): w for w in worlds}.values())
        if len(worlds) < n_raw:
            print(f"  n={want}: {n_raw} histories -> {len(worlds)} distinct worlds")
        rng.shuffle(worlds)
        n_val = max(1, int(len(worlds) * val_fraction))
        splits = {"val": worlds[:n_val], "train": worlds[n_val:]}

        data_dir = out_root / f"mergertrees{want}"
        for split, ws in splits.items():
            d = data_dir / split
            d.mkdir(parents=True, exist_ok=True)
            for stale in d.glob("*.mol"):
                stale.unlink()
            for i, f in enumerate(ws):
                (d / f"{i}.mol").write_text("\n".join(f) + "\n")
        print(f"  n={want}: {len(splits['train'])} train / {len(splits['val'])} val -> {data_dir}")

    onto = ["lowmass:1", "medmass:1", "highmass:1",
            "quenched:1", "starforming:1", "early:1", "mid:1", "late:1", "prog:2"]
    (out_root / "ontology_mergertrees").write_text("\n".join(onto) + "\n")
    print(f"  per-node mass cuts ({cut_kind}) {m_lo:.4g} / {m_hi:.4g}, shared across sizes")

    # Report what the labels actually look like: the entropy of the per-node
    # label is what the recall ceiling turns on, so it belongs in the build log
    # rather than being rediscovered from a failed training run.
    groups = {"mass": ("lowmass", "medmass", "highmass"),
              "sf": ("quenched", "starforming"),
              "epoch": ("early", "mid", "late")}
    counts = {g: Counter() for g in groups}
    for cnodes, cedges, attrs in all_kept:
        for n in cnodes:
            a = attrs[n]
            counts["mass"]["lowmass" if a["mass"] < m_lo else
                           "highmass" if a["mass"] >= m_hi else "medmass"] += 1
            counts["sf"]["starforming" if a["sfr"] > 0 else "quenched"] += 1
            z = a["z_frac"]
            counts["epoch"]["early" if z < 1/3 else "late" if z >= 2/3 else "mid"] += 1
    total_h = 0.0
    for g, c in counts.items():
        n = sum(c.values())
        h = -sum(v / n * math.log(v / n) for v in c.values())
        total_h += h
        shares = "  ".join(f"{k}={v/n:.0%}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        print(f"    {g:6s} H={h:.3f} of max {math.log(len(groups[g])):.3f} nats   {shares}")
    print(f"    per-node label entropy (independent sum) = {total_h:.3f} nats")
    print(f"  ontology -> {out_root / 'ontology_mergertrees'}")


if __name__ == "__main__":
    typer.run(main)
