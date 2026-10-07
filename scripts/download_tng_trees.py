"""Download SubLink merger trees for individual TNG subhalos.

Needs an API key from https://www.tng-project.org/users/register/ , passed via
--api-key or the TNG_API_KEY environment variable.

Why per-subhalo API calls rather than the bulk SubLink files: the full TNG100-1
SubLink dataset is ~442 GB and TNG100-3 ~10.6 GB, but we only want a few hundred
trees. The API serves the subtree of one subhalo directly.

Why not the stored TreeID: TNG's bookkeeping can place several unrelated
progenitor-descendant trees in one stored tree when the subhalos share a
Friends-of-Friends group. Rooting at a chosen subhalo and walking
FirstProgenitorID / NextProgenitorID gives the actual progenitor subtree, which
is the object we want.

Usage:
    python scripts/download_tng_trees.py --api-key KEY --n-trees 800
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import typer

BASE = "https://www.tng-project.org/api"


def get(url: str, key: str, params: dict | None = None) -> requests.Response:
    r = requests.get(url, headers={"api-key": key}, params=params, timeout=120)
    r.raise_for_status()
    return r


def main(
    api_key: str = typer.Option("", help="TNG API key (or set TNG_API_KEY)"),
    simulation: str = typer.Option("TNG100-3", help="TNG100-3 is the small one: ~10.6 GB total"),
    snapshot: int = typer.Option(99, help="99 = z=0, the present day"),
    n_trees: int = typer.Option(800),
    min_mass: float = typer.Option(1.0, help="lower cut on subhalo mass, 1e10 Msun/h"),
    max_mass: float = typer.Option(1000.0),
    out_dir: Path = typer.Option(Path("data/tng_raw")),
    sleep: float = typer.Option(0.2, help="pause between API calls, to be polite"),
) -> None:
    key = api_key or os.environ.get("TNG_API_KEY", "")
    if not key:
        raise SystemExit("No API key. Pass --api-key or set TNG_API_KEY.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ask for z=0 subhalos in a mass range, then pull each one's SubLink subtree.
    search = f"{BASE}/{simulation}/snapshots/{snapshot}/subhalos/"
    params = {"limit": n_trees, "mass__gt": min_mass, "mass__lt": max_mass}
    print(f"querying {search} for {n_trees} subhalos in mass ({min_mass}, {max_mass}) ...")
    subs = get(search, key, params).json()["results"]
    print(f"  got {len(subs)} subhalo records")

    ok = skipped = 0
    for i, sub in enumerate(subs):
        sub_id = sub["id"]
        dest = out_dir / f"sublink_{simulation}_s{snapshot}_{sub_id}.hdf5"
        if dest.exists():
            skipped += 1
            continue
        url = f"{BASE}/{simulation}/snapshots/{snapshot}/subhalos/{sub_id}/sublink/full.hdf5"
        try:
            r = get(url, key)
        except Exception as exc:  # a subhalo may simply have no stored tree
            print(f"  [{i}] subhalo {sub_id}: {type(exc).__name__}, skipping")
            continue
        dest.write_bytes(r.content)
        ok += 1
        if ok % 25 == 0:
            print(f"  {ok} trees downloaded ({dest.stat().st_size / 1024:.0f} kB last)")
        time.sleep(sleep)

    print(f"done: {ok} downloaded, {skipped} already present, in {out_dir}")


if __name__ == "__main__":
    typer.run(main)
