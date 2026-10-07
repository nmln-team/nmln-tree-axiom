"""
Evaluate a generated_smiles.txt file with the same metrics used in the NMLN MLflow framework.

Usage:
    python -m scripts.evaluate_generated_smiles \
        --generated path/to/generated_smiles.txt \
        --data-name tree8

Metrics computed (matching MLflow keys):
    val/validity       - fraction of parseable SMILES
    val/unique_ratio   - unique valid / total valid
    val/recall         - fraction of val set covered by generated set
    novelty            - fraction of unique valid not seen in training set
    + histogram diffs  - bonds, degree, rings, HBD, HBA, TPSA, etc.
"""

import typer
import typing as t
from pathlib import Path
from collections import Counter

from rdkit import Chem, RDLogger
from tqdm import tqdm

from nmln.stats import diffs_histograms_image, FS

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(pretty_exceptions_enable=False)


def load_canonical_smiles(path: Path) -> t.List[str]:
    results = []
    with open(path) as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            canon = Chem.CanonSmiles(smi)
            if canon:
                results.append(canon)
    return results


def load_and_validate(path: Path) -> t.Tuple[t.List[str], t.List[str]]:
    """Returns (valid_canonical_smiles, all_raw_lines)."""
    valid, all_lines = [], []
    with open(path) as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            smi = line.strip()
            if not smi:
                continue
            all_lines.append(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid.append(Chem.MolToSmiles(mol))
    return valid, all_lines


@app.command()
def main(
    generated: Path = typer.Option(..., help="Path to generated_smiles.txt"),
    data_name: str = typer.Option("tree8", help="Dataset name, e.g. tree8"),
    plot: bool = typer.Option(False, help="Save histogram comparison plots"),
) -> None:
    data_dir = PROJECT_ROOT / "data" / "molecules" / data_name
    val_smiles_path = data_dir / "val_smiles.txt"
    train_smiles_path = data_dir / "train_smiles.txt"

    if not val_smiles_path.exists():
        typer.echo(f"ERROR: val_smiles.txt not found at {val_smiles_path}")
        raise typer.Exit(1)

    # Load reference sets
    val_canons = set(load_canonical_smiles(val_smiles_path))
    train_canons = set(load_canonical_smiles(train_smiles_path)) if train_smiles_path.exists() else set()

    typer.echo(f"Val set size:   {len(val_canons)}")
    typer.echo(f"Train set size: {len(train_canons)}")

    # Load and validate generated SMILES
    valid_smiles, all_lines = load_and_validate(generated)

    n_total = len(all_lines)
    n_valid = len(valid_smiles)
    unique_valid = list(Counter(valid_smiles).keys())
    n_unique = len(unique_valid)

    validity = n_valid / n_total if n_total else 0.0
    unique_ratio = n_unique / n_valid if n_valid else 0.0

    covered = set(unique_valid) & val_canons
    recall = len(covered) / len(val_canons) if val_canons else 0.0

    novel = set(unique_valid) - train_canons
    novelty = len(novel) / n_unique if n_unique else 0.0

    typer.echo(f"\n=== Metrics for {generated.name} on {data_name} ===")
    typer.echo(f"Total generated:           {n_total}")
    typer.echo(f"Valid (val/validity):      {n_valid}/{n_total} = {validity:.4f}")
    typer.echo(f"Unique valid:              {n_unique}")
    typer.echo(f"Unique ratio:              {unique_ratio:.4f}")
    typer.echo(f"Recall (val coverage):     {len(covered)}/{len(val_canons)} = {recall:.4f}")
    typer.echo(f"Novelty (not in train):    {len(novel)}/{n_unique} = {novelty:.4f}")

    # Histogram diffs vs training set (same as in nmln/stats.py)
    if train_canons:
        typer.echo("\n--- Histogram diffs vs training set ---")
        generated_mols = [m for s in unique_valid if (m := Chem.MolFromSmiles(s)) is not None]
        train_mols = [m for s in train_canons if (m := Chem.MolFromSmiles(s)) is not None]
        diffs = diffs_histograms_image(generated_mols, train_mols)
        for stat_name, diff in diffs.items():
            typer.echo(f"  {stat_name:<25} {diff:.4f}")

        if plot:
            from nmln.stats import stats_histograms_image
            out_path = generated.parent / f"{generated.stem}_stats_{data_name}.jpg"
            stats_histograms_image(str(out_path), generated_mols, train_mols)
            typer.echo(f"\nHistogram plot saved to {out_path}")


if __name__ == "__main__":
    app()
