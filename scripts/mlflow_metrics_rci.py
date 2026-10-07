import typer
import mlflow
import typing as t
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from joblib import Memory
from rdkit import Chem
from tqdm import tqdm
from random import random
from nmln.model import DUMMY_INVALID_SMILE
from collections import Counter

CUT_OFF_N_GENERATED = 2_000_000
USE_ARTIFACT_TO_GET_SMILES = True
ONLY_VALID_SMILES_IN_RECALL_PLOT = True
MEMORY = Memory("./cachedir", verbose=0)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(pretty_exceptions_enable=False)

# --- Plot style ---
COLORS = {
    "Original (no tree)":   "#2166AC",
    "Wilson Sampler":        "#D6604D",
    "Matrix-Tree Sampler":   "#4DAC26",
}
MARKERS = {
    "Original (no tree)":   "o",
    "Wilson Sampler":        "s",
    "Matrix-Tree Sampler":   "^",
}
LINESTYLES = {
    "Original (no tree)":   "-",
    "Wilson Sampler":        "--",
    "Matrix-Tree Sampler":   "-.",
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "lines.linewidth":    2.0,
    "lines.markersize":   0,
    "figure.dpi":         150,
})


@app.command()
def main(
    atom_size: t.Optional[int] = typer.Option(None, help="Filter by atom size, e.g. 8. If None, generate all."),
    n_steps: int = 10,
    tracking_uri: str = typer.Option(
        str(PROJECT_ROOT / "mlruns"),
        help="MLflow store to read, e.g. mlj_revision/mlruns for the revision campaign",
    ),
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    cached_get_generated_smiles = MEMORY.cache(get_generated_smiles_from_run_file)

    out_folder = PROJECT_ROOT / "metrics_rci"
    out_folder.mkdir(parents=True, exist_ok=True)

    filter_string = f"params.data_name = 'tree{atom_size}'" if atom_size else None
    experiment_ids = [exp.experiment_id for exp in client.search_experiments(max_results=1000)]
    runs = client.search_runs(experiment_ids=experiment_ids, max_results=1000, filter_string=filter_string)

    run_ids_by_data_name: t.Dict[str, t.List[str]] = {}
    for run in runs:
        data_name = run.data.params.get("data_name")
        if not data_name:
            continue
        run_name = run.data.tags.get("mlflow.runName", "")
        chains = "20chains" if "20chain" in run_name else "10chains"
        key = f"{data_name}_{chains}"
        run_ids_by_data_name.setdefault(key, []).append(run.info.run_id)

    for data_name, run_ids in tqdm(run_ids_by_data_name.items(), desc="Datasets"):
        plots = []
        legends = []

        actual_data_name = data_name.split("_")[0]  # strip "_20chains" suffix
        val_smiles_path = PROJECT_ROOT / "data" / "molecules" / actual_data_name / "val_smiles.txt"
        if not val_smiles_path.exists():
            print(f"Validation smiles not found: {val_smiles_path}, skipping.")
            continue
        with open(val_smiles_path) as f:
            validation_canons = set(
                Chem.CanonSmiles(l.strip()) for l in tqdm(f, desc="Load val") if l.strip()
            )

        for run_id in tqdm(run_ids, desc="Runs"):
            run = client.get_run(run_id)
            run_name = run.data.tags.get("mlflow.runName", "")
            try:
                method_name = run_name_to_plot_name(run_name)
            except ValueError as e:
                print(f"Skipping: {e}")
                continue

            if USE_ARTIFACT_TO_GET_SMILES:
                generated_canons = cached_get_generated_smiles(run_id)
                if not generated_canons:
                    continue
                generated_count = len(generated_canons)
                if ONLY_VALID_SMILES_IN_RECALL_PLOT:
                    generated_canons = [x for x in generated_canons if not x.startswith("0.")]
                xs, ys = create_recall_curve(validation_canons, generated_canons)
            else:
                history = client.get_metric_history(run_id, "val/recall/by_gen_count_inc_invalid")
                history = [m for m in history if m.step <= CUT_OFF_N_GENERATED]
                if not history:
                    continue
                step = max(len(history) // n_steps, 1)
                xs = [m.step for m in history][::step]
                ys = [m.value for m in history][::step]
                generated_count = xs[-1] if xs else 0

            plots.append((xs, ys))
            legends.append(method_name)

        if not plots:
            print(f"No data for {data_name}.")
            continue

        sorted_pl = sorted(zip(plots, legends), key=lambda pl: pl[0][1][-1] if pl[0][1] else 0, reverse=True)
        plots, legends = zip(*sorted_pl)

        # Extract atom size and chain count from key (e.g. "tree8_20chains")
        parts = data_name.split("_")
        n_atoms = parts[0].replace("tree", "")
        chains_label = parts[1] if len(parts) > 1 else ""

        fig, ax = plt.subplots(figsize=(7, 5))

        for (xs, ys), label in zip(plots, legends):
            color = COLORS.get(label, None)
            marker = MARKERS.get(label, "o")
            ls = LINESTYLES.get(label, "-")
            # Only mark every N-th point to keep the plot clean
            ax.plot(xs, ys, label=label, color=color, linestyle=ls)

        ax.set_title(f"Recall curve — {n_atoms} atoms ({chains_label})", pad=8)
        ax.set_ylabel("Recall")
        ax.set_xlabel("Unique valid molecules (sorted by frequency)")
        ax.set_xlim(0, CUT_OFF_N_GENERATED)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{int(v/1000)}k" if v < 1_000_000 else f"{v/1_000_000:.1f}M"
        ))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.legend(loc="upper left", framealpha=0.9)

        fig.tight_layout()
        out_path = out_folder / f"{data_name}.eps"  # data_name is already "tree10_20chains"
        fig.savefig(out_path, format="eps", bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)


def run_name_to_plot_name(run_name: str) -> str:
    """Map a run name onto its legend label.

    Run names are built by scripts/train_nmln.py and carry the arm as either
    "no-tree" or "sampler=<name>".
    """
    if "no-tree" in run_name:
        return "Original (no tree)"
    elif "sampler=wilson" in run_name:
        return "Wilson Sampler"
    elif "sampler=matrix_tree" in run_name:
        return "Matrix-Tree Sampler"
    else:
        raise ValueError(f"Unknown run name: {run_name}")


def canon_smile_or_random(txt: str) -> str:
    if DUMMY_INVALID_SMILE in txt:
        return str(random())
    try:
        return Chem.CanonSmiles(txt.strip())
    except Exception:
        return str(random())


def get_generated_smiles_from_run_file(run_id: str) -> t.List[str]:
    filename = "all_generated_smiles.in-training.txt"
    client = mlflow.tracking.MlflowClient()
    available_artifacts = client.list_artifacts(run_id=run_id)
    if not any(filename in x.path for x in available_artifacts):
        print(f"No artifact {filename} for run {run_id}.")
        return []
    generated_smiles_path = client.download_artifacts(run_id=run_id, path=filename, dst_path="/tmp/")
    with open(generated_smiles_path) as f:
        return [
            canon_smile_or_random(line)
            for idx, line in tqdm(enumerate(f), desc="Load generated")
            if idx < CUT_OFF_N_GENERATED and line.strip()
        ]


def create_recall_curve(
    validation_canons: t.Set[str], generated_canons: t.List[str]
) -> t.Tuple[t.List[int], t.List[float]]:
    counter = Counter(generated_canons[:CUT_OFF_N_GENERATED])
    sorted_canons = [c for c, _ in counter.most_common(CUT_OFF_N_GENERATED)]
    x, recalls = [], []
    step = 10_000
    for i in range(0, len(sorted_canons), step):
        subset = set(sorted_canons[:i])
        x.append(len(subset))
        recalls.append(len(subset & validation_canons) / len(validation_canons))
    end_set = set(sorted_canons)
    if not x or x[-1] != len(end_set):
        x.append(len(end_set))
        recalls.append(len(end_set & validation_canons) / len(validation_canons))
    return x, recalls


if __name__ == "__main__":
    app()
