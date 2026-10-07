import os
import shutil
import random
import typer
import pandas as pd
import typing as t
import json
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from dig.ggraph import dataset as dig_dataset
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Process
from nmln.data_creation import smiles_to_fol_and_smiles

DATATESTS = [
    "tree",
    "chembl",
    "QM9",
    "ZINC250k",
    "MOSES",
]


def main(
    dataset: str,
    atom_size: int = 8,
    val_ratio: float = 0.2,
    min_val: int = 10,
    inc_skip_bond: bool = False,
    smiles_file: t.Optional[Path] = None,
    ontology: t.Optional[Path] = None,
):
    if smiles_file is None:
        assert dataset in DATATESTS
    pl.seed_everything(0)

    out_folder = Path(f"data/molecules/{dataset}{atom_size}")# + ("_SkipBond" if inc_skip_bond else ""))
    tmp_folder = Path(f"{str(out_folder)}_tmp")

    if out_folder.exists():
        shutil.rmtree(out_folder)
    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)

    tmp_folder.mkdir(parents=True, exist_ok=False)
    (out_folder / "train").mkdir(parents=True, exist_ok=False)
    (out_folder / "val").mkdir(parents=True, exist_ok=False)

    if smiles_file is not None:
        smiles = Path(smiles_file).read_text().splitlines()
        smiles = [s.strip() for s in smiles if s.strip()]
    elif dataset == "chembl":
        # Download from http://chembl.blogspot.com/2020/03/chembl-26-released.html.
        df = pd.read_csv("data/chembl_26_chemreps.txt", sep="\t")
        smiles = df["canonical_smiles"].values
    else:
        dig_ds = getattr(dig_dataset, dataset)(root="/app/data/dig")
        smiles = list(set(dig_ds.all_smiles))

    ontology_path = str(ontology) if ontology is not None else f"/app/data/molecules/ontology_{dataset.lower()}"

    size_dist = smiles_to_fol_and_smiles.smiles_to_fol_and_smiles(
        ontology_path,
        smiles,
        tmp_folder,
        atom_size=atom_size,
        smiles_dir=out_folder,
    )

    with open(f"data/molecules/{dataset}_size_dist.json", "w") as f:
        json.dump(size_dist, f, indent=2)

    # Plot the size distribution
    sizes, counts = zip(*(sorted(size_dist.items(), key=lambda x: x[0])[:30]))
    plt.figure(figsize=(10, 6))
    plt.bar(sizes, counts, color='skyblue')
    plt.xlabel('Molecule Size')
    plt.ylabel('Frequency')
    plt.title(f'Size Distribution of {dataset} Molecules')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot as a .jpg file
    plt.savefig(f"data/molecules/{dataset}_size_dist.jpg")
    plt.close()

    all_entries = list(tmp_folder.glob("*.mol"))
    random.shuffle(all_entries)

    if len(all_entries) > 25:
        for idx_entry, entry in enumerate(tqdm(all_entries)):
            if random.random() < val_ratio or idx_entry < min_val:
                out = out_folder / "val"
                out_smiles_file = out_folder / "val_smiles.txt"

            else:
                out = out_folder / "train"
                out_smiles_file = out_folder / "train_smiles.txt"

            with open(out_smiles_file, "a") as f:
                with open(str(entry).replace(".mol", ".smile"), "r") as f2:
                    f.write(f2.read().strip() + "\n")

            # Launch in process because some occasional Prolog's fatal error can not be catched by except Exception block.
            if inc_skip_bond:
                from nmln.data_creation import infer_skipbonds
                work = Process(
                    target=infer_skipbonds.infer_skipbonds,
                    args=(os.path.basename(entry), tmp_folder, out),
                )
                work.start()
                work.join()
            else:
                shutil.move(entry, out / entry.name)
    else:
        print(f"No or too little entries found in the temporary folder {tmp_folder} ({len(all_entries)}).")
        shutil.rmtree(out_folder)

    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    typer.run(main)
