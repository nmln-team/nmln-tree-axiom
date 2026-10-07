"""
Generate train_smiles.txt for tree8–tree15 folders from existing .mol files in train/.
No GPU required — only RDKit and the nmln package.

Usage (from the repo root):
    python scripts/generate_train_smiles.py
"""

import sys
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem

# Add repo root to path so 'nmln' is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nmln import ontology, handler

DATA_DIR = REPO_ROOT / "data" / "molecules"
ONTOLOGY_PATH = str(DATA_DIR / "ontology_tree")
TREE_FOLDERS = [DATA_DIR / f"tree{n}" for n in range(8, 16)]


def mol_file_to_smiles(mol_path: Path, handler_initialized, ontology_) -> str:
    lines = mol_path.read_text().splitlines()
    linear = ontology_.file_content_to_linearState(lines)
    mol = handler_initialized.fromLin2Mol([linear])[0]
    if mol is None:
        return None
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def process_tree(tree_folder: Path):
    atom_size = int(tree_folder.name.replace("tree", ""))
    train_dir = tree_folder / "train"
    out_file = tree_folder / "train_smiles.txt"

    if not train_dir.exists():
        print(f"  [skip] {train_dir} does not exist")
        return

    constants = [str(i) for i in range(atom_size)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in ontology.load_predicates(ONTOLOGY_PATH).items()
    ]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)
    handler_initialized = handler.MoleculesHandler(atom_size, ontology_)

    mol_files = sorted(train_dir.glob("*.mol"))
    smiles_list = []
    errors = 0

    for mol_path in tqdm(mol_files, desc=tree_folder.name):
        try:
            smi = mol_file_to_smiles(mol_path, handler_initialized, ontology_)
            if smi:
                smiles_list.append(smi)
            else:
                errors += 1
        except Exception as e:
            errors += 1

    with open(out_file, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    print(f"  -> wrote {len(smiles_list)} SMILES to {out_file}  ({errors} errors)")


if __name__ == "__main__":
    for tree_folder in TREE_FOLDERS:
        if not tree_folder.exists():
            print(f"[skip] {tree_folder} not found")
            continue
        print(f"\nProcessing {tree_folder.name} ...")
        process_tree(tree_folder)

    print("\nDone.")
