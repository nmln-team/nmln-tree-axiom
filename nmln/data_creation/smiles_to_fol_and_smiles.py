import typer
import typing as t
import pandas as pd
import traceback

from collections import Counter
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from nmln import ontology, handler


def smiles_to_fol_and_smiles(
    ontology_path: str,
    smiles: list[str],
    save_folder: Path,
    atom_size: int,
    smiles_dir: t.Optional[Path] = None,
) -> dict[int, int]:
    smiles_list = []
    smile_to_idx = {}
    smile_to_write_id = {}

    constants = [str(i) for i in range(atom_size)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in ontology.load_predicates(ontology_path).items()
    ]
    bond_and_atom_types = [p.name for p in predicates]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)
    handler_initialized = handler.MoleculesHandler(atom_size, ontology_)

    size_distribution = Counter()

    all_missing_types = set()
    all_used_types = set()

    for i, smile in enumerate(tqdm(smiles, desc="smiles_to_fol_and_smiles")):
        try:
            if "+" in smile or "-" in smile:
                continue

            smile = Chem.CanonSmiles(smile)
            mol = Chem.MolFromSmiles(smile)
            num = int(mol.GetNumAtoms())

            size_distribution[num] += 1

            if num == atom_size:
                if "AROMATIC" not in bond_and_atom_types:
                    Chem.Kekulize(mol, clearAromaticFlags=True)

                smile = Chem.CanonSmiles(Chem.MolToSmiles(mol))

                to_write = []
                missing_types = set()

                for a in mol.GetAtoms():
                    sym = str(a.GetSymbol()).lower()

                    if sym not in bond_and_atom_types:
                        to_write = []
                        missing_types.add((sym, 1))

                    else:
                        aid = a.GetIdx()
                        to_write.append("%s(%d)." % (sym.lower(), aid))
                        all_used_types.add((sym.lower(), 1))

                for b in mol.GetBonds():
                    sid = b.GetBeginAtom().GetIdx()
                    eid = b.GetEndAtom().GetIdx()
                    type = str(b.GetBondType()).lower()

                    if type not in bond_and_atom_types:
                        to_write = []
                        missing_types.add((type, 2))

                    else:
                        to_write.append(f"{type.lower()}({sid},{eid}).")
                        to_write.append(f"{type.lower()}({eid},{sid}).")
                        all_used_types.add((type.lower(), 2))

                if missing_types:
                    print(f"Missing types for {smile}: {missing_types}")
                    all_missing_types.update(missing_types)
                    continue
                
                assert len(to_write) > 0, f"Empty to_write for {smile}"
                to_write = sorted(to_write)
                to_write_id = "".join(to_write)
                smile_to_write_id[to_write_id] = smile_to_write_id.get(
                    to_write_id, len(smile_to_write_id)
                )

                filepath = save_folder / f"{smile_to_write_id[to_write_id]}.mol"
                filepath_smile = save_folder / f"{smile_to_write_id[to_write_id]}.smile"

                our_smile = Chem.CanonSmiles(
                    Chem.MolToSmiles(
                        handler_initialized.fromLin2Mol(
                            [ontology_.file_content_to_linearState(to_write)]
                        )[0]
                    )
                )

                if filepath.exists() or our_smile in smile_to_idx:
                    continue

                smile_to_idx[our_smile] = smile_to_idx.get(our_smile, len(smile_to_idx))

                with open(filepath, "w") as f:
                    for s in to_write:
                        f.write(s + "\n")

                with open(filepath_smile, "w") as f:
                    f.write(our_smile)

                smiles_list.append(our_smile + "\n")

        except Exception as e:
            traceback.print_exc()
            print(e)

    if smiles_dir is not None:
        with open(smiles_dir / f"smile{atom_size}", "w") as smiles_file:
            for line in smiles_list:
                smiles_file.write(line)

        with open(smiles_dir / f"smile{atom_size}_missing_types", "w") as missing_file:
            for item in sorted(all_missing_types):
                missing_file.write(f"{item[0]}:{item[1]}\n")

        with open(smiles_dir / f"smile{atom_size}_used_types", "w") as used_file:
            for item in sorted(all_used_types):
                used_file.write(f"{item[0]}:{item[1]}\n")

    print(Counter(smiles_list).most_common(5))
    return dict(size_distribution.most_common())
