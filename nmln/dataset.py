import torch
import numpy as np
import typing as t
from rdkit import Chem

from pathlib import Path
from tqdm import tqdm
from nmln.ontology import Predicate, Domain, Ontology
from nmln import handler


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        constants: t.List[str],
        predicates: t.List[Predicate],
        domain: Domain,
        ontology: Ontology,
        mol_dir: t.Union[str, Path],
        limit: int | None,
    ):
        super().__init__()

        self.mol_dir = Path(mol_dir)

        self.constants = constants
        self.files = self.load_files(self.mol_dir)
        self.limit = limit
        if limit is not None:
            self.files = self.files[:limit]
        self.domain = domain
        self.predicates = predicates
        self.ontology = ontology

        self.y = torch.from_numpy(
            np.stack(
                [self.ontology.file_to_linearState(file) for file in tqdm(self.files, desc=f"Loading {self.mol_dir}")], axis=0
            ).astype(np.float32)
        )

        self.handler = handler.MoleculesHandler(len(constants), ontology)
        self.molecules = self.handler.fromLin2Mol(self.y)
        self.smiles = [
            Chem.CanonSmiles(Chem.MolToSmiles(mol))
            for mol in tqdm(self.molecules, desc="dataset to smiles")
        ]

    @staticmethod
    def load_files(path: Path) -> list:
        print(f"Loading files from {path}")
        return sorted([p for p in path.glob("*.mol") if "ontology" not in str(p)])

    @property
    def max_num_atoms(self):
        return len(self.constants)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.y[idx], self.smiles[idx]

    def get_collate_fn(self) -> t.Callable:
        def collate_fn(batch):
            return torch.vstack([b[0] for b in batch]), [b[1] for b in batch]

        return collate_fn

    def get_dataloader(
        self,
        shuffle: bool,
        batch_size: int,
        num_workers: int,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.get_collate_fn(),
            prefetch_factor=22 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )


class FOLDataset(torch.utils.data.Dataset):
    """
    Dataset for non-molecular FOL worlds.

    Identical to Dataset but skips all RDKit/MoleculesHandler logic, and keys
    each world by the sorted string of its ground FOL facts, which plays the
    role the canonical SMILES plays for molecules.
    """

    def __init__(
        self,
        constants: t.List[str],
        predicates: t.List[Predicate],
        domain: Domain,
        ontology: Ontology,
        mol_dir: t.Union[str, Path],
        limit: int | None,
    ):
        super().__init__()

        self.mol_dir = Path(mol_dir)
        self.constants = constants
        self.files = Dataset.load_files(self.mol_dir)
        self.limit = limit
        if limit is not None:
            self.files = self.files[:limit]
        self.domain = domain
        self.predicates = predicates
        self.ontology = ontology

        self.y = torch.from_numpy(
            np.stack(
                [
                    self.ontology.file_to_linearState(file)
                    for file in tqdm(self.files, desc=f"Loading {self.mol_dir}")
                ],
                axis=0,
            ).astype(np.float32)
        )

        # Sorted FOL facts string plays the role of the canonical SMILES.
        self.smiles = [self._file_to_key(f) for f in tqdm(self.files, desc="building keys")]
        self.molecules = []

    @staticmethod
    def _file_to_key(path: Path) -> str:
        with open(path) as f:
            return "".join(sorted(line.strip() for line in f if line.strip()))

    @property
    def max_num_atoms(self):
        return len(self.constants)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.y[idx], self.smiles[idx]

    def get_collate_fn(self) -> t.Callable:
        def collate_fn(batch):
            return torch.vstack([b[0] for b in batch]), [b[1] for b in batch]

        return collate_fn

    def get_dataloader(
        self,
        shuffle: bool,
        batch_size: int,
        num_workers: int,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.get_collate_fn(),
            prefetch_factor=22 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
