import os
import torch
import time
import itertools
import rdkit
import typing as t
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from collections import Counter

from nmln import potentials, constraints, inference, handler, stats, ontology

DUMMY_INVALID_SMILE = "DUMMY_INVALID_SMILE"


class Model(pl.LightningModule):
    def __init__(
        self,
        ks: t.List[int],
        num_examples: int,
        p_noise: float,
        num_chains: int,
        potential_output_dim: int,
        lr: float,
        constants: t.List[str],
        predicates: t.List[ontology.Predicate],
        domain: ontology.Domain,
        ontology: ontology.Ontology,
        train_ds,
        output_dir: str,
        optim: str,
        oneof: bool,
        unary_oneof_groups: t.Optional[t.List[t.List[str]]],
        unary_groups_also_none: bool,
        symmetric: bool,
        antireflexive: bool,
        tree_constraint: bool,
        sampler_type: str,
        non_molecular: bool,
        check_every_n_samples: int,
        seed: int = 0,
    ):
        super().__init__()
        self.automatic_optimization = False

        # Ignore parameters you don't want to be auto-saved in checkpoint.
        self.save_hyperparameters(
            ignore=[
                # TODO: Save it to checkpoint.
                "constants",
                "predicates",
                "domain",
                "ontology",
                # Not these.
                "train_ds",
            ]
        )

        self.constants = constants
        self.predicates = predicates
        self.domain = domain
        self.ontology = ontology

        self.generated_smiles_path = Path(f"{output_dir}/generated_smiles.txt")
        if self.generated_smiles_path.exists():
            self.generated_smiles_path.unlink()
        self.generated_states_path = Path(f"{output_dir}/generated_states.txt")
        if self.generated_states_path.exists():
            self.generated_states_path.unlink()
        print(
            f"Generated molecules will be saved to {self.generated_smiles_path=} and {self.generated_states_path=}"
        )

        # Sampling cost accumulated separately from training/validation/RDKit
        # work, so runtime comparisons between samplers are not diluted by the
        # rest of the job.
        self.cumulative_sampling_seconds = 0.0
        self.n_sampling_calls = 0

        self.global_potential = potentials.GlobalPotential(
            torch.nn.ModuleList(
                [
                    potentials.NeuralMLPPotential(
                        k=k,
                        ontology=self.ontology,
                        potential_output_dim=self.hparams.potential_output_dim,
                        constants=self.constants,
                        beta=1.0,
                        sample_data=train_ds.y,
                    )
                    for k in self.hparams.ks
                ]
            )
        )

        self.constraints: t.List[constraints.NewOneOfNConstraint | constraints.TreeConstraint] = []

        # Each RNG consumer gets its own derived stream, so adding or removing a
        # constraint cannot shift the draws of the others.
        constraint_seed = itertools.count(seed * 1000 + 1)

        if oneof:
            self.constraints.append(
                constraints.NewOneOfNConstraint(
                    predicates_names=[
                        p.name for p in self.predicates if len(p.domains) == 1
                    ],
                    ontology=self.ontology,
                    seed=next(constraint_seed),
                )
            )

        elif unary_oneof_groups:
            # Domains where the unary predicates are not globally one-of-n
            # still need their unary atoms to belong to *some* sampling block:
            # the Gibbs sampler only ever resamples variables reachable through
            # a constraint's masks, so an uncovered atom stays at its zero
            # initialisation for the whole run.
            #
            # alsoNone says whether a group may be entirely false. Blocks world
            # needs it True, because a mid-stack block is neither `table` nor
            # `clear`. A domain whose groups are genuine one-of-n partitions --
            # every node carries exactly one mass class, say -- needs it False,
            # or the model must learn "exactly one" from scratch and spends the
            # run emitting states with no label at all.
            for group in unary_oneof_groups:
                names = [
                    name for name in group if name in self.ontology._dict_indices
                ]
                if not names:
                    continue
                self.constraints.append(
                    constraints.NewOneOfNConstraint(
                        predicates_names=names,
                        ontology=self.ontology,
                        alsoNone=unary_groups_also_none,
                        seed=next(constraint_seed),
                    )
                )

        if symmetric or antireflexive:
            self.constraints.append(
                constraints.NewOneOfNConstraint(
                    predicates_names=[
                        p.name for p in self.predicates if len(p.domains) == 2
                    ],
                    ontology=self.ontology,
                    symmetric=symmetric,
                    antireflexive=antireflexive,
                    alsoNone=True,
                    seed=next(constraint_seed),
                )
            )

        if tree_constraint:
            self.constraints.append(constraints.TreeConstraint(
                predicate_name=[
                    p.name for p in self.predicates if len(p.domains) == 2
                ],
                ontology=self.ontology,
                sampler_type=sampler_type,
                seed=next(constraint_seed),
            ))

        if not self.constraints:
            raise RuntimeError("No constraints defined.")

        self._assert_every_atom_is_sampled()

        self.sampler = inference.GPUGibbsSampler(
            global_potential=self.global_potential,
            constraints=self.constraints,
            num_examples=num_examples,
            num_variables=self.ontology.linear_size,
            num_chains=num_chains,
            seed=seed,
            log_function=self.log,
        )

        if not non_molecular:
            self.handler = handler.MoleculesHandler(self.max_num_atoms, self.ontology)
            self.train_dataset_molecules = train_ds.molecules
        else:
            self.handler = None
            self.train_dataset_molecules = []

        self.all_smiles_including_invalid = []
        self.smile_to_count = Counter()

        self.apply(self.init_weights)

        self.validation_step_outputs = []
        self.generated_invalid_count = 0
        self.check_every_n_samples = check_every_n_samples

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                m.bias.data.fill_(0.001)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["all_smiles_including_invalid"] = self.all_smiles_including_invalid
        checkpoint["smile_to_count"] = self.smile_to_count
        checkpoint["validation_step_outputs"] = self.validation_step_outputs
        checkpoint["sampler_current_state"] = self.sampler.current_state
        checkpoint["generated_invalid_count"] = self.generated_invalid_count

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.all_smiles_including_invalid = checkpoint["all_smiles_including_invalid"]
        self.smile_to_count = checkpoint["smile_to_count"]
        self.validation_step_outputs = checkpoint["validation_step_outputs"]
        self.sampler.current_state = checkpoint["sampler_current_state"]
        self.generated_invalid_count = checkpoint["generated_invalid_count"]

    @property
    def max_num_atoms(self):
        return len(self.constants)

    @property
    def current_epoch_window_idx(self) -> int:
        """
        In every N-steps window, returns the same number.
        So the artifacts in this window will get overwriten, but not the other ones.
        """
        return int(self.current_epoch / 500)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError("Use self.global_potential variable directly.")

    def sample(self):
        """Draw one blocked-Gibbs sample per chain."""
        sampling_start = time.perf_counter()
        samples = self.sampler.sample()
        self.cumulative_sampling_seconds += time.perf_counter() - sampling_start
        self.n_sampling_calls += 1
        return samples.reshape(-1, samples.shape[-1])

    def configure_optimizers(self) -> torch.optim.Adam:
        return getattr(torch.optim, self.hparams.optim)(
            self.global_potential.parameters(), lr=self.hparams.lr
        )

    def training_step(self, input_vectors_and_smiles_batch, batch_idx: int) -> torch.Tensor:
        input_batch, input_smiles = input_vectors_and_smiles_batch

        # Apply noise.
        batch = input_batch.clone()
        for constraint in self.constraints:
            batch = constraint.generate_noise_coherent_with_transformations(
                batch, self.hparams.p_noise
            )

        opt = self.optimizers()
        opt.zero_grad()

        real_potential = self.global_potential(batch)
        samples = self.sample()
        sample_potential = self.global_potential(samples)

        loss = sample_potential.mean() - real_potential.mean()
        self.manual_backward(loss)
        opt.step()

        self.log("train/real_potential", real_potential.mean(), add_dataloader_idx=False)
        self.log(
            "train/sample_potential", sample_potential.mean(), add_dataloader_idx=False
        )
        self.log("train/loss", loss, add_dataloader_idx=False)

        self.validate_and_store(samples, sample_potential, "train")

        return loss

    def validation_step(self, input_vectors_and_smiles_batch, batch_idx: int) -> None:
        input_batch, input_smiles = input_vectors_and_smiles_batch

        self.validation_step_outputs.extend(
            int(smile in self.smile_to_count) for smile in input_smiles
        )

    def on_validation_epoch_end(self):
        start_time = time.time()

        if self.validation_step_outputs:
            recall = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
            self.log("val/recall", recall)
            self.logger.log_metric(
                "val/recall/by_gen_count",
                recall,
                step=int(sum(self.smile_to_count.values())),
            )
            self.logger.log_metric(
                "val/recall/by_gen_count_inc_invalid",
                recall,
                step=int(len(self.all_smiles_including_invalid)),
            )

        self.validation_step_outputs.clear()

        for p_idx, potential in enumerate(self.global_potential.potentials):
            self.log(
                f"global_potential/potential/{p_idx}/{potential.name}/beta",
                potential.beta.data.item(),
            )

        print(f"Validation epoch end took {time.time() - start_time:.2f} seconds.")

    # TODO: If we add test data and test logic.
    # def test_step(self, batch, batch_idx: int) -> torch.Tensor:
    #     ...

    def convert_to_mol(self, molecules):
        if isinstance(molecules, np.ndarray) or isinstance(molecules, torch.Tensor):
            molecules = self.handler.fromLin2Mol(molecules)

        elif isinstance(molecules, list) and isinstance(molecules[0], str):
            molecules = list(map(Chem.MolFromSmiles, molecules))

        return molecules

    def log_molecules_or_graphs(self, *args, **kwargs):
        if self.hparams.non_molecular:
            return  # blocks / taxonomy worlds are not molecules

        self.log_molecules(*args, **kwargs)

    def log_molecules(self, name, molecules, artifact_path=None, logger=None):
        img_path = os.path.join(self.hparams.output_dir, f"{name}.jpg")

        try:
            molecules = self.convert_to_mol(molecules)
            img = Draw.MolsToGridImage(
                molecules,
                molsPerRow=5,
                subImgSize=(200, 200),
            )
            img.save(img_path)
        except Exception as e:
            print(f"Error in log_molecules [{name}]: {e}")
            return

        (logger or self.logger).log_artifact(
            img_path, artifact_path
        )

    def get_valid_molecules(self, molecules, return_indices: bool = False):
        indices = []
        valid_molecules = []

        for idx, molecule in enumerate(molecules):
            if self.hparams.non_molecular:
                # Not a molecule: keep the linear state as-is.
                mol = molecule

            else:
                molecule_dict = self.ontology.linear_to_fol_dictionary(molecule)

                try:
                    mol = self.handler.fromFol2Mol(molecule_dict, sanitize=True)

                except rdkit.Chem.rdchem.AtomValenceException:
                    # Not a valid molecule, saddy sad.
                    continue

                except Exception as e:
                    print(f"Error in get_valid_molecules while converting to mol: {e}")
                    continue

            valid_molecules.append(mol)
            indices.append(idx)

        if return_indices:
            return indices, valid_molecules

        return valid_molecules

    def _assert_every_atom_is_sampled(self) -> None:
        """Fail loudly if some atom is outside every constraint's sampling blocks.

        The Gibbs sampler walks `self.constraints` and only touches the indices
        their masks expose, so an atom covered by no constraint is frozen at its
        zero initialisation and silently removed from the model. Reflexive atoms
        of binary predicates are the one legitimate exception: the antireflexive
        and tree constraints pin them to zero by construction.
        """
        covered: set[int] = set()
        for constraint in self.constraints:
            if isinstance(constraint, constraints.TreeConstraint):
                for name in constraint.predicate_name:
                    base = int(
                        np.reshape(self.ontology._dict_indices[name], [-1])[0]
                    )
                    covered.update((base + constraint.indices_tree).tolist())
            else:
                covered.update(
                    np.asarray(constraint.integer_indices).reshape(-1).tolist()
                )

        pinned: set[int] = set()
        for predicate in self.predicates:
            if len(predicate.domains) == 2:
                indices = self.ontology._dict_indices[predicate.name]
                pinned.update(
                    int(indices[i, i]) for i in range(self.ontology.num_constants)
                )

        uncovered = set(range(self.ontology.linear_size)) - covered - pinned
        if uncovered:
            index_to_predicate = {}
            for predicate in self.predicates:
                for index in np.reshape(
                    self.ontology._dict_indices[predicate.name], [-1]
                ):
                    index_to_predicate[int(index)] = predicate.name
            affected = sorted(
                {index_to_predicate.get(i, "?") for i in uncovered}
            )
            raise RuntimeError(
                f"{len(uncovered)} atoms are covered by no constraint and would "
                f"never be sampled (predicates: {', '.join(affected)}). Add a "
                "constraint covering them, e.g. via unary_oneof_groups."
            )

    def hash_molecule(self, mol) -> str:
        if self.hparams.non_molecular:
            # mol is the raw linear state (numpy/tensor); hash via sorted FOL facts string.
            # Threshold at 0.5 so continuous MCMC samples are binarised the same way
            # as the binary ground-truth read from disk.
            raw = mol if isinstance(mol, np.ndarray) else mol.cpu().numpy()
            fol_dict = self.ontology.linear_to_fol_dictionary(raw)
            facts = []
            for pred_name, arr in fol_dict.items():
                binary = arr >= 0.5
                if binary.ndim == 1:
                    for i in np.where(binary)[0]:
                        facts.append(f"{pred_name}({i}).")
                else:
                    for i, j in zip(*np.where(binary)):
                        facts.append(f"{pred_name}({i},{j}).")
            return "".join(sorted(facts))

        else:
            return Chem.CanonSmiles(Chem.MolToSmiles(mol))

    def validate_and_store(self, samples, potentials, name):
        samples = np.reshape(
            samples.detach().cpu().numpy(), [-1, self.ontology.linear_size]
        )
        assert len(samples) == len(potentials), (
            f"Invalid length of samples and potentials: "
            f"{samples.shape} vs {potentials.shape}"
        )

        valid_indices, valid_molecules = self.get_valid_molecules(
            samples, return_indices=True
        )
        self.generated_invalid_count += len(samples) - len(valid_indices)
        try:
            valid_smiles = list(map(self.hash_molecule, valid_molecules))
        except Exception as e:
            print(f"Error in validate_and_store while hashing mols: {e}")
            return
        assert len(valid_indices) == len(valid_smiles)

        dummied_list = [DUMMY_INVALID_SMILE for _ in range(len(samples))]
        with open(self.generated_smiles_path, "a") as f:
            for smile, molecule, smile_idx in zip(
                valid_smiles, valid_molecules, valid_indices
            ):
                dummied_list[smile_idx] = smile
                self.smile_to_count[smile] += 1
                # Trailing 0 preserves the historical column layout of this file,
                # which the offline legality oracles parse positionally.
                f.write(
                    f"{self.current_epoch} {self.global_step} {smile} "
                    f"{potentials[smile_idx]} 0\n"
                )
        self.all_smiles_including_invalid.extend(dummied_list)

        with open(self.generated_states_path, "a") as f:
            for mol_idx, molecule in enumerate(samples):
                molecule_str = "-".join(str(int(x)) for x in molecule.tolist())
                f.write(
                    f"{self.current_epoch} {self.global_step} {molecule_str} "
                    f"{potentials[mol_idx]} 0\n"
                )

        if len(self.all_smiles_including_invalid) % self.check_every_n_samples:
            return

        start_time = time.time()
        print(
            f"Storing artifacts. Running full validate_and_store. "
            f"Epoch {self.current_epoch} Step {self.global_step}."
        )

        # Skip self.log and use logger class directly, because this method is invoked
        # only occasionally and optimalization in the background could cause that
        # metric would not be visible.
        try:
            n_generated_samples = float(sum(self.smile_to_count.values()))
            n_generated_samples_unique = float(len(self.smile_to_count))
            total_generated_samples_inc_invalid = len(self.all_smiles_including_invalid)

            self.logger.log_metric(
                f"{name}/valid_ratio",
                n_generated_samples / total_generated_samples_inc_invalid
                if total_generated_samples_inc_invalid
                else 0,
                step=self.global_step,
            )
            self.logger.log_metric(
                f"{name}/len_unique_smiles",
                n_generated_samples_unique,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/n_generated_samples", n_generated_samples, step=self.global_step
            )
            self.logger.log_metric(
                "val/n_generated_samples_unique",
                n_generated_samples_unique,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/n_generated_samples_inc_invalid",
                total_generated_samples_inc_invalid,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/cumulative_sampling_seconds",
                self.cumulative_sampling_seconds,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/n_sampling_calls",
                float(self.n_sampling_calls),
                step=self.global_step,
            )
            # Diagnostic only: the tree draw is exact, but a rising slow-walk
            # rate means the learned weights are becoming badly skewed, which is
            # the regime where Wilson gets expensive and matrix-tree is preferable.
            for constraint in self.constraints:
                if getattr(constraint, "total_walks", 0):
                    self.logger.log_metric(
                        "val/tree_sampler_slow_walk_ratio",
                        constraint.slow_walks / constraint.total_walks,
                        step=self.global_step,
                    )
                    self.logger.log_metric(
                        "val/tree_sampler_slow_walks",
                        float(constraint.slow_walks),
                        step=self.global_step,
                    )
                    self.logger.log_metric(
                        "val/tree_sampler_draws",
                        float(constraint.tree_draws),
                        step=self.global_step,
                    )
                    # Share of draws routed to matrix-tree because the weights
                    # were too badly conditioned for Wilson. Exact either way,
                    # but it says how often the nominal sampler was not used.
                    self.logger.log_metric(
                        "val/tree_sampler_matrix_tree_share",
                        constraint.matrix_tree_fallbacks / max(constraint.tree_draws, 1),
                        step=self.global_step,
                    )
                    # Share of draws whose weights underflowed far enough to
                    # disconnect the graph, and so needed the floor repair.
                    self.logger.log_metric(
                        "val/tree_sampler_underflow_repair_share",
                        constraint.underflow_repairs / max(constraint.tree_draws, 1),
                        step=self.global_step,
                    )
            self.logger.log_metric(
                "val/seconds_per_sampling_call",
                self.cumulative_sampling_seconds / self.n_sampling_calls
                if self.n_sampling_calls
                else 0.0,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/unique_ratio",
                n_generated_samples_unique / n_generated_samples
                if n_generated_samples
                else 0,
                step=self.global_step,
            )
            self.logger.log_metric(
                "val/valid_ratio",
                n_generated_samples / total_generated_samples_inc_invalid
                if total_generated_samples_inc_invalid
                else 0,
                step=self.global_step,
            )
        except Exception as e:
            print(f"Warning: Metric logging skipped because of connection error. {e}")

        smiles_file = os.path.join(
            self.hparams.output_dir, "all_generated_smiles.in-training.txt"
        )
        with open(smiles_file, "w") as f_sml:
            for sml in self.all_smiles_including_invalid:
                f_sml.write(sml + "\n")
        self.logger.log_artifact(smiles_file)

        if valid_smiles:
            self.log_molecules_or_graphs(
                "latest_generated_molecules", valid_molecules
            )

        if len(self.smile_to_count) and not self.hparams.non_molecular:
            most_common = self.convert_to_mol(
                [smile for smile, _ in self.smile_to_count.most_common(42)]
            )

            self.log_molecules_or_graphs(
                f"most_commonly_generated_molecules.{self.current_epoch_window_idx}",
                most_common,
                "most_commonly_generated_molecules",
            )

            stats_file = os.path.join(
                self.hparams.output_dir,
                f"most_commonly_generated_molecules_vs_true_stats."
                f"{self.current_epoch_window_idx}.jpg",
            )
            stats.stats_histograms_image(
                stats_file, most_common, self.train_dataset_molecules[: len(most_common)]
            )
            self.logger.log_artifact(
                stats_file,
                artifact_path="most_commonly_generated_molecules_vs_true_stats",
            )

        if not self.hparams.non_molecular:
            stats_file = os.path.join(
                self.hparams.output_dir,
                f"latest_generated_molecules_vs_true_stats."
                f"{self.current_epoch_window_idx}.jpg",
            )
            stats.stats_histograms_image(
                stats_file,
                self.handler.fromLin2Mol(samples),
                self.train_dataset_molecules[: len(samples)],
            )

        print(
            f"validate_and_store took {time.time() - start_time:.2f} seconds. "
            f"Epoch {self.current_epoch} Step {self.global_step}."
        )
