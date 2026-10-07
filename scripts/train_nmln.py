import os
import typer
import dagshub
import typing as t
import pytorch_lightning as pl

from tqdm import tqdm
from rdkit import RDLogger
from icecream import install as ic_install
#from scripts.create_data_molecules import DATATESTS

from nmln import utils, dataset, model, ontology
from nmln.dataset import FOLDataset

ic_install()
app = typer.Typer(pretty_exceptions_enable=False)


class MaxSamplesCallback(pl.Callback):
    """Stop training when val/n_generated_samples_inc_invalid reaches max_samples."""

    def __init__(self, max_samples: int):
        self.max_samples = max_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        generated = getattr(pl_module, "all_smiles_including_invalid", None)
        if generated and len(generated) >= self.max_samples:
            trainer.should_stop = True


@app.command()
def molecules(
    name: str = "",
    comment: str = "",
    dataset_name: str = "tree",
    max_num_atoms: int = 8,
    num_examples: int = 1,
    p_noise: float = 0.1,
    gpus: t.Optional[t.List[int]] = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    ks: t.List[int] = [2],
    tree_constraint: bool = False,
    sampler_type: str = typer.Option(
        "wilson", help="Tree sampler: 'wilson' or 'matrix_tree'"
    ),
    symmetric_constraint: bool = True,
    antireflexive_constraint: bool = True,
    potential_output_dim: int = 1,
    lr: float = 1e-5,
    num_chains: int = 20,
    batch_size: int | None = 10,
    max_generated_samples: int | None = 2_000_000,
    seed: int = 22,
) -> None:
    """Tree-structured molecules (ChEMBL, 8-15 heavy atoms)."""
    if not name:
        constraint_tag = f"sampler={sampler_type}" if tree_constraint else "no-tree"
        name = f"mlj-{dataset_name}{max_num_atoms}-{constraint_tag}-seed={seed}"
    train(
        name=name,
        comment=comment,
        data_type="molecules",
        data_name=f"{dataset_name}{max_num_atoms}",
        ontology_path=f"data/molecules/ontology_{dataset_name.lower()}",
        max_num_atoms=max_num_atoms,
        num_examples=num_examples,
        p_noise=p_noise,
        gpus=gpus,
        limit_train=limit_train,
        limit_val=limit_val,
        ks=ks,
        batch_size=batch_size,
        tree_constraint=tree_constraint,
        sampler_type=sampler_type,
        symmetric_constraint=symmetric_constraint,
        antireflexive_constraint=antireflexive_constraint,
        potential_output_dim=potential_output_dim,
        lr=lr,
        num_chains=num_chains,
        max_generated_samples=max_generated_samples,
        seed=seed,
    )


def train(
    name: str,
    comment: str,
    data_type: str,
    data_name: str,
    ontology_path: str,
    max_num_atoms: int,
    ks: t.List[int] = [2],
    num_examples: int = 1,
    p_noise: float = 0.1,
    num_chains: int = 20,
    potential_output_dim: int = 1,
    lr: float = 1e-5,
    max_epochs: int = 100_000,
    enable_checkpoint: bool = True,
    enable_earlystopping: bool = False,
    dataloader_num_workers: int = 4,
    optim: str = "Adam",
    gpus: t.Optional[t.List[int]] = None,
    precision: int = 32,
    versioning: bool = False,
    autocreate: bool = True,
    oneof_constraint: bool = True,
    unary_oneof_groups: t.Optional[t.List[t.List[str]]] = None,
    unary_groups_also_none: bool = True,
    symmetric_constraint: bool = True,
    antireflexive_constraint: bool = True,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
    tree_constraint: bool = False,
    non_molecular: bool = False,
    sampler_type: str = "wilson",
    seed: int = 22,
    max_generated_samples: int | None = None,
) -> None:
    n_samples_per_sample = num_examples * num_chains
    batch_size = batch_size or n_samples_per_sample
    script_parameters = dict(locals())
    pl.seed_everything(seed, workers=True)

    # Disable unnecessary warnings about unvalid molecules when using `validate_and_store`.

    RDLogger.DisableLog("rdApp.*")

    # Prepare directories.

    exp_name, output_dir = utils.get_next_version_directory(
        name, versioning=versioning, autocreate=autocreate
    )
    print(f"Experiment name: {exp_name}")
    print(f"Output directory: {output_dir}")

    # Initialize datasets.

    constants = [str(i) for i in range(max_num_atoms)]
    domain = ontology.Domain(name="atoms", constants=constants)
    predicates_dict = ontology.load_predicates(ontology_path)
    script_parameters["predicates_dict"] = predicates_dict
    predicates = [
        ontology.Predicate(p, domains=[domain for _ in range(a)])
        for p, a in predicates_dict.items()
    ]
    ontology_ = ontology.Ontology(domains=[domain], predicates=predicates)

    DatasetClass = FOLDataset if non_molecular else dataset.Dataset
    train_ds = DatasetClass(
        constants=constants,
        domain=domain,
        predicates=predicates,
        ontology=ontology_,
        mol_dir=f"data/{data_type}/{data_name}/train",
        limit=limit_train,
    )
    val_ds = DatasetClass(
        constants=constants,
        domain=domain,
        predicates=predicates,
        ontology=ontology_,
        mol_dir=f"data/{data_type}/{data_name}/val",
        limit=limit_val,
    )
    test_ds = None

    if not non_molecular:
        val_smiles_path = os.path.join("data", "molecules", data_name, "val_smiles.txt")
        if not os.path.exists(val_smiles_path):
            with open(val_smiles_path, "w") as f:
                for smile in val_ds.smiles:
                    f.write(f"{smile}\n")


    # Initialize logger.


    if os.environ.get("DAGSHUB_REPO_OWNER") or os.environ.get("DAGSHUB_REPO_NAME"):
        dagshub.init(
            repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"),
            repo_name=os.environ.get("DAGSHUB_REPO_NAME"),
            mlflow=True,
        )

    logger_mlflow = utils.SafeMLFlowLogger(
        run_name=exp_name,
        tags={
            "mlflow.note.content": f"""{comment}

{output_dir=}
""",
        },
    )
    logger_mlflow.log_hyperparams(script_parameters) 
    logger_mlflow.log_metric(
        "len/train_dataset", len(train_ds)
    )
    logger_mlflow.log_metric("len/val_dataset", len(val_ds))

    # Test that splits does not contain same molecules

    #for a, b in tqdm(it.combinations([set(train_ds.smiles), set(val_ds.smiles)], 2), desc="checking common"):
    #    if common := a & b:
    #        raise RuntimeError(
    #            f"Splits share {len(common) / min(len(a), len(b))} of data."
    #        )
    # Remove common molecules from val to ensure clean splits
    common = set(train_ds.smiles) & set(val_ds.smiles)
    if common:
        print(f"Removing {len(common)} common molecules from val split.")
        keep = [i for i, s in enumerate(val_ds.smiles) if s not in common]
        val_ds.y = val_ds.y[keep]
        val_ds.smiles = [val_ds.smiles[i] for i in keep]
    # Initialize model.

    check_every_n_samples = n_samples_per_sample * 100
    nmln = model.Model(
        ks=ks,
        num_examples=num_examples,
        p_noise=p_noise,
        num_chains=num_chains,
        potential_output_dim=potential_output_dim,
        lr=lr,
        constants=constants,
        domain=domain,
        predicates=predicates,
        ontology=ontology_,
        train_ds=train_ds,
        output_dir=str(output_dir),
        optim=optim,
        oneof=oneof_constraint,
        unary_oneof_groups=unary_oneof_groups,
        unary_groups_also_none=unary_groups_also_none,
        symmetric=symmetric_constraint,
        antireflexive=antireflexive_constraint,
        check_every_n_samples=check_every_n_samples,
        tree_constraint=tree_constraint,
        sampler_type=sampler_type,
        non_molecular=non_molecular,
        seed=seed,
    )

    # Save anything relevant to a single json file.

    logger_mlflow.experiment.log_dict(
        logger_mlflow.run_id,
        {
            "script": script_parameters,
            "nmln": nmln.hparams,
        },
        "all_params.json",
    )

    # Plot some samples of molecules.

    for split_name, ds in tqdm([
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ], desc="logging molecules"):
        if ds is not None:
            nmln.log_molecules_or_graphs(
                f"sample_molecules_from_{split_name}", ds.y[:50], logger=logger_mlflow
            )

    # Prepare for training.

    callbacks = []

    checkpoint_filename = "last"
    checkpoint_filepath = f"{output_dir}/{checkpoint_filename}.ckpt"
    if enable_checkpoint:
        callbacks.append(
            utils.MLFlowModelCheckpoint(
                logger_mlflow=logger_mlflow,
                dirpath=output_dir,
                filename=checkpoint_filename,
                monitor=None,
            )
        )

    if enable_earlystopping:
        # See https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#earlystopping.
        callbacks.append(
            pl.callbacks.EarlyStopping(
                # TODO: What can we monitor for early stopping?
                monitor="val/loss" if val_ds is not None else "train/loss",
                patience=1000,
                mode="min",
            )
        )

    if max_generated_samples is not None:
        callbacks.append(MaxSamplesCallback(max_samples=max_generated_samples))

    val_check_interval=check_every_n_samples // n_samples_per_sample
    assert val_check_interval >= 1, val_check_interval

    trainer = pl.Trainer(
        devices=gpus if gpus is not None else "auto",
        accelerator="gpu" if gpus is not None else "cpu",
        max_epochs=max_epochs,
        logger=[logger_mlflow],
        callbacks=callbacks,
        # Run validation after every N samples.
        val_check_interval=val_check_interval,  
        check_val_every_n_epoch=None,
        precision=precision,
        log_every_n_steps=max(min(len(train_ds) // batch_size, 50), 1),
        # Uncomment for profiling.
        # profiler=pl.profilers.AdvancedProfiler(
        #     line_count_restriction=100, dirpath="/app", filename="profiler"
        # ),
        # fast_dev_run=10,
    )

    # Train!

    print("Training!")

    trainer.fit(
        nmln,
        train_dataloaders=train_ds.get_dataloader(
            shuffle=True,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
        ),
        val_dataloaders=(
            val_ds.get_dataloader(
                shuffle=False,
                batch_size=batch_size,
                num_workers=dataloader_num_workers,
            )
            if val_ds is not None
            else None
        ),
    )

    if test_ds is not None:
        trainer.test(
            nmln,
            dataloaders=test_ds.get_dataloader(
                shuffle=False,
                batch_size=batch_size,
                num_workers=dataloader_num_workers,
            ),
        )


@app.command()
def mergertrees(
    name: str = "",
    num_nodes: int = typer.Option(16, help="constants per world; must match the built data"),
    sampler_type: str = typer.Option("wilson", help="Tree sampler: 'wilson' or 'matrix_tree'"),
    tree_constraint: bool = typer.Option(True),
    seed: int = typer.Option(22),
    p_noise: float = 0.1,
    num_examples: int = 1,
    num_chains: int = 20,
    ks: t.List[int] = [2],
    potential_output_dim: int = 1,
    lr: float = 1e-5,
    gpus: t.Optional[t.List[int]] = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = 10,
    max_generated_samples: int | None = 2_000_000,
) -> None:
    """TNG galaxy merger trees: assembly histories with binned physical attributes."""
    constraint_tag = f"sampler={sampler_type}" if tree_constraint else "no-tree"
    run_name = f"mlj-mergertrees{num_nodes}-{constraint_tag}-seed={seed}"
    if name:
        run_name = f"{run_name}-{name}"
    train(
        name=run_name,
        comment=f"TNG merger trees n={num_nodes} {constraint_tag} seed={seed}",
        data_type="mergertrees",
        data_name=f"mergertrees{num_nodes}",
        ontology_path="data/mergertrees/ontology_mergertrees",
        max_num_atoms=num_nodes,
        ks=ks,
        num_examples=num_examples,
        num_chains=num_chains,
        p_noise=p_noise,
        potential_output_dim=potential_output_dim,
        lr=lr,
        tree_constraint=tree_constraint,
        symmetric_constraint=not tree_constraint,
        antireflexive_constraint=not tree_constraint,
        # Four independent one-of-n attribute groups, so the global oneof is off
        # and each group gets its own sampling block.
        oneof_constraint=False,
        unary_oneof_groups=[
            ["lowmass", "medmass", "highmass"],
            ["quenched", "starforming"],
            ["early", "mid", "late"],
        ],
        # every galaxy carries exactly one label from each group
        unary_groups_also_none=False,
        non_molecular=True,
        gpus=gpus,
        limit_train=limit_train,
        limit_val=limit_val,
        batch_size=batch_size,
        sampler_type=sampler_type,
        seed=seed,
        max_generated_samples=max_generated_samples,
    )


if __name__ == "__main__":
    app()
