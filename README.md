# Neural Markov Logic Networks with Tree Axiom

Code for the paper *"Neural Markov Logic Networks with Tree Axiom"*
(Idlahcen, Jung, Kuželka — Machine Learning Journal).

NMLNs whose single binary predicate is hard-constrained to form an undirected
spanning tree. Training uses persistent-chain Gibbs sampling in which the tree
block is resampled *exactly* with a weighted spanning-tree sampler (Wilson's
algorithm or a matrix-tree contraction/deletion sampler), while unary and
edge-label atoms are sampled conventionally. The experiments cover
tree-structured molecule generation (ChEMBL, 8–15 heavy atoms), galaxy merger
trees from the IllustrisTNG simulation, and a sampler-only scaling study.

---

## 1. Repository layout

```
nmln/                     the NMLN package
  model.py                LightningModule: training loop, generation, validation
  potentials.py           the width-2 neural local score (MLP over fragments)
  constraints.py          one-of-n + tree constraints (the blocked sampler lives here)
  sampler.py              spanning-tree samplers (Wilson, matrix-tree)
  dataset.py              Dataset (molecules) and FOLDataset (generic FOL worlds)
  ontology.py             predicates, domains, linear-state encoding
scripts/
  train_nmln.py                 training entry point (typer CLI: molecules / mergertrees)
  create_data_molecules.py      ChEMBL -> tree-molecule worlds
  download_tng_trees.py         fetch SubLink merger trees from the TNG API
  create_data_mergertrees.py    raw TNG trees -> collapsed, attributed FOL worlds
  recall_at_cutoff.py           recall-at-cutoff curves for molecules (Fig. 1)
  recall_isomorphic.py          recall up to isomorphism for FOL domains (Fig. 4)
  generate_time_table.py        runtime tables and plot, mean ± std over seeds (Table 3, Fig. 2)
  benchmark_tree_samplers.py    sampler-only scaling benchmark (Fig. 3)
  inspect_generated_worlds.py   decode generated_states.txt into readable facts
sbatches_shell_files/     SLURM launchers for the paper grids (see §6)
data/                     not committed; see §4
```

## 2. World format

Each training example is one `.mol` file of ground FOL facts over constants
`0..n-1`, e.g. for a molecule (`data/molecules/tree8/train/1.mol`):

```
c(0).            # unary: element labels
n(4).
single(0,1).     # binary: bonds, stored symmetrically
single(1,0).
double(5,6).
double(6,5).
```

An ontology file lists `predicate:arity`, one per line. Merger-tree worlds use
the same format with a symmetric `edge` predicate and one-of-n unary predicates
for the binned attributes (mass class, star-formation flag, epoch bin).

## 3. Installation

Python 3.10, PyTorch 2.4 + CUDA 11.8. Either:

* **Poetry (local):** `poetry install` from the repo root.
* **Conda (cluster):** `bash sbatches_shell_files/setup_conda_env.sh`
  (creates env `nmln-env` and `pip install -e .`).
* **Docker:** `docker compose build` using the provided `Dockerfile`.

## 4. Data

The datasets are not part of this repository: both are rebuilt from public
sources with the scripts below, and the raw TNG data may not be redistributed.
Only the two ontology files and the molecule tree-filter script ship under
`data/`. After the steps below, `data/molecules/tree{8..15}` and
`data/mergertrees/mergertrees{4,6,8,10,12}` sit at the repo root.

### 4.1 Tree molecules (ChEMBL)

1. Download the ChEMBL 26 compound representations and extract them to
   `data/chembl_26_chemreps.txt`:

   ```bash
   wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_26/chembl_26_chemreps.txt.gz
   gunzip -c chembl_26_chemreps.txt.gz > data/chembl_26_chemreps.txt
   ```

2. Convert every molecule with exactly N heavy atoms to a FOL world (the
   `chembl` command of the original NMLN code; one call per N in 8..15):

   ```bash
   python scripts/create_data_molecules.py chembl --atom-size 8 --ontology data/molecules/ontology_tree
   ```

   This writes `data/molecules/chembl8/{train,val}/*.mol` and `chembl8/smile8`,
   the SMILES string of every world in file order.

3. Keep the trees only, and re-split. `data/molecules/tree_data_generation_process.sh`
   merges train and val, keeps the molecules whose SMILES contain no ring-closure
   digit (i.e. ring-free molecules, which are exactly the trees), renumbers the
   files, and splits 80/20 with `random.seed(42)`:

   ```bash
   cd data/molecules && bash tree_data_generation_process.sh 8 && cd ../..
   ```

   This writes `data/molecules/tree8/{train,val}`. `train_smiles.txt` is produced
   by `python scripts/generate_train_smiles.py`; `val_smiles.txt` is written by
   the training script on its first run.

Resulting sizes, train/val: 8 atoms 306/79, 9 379/93, 10 494/112, 11 513/148,
12 523/122, 13 516/119, 14 542/128, 15 512/117. Vocabulary: seven element
types (C, N, O, S, P, Cl, F), bond labels `single` and `double`
(`data/molecules/ontology_tree`).

### 4.2 Galaxy merger trees (IllustrisTNG)

```bash
export TNG_API_KEY=...          # free account at https://www.tng-project.org
python scripts/download_tng_trees.py --simulation TNG100-3 --n-trees 5000
python scripts/create_data_mergertrees.py --exact-nodes 4,6,8,10,12
```

The TNG project distributes its data after free registration; the API key
is personal and the raw files may not be redistributed, which is why this
step cannot be skipped. The download fetches the SubLink trees rooted at z=0
subhaloes of TNG100-3 into `data/tng_raw/` (4,999 trees were parsed for the paper). The builder collapses non-branching runs of snapshots so that
the node count measures the merger history, bins total subhalo mass into
three classes at 10^10 and 10^11 M_sun/h (`--mass-cuts 1,10`, the default),
star formation into a binary flag and epoch into three bins, writes
`data/mergertrees/mergertrees{N}/{train,val}` with an 80/20 split, and the
ontology. Constants follow one fixed numbering per world (0 is the z=0 root, 1
its main progenitor), which is why recall is scored up to isomorphism (§7).

## 5. Training

All training goes through `scripts/train_nmln.py` (one subcommand per domain).
The paper configuration for molecules:

```bash
python scripts/train_nmln.py molecules \
    --name my-run --dataset-name tree --max-num-atoms 12 --seed 22 \
    --gpus 0 --tree-constraint --sampler-type wilson \
    --no-symmetric-constraint --no-antireflexive-constraint \
    --batch-size 10 --num-chains 20 --ks 2 --num-examples 1 \
    --potential-output-dim 1 --p-noise 0.1 --lr 1e-5 \
    --max-generated-samples 2000000
```

* Baseline (no tree axiom): drop `--tree-constraint --sampler-type ...
  --no-symmetric-constraint --no-antireflexive-constraint`.
* Matrix-tree sampler: `--sampler-type matrix_tree`.
* `--max-generated-samples 2000000` stops the run after 2M generated samples,
  the budget used for every reported number.

Merger trees use the same hyperparameters, built in as defaults:

```bash
python scripts/train_nmln.py mergertrees --num-nodes 6 --seed 22 --gpus 0 \
    --sampler-type wilson --num-chains 20 --batch-size 10 --max-generated-samples 2000000
```

with `--no-tree-constraint` for the baseline. The `wilson` sampler is the
paired sampler of the paper: a draw starts with Wilson's walk and is routed to
the matrix-tree sampler when the learned weights are too badly conditioned for
the walk to finish within the step bound; the share of such draws is logged as
`val/tree_sampler_matrix_tree_share`.

Outputs per run:

* an MLflow run (metrics incl. `val/recall...`, `val/valid_ratio`,
  `val/n_generated_samples_inc_invalid`, `val/cumulative_sampling_seconds`) in
  the store given by `MLFLOW_TRACKING_URI` (default `./mlruns`);
* generated worlds in `<experiments dir>/<run>/generated_states.txt` (and
  `all_generated_smiles.in-training.txt` for molecules), where the experiments
  dir is `data/experiments/` or `$NMLN_EXPERIMENTS_DIR` if set.

## 6. Reproducing the paper grids (SLURM)

| script | contents |
|---|---|
| `sbatch_molecules_exact_rerun.sh` | molecules: 8 sizes × {no tree axiom, wilson, matrix-tree} × seeds 22–26 (120 tasks) |
| `sbatch_mergertrees_grid.sh` | merger trees; sizes, arms, seeds and budget are set through `MT_SIZES`, `MT_ARMS`, `MT_SEEDS`, `MT_SAMPLES` (see the header for the paper's calls) |
| `sbatch_scaling_benchmark.sh` | sampler-only scaling curve, n up to 5000, 60 s budget, timeouts recorded |

Adjust the `REPO`/`PYTHON` paths and `--partition` at the top of each file for
your cluster. Without SLURM, run the §5 commands directly with
`MLFLOW_TRACKING_URI` and `NMLN_EXPERIMENTS_DIR` exported.

## 7. Evaluation

**Recall at cutoff, molecules (Fig. 1):** ranks each run's unique generated
molecules by frequency and reports the fraction of validation molecules among
the top t, aggregated over seeds. Stdlib only; matplotlib for the figure.

```bash
python scripts/recall_at_cutoff.py --experiments <experiments dir> --out recall_cutoff
python scripts/recall_at_cutoff.py --from-csv recall_cutoff.csv --out recall_cutoff   # plot only
```

**Recall up to isomorphism, merger trees (Fig. 4):** the model is
permutation-invariant over constants, so a validation world counts as recovered
when any renumbering of it is generated. Canonical keys come from colour
refinement, which decides isomorphism exactly on trees.

```bash
python scripts/recall_isomorphic.py --domain mergertrees --sizes 4,6,8,10 --out iso_galaxy
```

**Runtime table and plot (Table 3, Fig. 2):** mean ± std over seeds;
`sampling` = time inside the sampler from `val/cumulative_sampling_seconds`,
`wall` = total job time.

```bash
python scripts/generate_time_table.py --tracking-uri mlruns --metric sampling --out-dir paper_tables
python scripts/generate_time_table.py --tracking-uri mlruns --metric wall     --out-dir paper_tables
```

**Sampler scaling benchmark (Fig. 3):** times one tree draw per domain size,
outside any learning loop; a draw over the time limit ends its configuration,
and a sampler whose mean exceeds the limit is skipped at larger n and recorded
as a timeout.

```bash
python scripts/benchmark_tree_samplers.py \
    --sizes 10,20,50,100,200,500,1000,2000,5000 \
    --samplers wilson,matrix_tree --regimes uniform,lognormal \
    --reps 5 --time-limit 60 --device cpu --out benchmark_results/tree_samplers.csv
```


## License

MIT, see `LICENSE`. The NMLN core builds on the original NMLN implementation by Marra and Kuželka.
