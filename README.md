# dGSEA Paper Code

Publication-oriented, training-reproducible release for **dGSEA**, a differentiable GSEA framework for pathway-level supervision in transcriptomic learning.

dGSEA addresses the mismatch between **gene-level training objectives** and **pathway-level interpretation**. It turns classical GSEA into a training-compatible objective by replacing hard ranking, discrete prefix accumulation, and hard extremum selection with smooth differentiable relaxations.

This repository is designed for **paper-level reproducibility**: from algorithm sanity checks to training runs, result tables, and conclusion-consistency validation.

## What is included

- **Core implementation**
  - `src/dgsea_core.py`
  - `src/dgsea_torch.py`
  - `src/dgsea_backend.py`
- **Training notebook**
  - `notebooks/exp3a_release_repro.ipynb`
- **Reproducibility scripts**
  - one-command training reproduction
  - smoke tests
  - conclusion-consistency checks
  - comparison against paper tables
- **Release assets**
  - configs
  - paper reference tables
  - reproducibility notes

## Quick Start

### 1. Environment setup

#### Conda

```bash
conda env create -f environment.yml
conda activate dgsea-release
```

#### CPU-only (macOS / no GPU)

```bash
python3 -m venv .venv_repro
. .venv_repro/bin/activate
pip install -U pip
pip install -r requirements-cpu.txt
```

If RDKit installation fails in your platform-specific pip environment, use the Conda setup or run training with:

```bash
--filter-invalid-smiles 0 --use-fp 0
```

### 2. Smoke test

```bash
python scripts/smoke_test.py
```

### 3. Full training reproduction

```bash
python scripts/run_training_repro.py \
  --data-path /abs/path/to/smiles_signatures.parquet \
  --model-dir /abs/path/to/ChemBERTa \
  --local-files-only \
  --device cpu \
  --batch-size 16 \
  --epochs-mse 5 \
  --epochs-dgsea 5 \
  --epochs-hybrid 4 \
  --filter-invalid-smiles 1 \
  --use-fp 0 \
  --check-conclusion
```

### 4. Fast CPU pilot run

```bash
python scripts/run_training_repro.py \
  --data-path /abs/path/to/smiles_signatures.parquet \
  --model-dir /abs/path/to/ChemBERTa \
  --local-files-only \
  --device cpu \
  --max-samples 4000 \
  --batch-size 8 \
  --epochs-mse 2 \
  --epochs-dgsea 2 \
  --epochs-hybrid 2 \
  --filter-invalid-smiles 0 \
  --use-fp 0
```

If you intentionally want online model download, use for example:

```bash
--model-dir seyonec/ChemBERTa-zinc-base-v1 --allow-hf-download
```

## Reproducibility

This release supports two levels of reproducibility:

1. **Algorithm sanity reproducibility**  
   CPU-friendly smoke tests for core method behavior.

2. **Training reproducibility**  
   From data and model inputs to reproduced result tables and conclusion checks.

The target is **conclusion-level consistency** (trend and relative improvements), not bitwise-identical checkpoints.

### Required inputs

- **L1000 parquet file** containing SMILES and `gene_*` columns  
  example: `data/raw/smiles_signatures.parquet`
- **ChemBERTa model source**
  - local folder via `--model-dir` (recommended), or
  - HuggingFace model id with explicit `--allow-hf-download`
- **RDKit** for strict paper-style preprocessing (`--filter-invalid-smiles 1`)

### Outputs

After training reproduction, the main outputs are:

- `results/training_repro/exp3a_table_test_models_core.csv`
- `results/training_repro/exp3a_table_test_per_pathway.csv`
- `results/training_repro/conclusion_check.json` (if checker is enabled)

Notebook artifacts and checkpoints are written under `OUT_DIR` (default: `runs/chemberta_exp3a`).

Reference manuscript tables are provided under:

- `results/paper_tables/exp3a_table4_gene_level_paper.csv`
- `results/paper_tables/exp3a_table5_pathway_level_paper.csv`

### Validation

Run the standalone conclusion checker:

```bash
python scripts/check_conclusion_consistency.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

Compare reproduced metrics against manuscript reference tables:

```bash
python scripts/compare_with_paper_tables.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

Default thresholds are defined in:

- `configs/conclusion_thresholds.json`

## Repository Structure

```text
dgsea-paper-release/
├── src/
│   ├── dgsea_core.py
│   ├── dgsea_torch.py
│   └── dgsea_backend.py
├── notebooks/
│   └── exp3a_release_repro.ipynb
├── scripts/
│   ├── run_training_repro.py
│   ├── check_conclusion_consistency.py
│   ├── compare_with_paper_tables.py
│   └── smoke_test.py
├── configs/
│   ├── release_paths.example.yaml
│   └── conclusion_thresholds.json
├── data/
│   └── README.md
├── checkpoints/
│   └── README.md
├── results/
│   └── README.md
├── docs/
│   └── reproducibility.md
├── environment.yml
├── requirements.txt
├── requirements-cpu.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## Notes

- Use fixed seeds and explicit paths via environment variables or CLI arguments.
- For archival reproducibility, record split indices and checkpoint SHA256.
- CPU-only runs are expected to be slower and may show small numeric drift relative to GPU runs.
- The **dGSEA-only** setting is included as an ablation. For faithful transcriptome prediction, the **hybrid objective** is the recommended setting.

## Citation

If you use this repository, please cite the paper. See `CITATION.cff`.

## License

MIT License (see `LICENSE`).
