# dGSEA Paper Release

This repository is a publication-oriented, **training-reproducible** release for dGSEA experiments.

It contains:
- 3 core method files (`dgsea_core.py`, `dgsea_torch.py`, `dgsea_backend.py`)
- 1 de-duplicated training notebook (`exp3a_release_repro.ipynb`) covering MSE-only, DGSEA-only, and Hybrid training/evaluation
- scripts for one-command execution and conclusion-consistency checks

## Scope

This release supports two reproducibility levels:
1. **Algorithm sanity reproducibility** (CPU-friendly smoke test)
2. **Training reproducibility** (from data to model comparison table, then conclusion check)

The target is **conclusion-level consistency** (trend and relative improvements), not bitwise-identical checkpoints.

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

## Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate dgsea-release
```

### CPU-only (macOS, no GPU)

```bash
python3 -m venv .venv_repro
. .venv_repro/bin/activate
pip install -U pip
pip install -r requirements-cpu.txt
```

If RDKit installation fails in your platform-specific pip environment, use the Conda setup or run training with:
`--filter-invalid-smiles 0 --use-fp 0`.

## Data and Model Prerequisites

Required for training notebook execution:
- L1000 parquet file (SMILES + `gene_*` columns), e.g. `data/raw/smiles_signatures.parquet`
- ChemBERTa model source:
  - local folder via `--model-dir` (recommended), or
  - HuggingFace model id with explicit `--allow-hf-download`
- RDKit for strict paper-style preprocessing (`--filter-invalid-smiles 1`).

## Training Reproducibility (Recommended Entry)

Run full notebook execution non-interactively:

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

For faster CPU pilot runs (trend check):

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

For pilot runs, run `--check-conclusion` only if `max-samples` is large enough; tiny subsets can fail threshold checks by noise.

If you intentionally want online model download, use for example:

```bash
--model-dir seyonec/ChemBERTa-zinc-base-v1 --allow-hf-download
```

## Outputs

After training notebook execution:
- `results/training_repro/exp3a_table_test_models_core.csv`
- `results/training_repro/exp3a_table_test_per_pathway.csv`
- `results/training_repro/conclusion_check.json` (if checker is run)

Notebook artifacts and checkpoints are written under `OUT_DIR` (default: `repository-root/runs/chemberta_exp3a`).
When `run_training_repro.py` is used, export paths are resolved to repository-absolute locations to avoid notebook working-directory drift.

Reference manuscript tables are provided under:
- `results/paper_tables/exp3a_table4_gene_level_paper.csv`
- `results/paper_tables/exp3a_table5_pathway_level_paper.csv`

## Conclusion Consistency Check

Standalone check:

```bash
python scripts/check_conclusion_consistency.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

Default thresholds are in:
- `configs/conclusion_thresholds.json`

The checker validates that Hybrid remains better on key pathway metrics while gene-level quality does not collapse.

Compare reproduced metrics against manuscript reference tables:

```bash
python scripts/compare_with_paper_tables.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

## Smoke Test

Core-method smoke test (no real data required):

```bash
python scripts/smoke_test.py
```

Reduced notebook smoke test (requires real data):

```bash
python scripts/smoke_test.py \
  --run-notebook \
  --model-dir /abs/path/to/ChemBERTa \
  --data-path /abs/path/to/smiles_signatures.parquet
```

Enable checker in reduced smoke only when desired:

```bash
python scripts/smoke_test.py \
  --run-notebook \
  --run-checker \
  --model-dir /abs/path/to/ChemBERTa \
  --data-path /abs/path/to/smiles_signatures.parquet
```

## Reproducibility Notes

- Use fixed seeds and explicit paths via environment variables or CLI args.
- For archival reproducibility, record split indices and checkpoint SHA256.
- For CPU-only runs, expect longer runtime and small numeric drift versus GPU runs.

## Citation

See `CITATION.cff`.

## License

MIT License (see `LICENSE`).
