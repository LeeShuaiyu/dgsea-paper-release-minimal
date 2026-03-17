# Reproducibility Checklist

## 1. Environment

- Prefer pinned environment from `environment.yml`.
- CPU-only fallback: `requirements-cpu.txt`.
- RDKit is required when `--filter-invalid-smiles 1` or `--use-fp 1`.
  - If RDKit is unavailable in pip-only environments, run with `--filter-invalid-smiles 0 --use-fp 0`.
- Record versions:
  - `python --version`
  - `torch.__version__` (if installed)
  - `numpy.__version__`

## 2. Data and Split

- Keep exact source and version of L1000 parquet.
- Record split policy and seed (`SEED`, `TEST_FRAC`, `VAL_FRAC`).
- Release default seed is `42`.
- For archival release, export split indices.

## 3. Pathway Definitions

- Publish exact 5 pathway definitions and gene-id namespace.
- Keep the same pathway mapping policy across runs.

## 4. Checkpoint Integrity

- Record SHA256 for released checkpoints.
- Suggested command:

```bash
sha256sum checkpoints/*.pt
```

## 5. Train-Then-Check Workflow

Run training notebook non-interactively:

```bash
python scripts/run_training_repro.py \
  --data-path /abs/path/to/smiles_signatures.parquet \
  --model-dir /abs/path/to/ChemBERTa \
  --local-files-only \
  --filter-invalid-smiles 1 \
  --use-fp 0 \
  --check-conclusion
```

Then (or automatically with `--check-conclusion`) run:

```bash
python scripts/check_conclusion_consistency.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

To compare against manuscript reference values:

```bash
python scripts/compare_with_paper_tables.py \
  --core-table results/training_repro/exp3a_table_test_models_core.csv
```

## 6. Conclusion-Level Criteria

Default thresholds in `configs/conclusion_thresholds.json` verify:
- Hybrid pathway correlation improves
- Hybrid pathway sign accuracy improves
- Hybrid pathway MSE decreases
- Gene-level performance does not collapse

This is intended for **conclusion-level reproducibility** (trend consistency), not exact bitwise equality.

## 7. CPU-Only Smoke Test

Core-only smoke test:

```bash
python scripts/smoke_test.py
```

Reduced notebook smoke test (requires data):

```bash
python scripts/smoke_test.py \
  --run-notebook \
  --data-path /abs/path/to/smiles_signatures.parquet
```

Expected terminal ending:

`[smoke] PASS`
