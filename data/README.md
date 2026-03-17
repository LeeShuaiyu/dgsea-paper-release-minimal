# Data Folder

Place non-public or large data locally; do not commit raw large files to Git.

Recommended layout:

```text
data/
├── raw/
│   └── smiles_signatures.parquet
└── processed/
    ├── split_seed42.npz
    └── pathways_5.json
```

`split_seed42.npz` should contain train/val/test indices used in the manuscript run.
