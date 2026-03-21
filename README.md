# dGSEA: Differentiable Gene Set Enrichment Analysis for Pathway-Level Supervision

[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.03.18.712610v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official implementation of **"Differentiable Gene Set Enrichment Analysis for Pathway-Level Supervision in Transcriptomic Learning"** (Li et al., 2026).

---

## Overview

<p align="center">
  <img src="figures/fig1.png" width="700" alt="Objective mismatch and dGSEA overview"/>
</p>

Transcriptomic prediction models are trained with gene-level objectives (MSE, Pearson correlation) but evaluated via pathway-level statistics such as Gene Set Enrichment Analysis (GSEA). This objective–functional mismatch causes unreliable pathway conclusions under imperfect prediction.

**dGSEA** bridges this gap by constructing a differentiable surrogate of classical GSEA that maps predicted gene-level scores to pathway enrichment with stable gradients. It replaces three non-differentiable operations with principled continuous relaxations:

| Classical GSEA | dGSEA |
|---|---|
| Hard ranking | Temperature-controlled soft ranking |
| Discrete prefix accumulation | Smooth sigmoid prefix kernel |
| Extremum selection | Softmax-weighted aggregation |

<p align="center">
  <img src="figures/fig2.png" width="700" alt="dGSEA differentiable pipeline"/>
</p>

Sign-specific robust permutation normalization (**dNES**) preserves the statistical semantics of the classical normalized enrichment score. A Nyström–windowing approximation (**nyswin**) reduces the naive O(G²) complexity to near-linear, enabling genome-scale integration into training loops.

When used as an auxiliary objective for SMILES-to-transcriptome prediction on LINCS L1000, dGSEA improves pathway-level agreement without sacrificing gene-level fidelity:

<p align="center">
  <img src="figures/fig3.png" width="700" alt="Training pipeline with dGSEA supervision"/>
</p>

| Metric | Baseline | + dGSEA |
|---|---|---|
| Macro pathway correlation | 0.257 | **0.306** (+19%) |
| Sign accuracy | 0.620 | **0.641** (+3.4%) |
| Pathway MSE | 1.784 | **1.610** (−9.8%) |
| Mean gene Pearson *r* | 0.449 | 0.452 |
| Gene RMSE | 0.420 | 0.418 |

---

## Installation

```bash
git clone https://github.com/LeeShuaiyu/dgsea-paper-code
cd dgsea-paper-code
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, RDKit. See `requirements.txt` for the full dependency list.

---

## Data

**LINCS L1000** (Level-5 signatures, 978 landmark genes):

```bash
# Download from GEO accession GSE92742
# Expected file: data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx
```

Reproduce the compound-level aggregation used in the paper (N = 10,554 compounds after SMILES filtering and replicate averaging):

```bash
python scripts/prepare_data.py \
  --gctx-path /path/to/GSE92742.gctx \
  --out data/lincs_compound_level.parquet
```

**ChemBERTa encoder:** downloaded automatically from HuggingFace Hub (`seyonec/ChemBERTa-zinc-base-v1`) on first run. To use a local copy, pass `--model-dir /path/to/ChemBERTa`.

---

## Quick Start

**Algorithm sanity check** (no data required, runs in < 1 min):

```bash
python scripts/smoke_test.py
```

Expected output: dES values consistent with Table 3 (mean relative error vs. nyswin < 0.2%), Spearman(dNES, NES) > 0.85 across all synthetic scenarios.

**Full training reproduction:**

```bash
python scripts/run_training_repro.py \
  --data-path data/lincs_compound_level.parquet \
  --model-dir /path/to/ChemBERTa \
  --seed 42
```

Target: conclusion-level consistency with Tables 4–5 (not bitwise-identical checkpoints due to GPU nondeterminism). Training takes approximately 2–3 hours on a single A100.

---

## Repository Structure

```
dgsea-paper-code/
├── dgse/                     # Core library
│   ├── functional.py         # dES, dNES, nyswin (Algorithms 1–3)
│   ├── loss.py               # Hybrid training objective (Eq. 7–9)
│   └── normalize.py          # Sign-specific robust permutation normalization
├── scripts/
│   ├── smoke_test.py         # Algorithm sanity check
│   ├── prepare_data.py       # LINCS L1000 preprocessing
│   └── run_training_repro.py # Full training reproduction
├── configs/                  # Hyperparameter configs used in the paper
├── figures/
└── CITATION.cff
```

---

## Citation

If you use dGSEA in your work, please cite:

```bibtex
@article{li2026dgsea,
  title   = {Differentiable Gene Set Enrichment Analysis for Pathway-Level Supervision
             in Transcriptomic Learning},
  author  = {Li, Shuaiyu and Ruan, Yang and Yang, Xinyue and Zhang, Wen and Saigo, Hiroto},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.18.712610}
}
```

---

## Acknowledgements

Supported by JSPS KAKENHI Grant Number JP23H03356.

---

## License

MIT
