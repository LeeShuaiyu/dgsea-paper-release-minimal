# Checkpoints Folder

Place model checkpoints used for manuscript evaluation in this folder.

Recommended files:

- `exp3a_mse_best.pt`
- `exp3a_dgsea_best.pt`
- `exp3a_hybrid_lam*.pt` (lambda sweep checkpoints)
- `exp3a_hybrid_<best>.pt` or an explicit record of the best lambda from the sweep

For archival reproducibility, provide SHA256 checksums in `docs/reproducibility.md`.
