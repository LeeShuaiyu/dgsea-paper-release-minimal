#!/usr/bin/env python3
"""CPU-friendly smoke test for dGSEA paper release.

Default checks:
1) core module import and backend selection
2) synthetic score computation

Optional (--run-notebook): execute reduced training reproducibility run,
which requires a real parquet data path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run release smoke test.")
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to repository root.",
    )
    p.add_argument(
        "--run-notebook",
        action="store_true",
        help="Execute reduced training notebook (requires real data).",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Path to L1000 parquet (required with --run-notebook).",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="Local model directory or HF model id.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Reduced run sample cap when --run-notebook is set.",
    )
    p.add_argument(
        "--notebook-timeout",
        type=int,
        default=7200,
        help="Notebook execution timeout in seconds.",
    )
    p.add_argument(
        "--run-checker",
        action="store_true",
        help="Run conclusion checker after reduced notebook run (off by default).",
    )
    return p.parse_args()


def check_core(repo: Path) -> None:
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import dgsea_backend as DG  # type: ignore

    DG.set_backend("auto")
    backend = DG.backend()
    print(f"[smoke] backend={backend}")

    rng = np.random.default_rng(7)
    g = np.zeros(256, dtype=int)
    g[rng.choice(256, size=24, replace=False)] = 1
    s = rng.normal(size=256)

    perms = DG.shared_permutations(256, 128, seed=7)
    nes, es, p = DG.classical_gsea_nes_with_perms(
        s=s, g=g, p=1.0, perms=perms, trim=0.1, shrink_lambda=0.1, split_ratio=0.5
    )
    dnes, des, dp = DG.dgsea_dnes_with_perms(
        s=s,
        g=g,
        p=1.0,
        tau_rank=0.8,
        tau_prefix=1.1,
        tau_abs=0.7,
        perms=perms,
        trim=0.1,
        shrink_lambda=0.1,
        split_ratio=0.5,
        calibrate_kappa=False,
        variant="nyswin",
        m=128,
        frac=0.1,
        margin=50,
        chunk_size=128,
    )

    vals = [float(np.asarray(x).reshape(-1)[0]) for x in [nes, es, p, dnes, des, dp]]
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Non-finite scores found: {vals}")
    print("[smoke] core scores are finite")


def run_reduced_notebook(repo: Path, args: argparse.Namespace) -> None:
    if not args.data_path:
        raise ValueError("--data-path is required when --run-notebook is set.")

    cmd = [
        sys.executable,
        "scripts/run_training_repro.py",
        "--repo",
        str(repo),
        "--data-path",
        args.data_path,
        "--model-dir",
        args.model_dir,
        "--allow-hf-download",
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--num-workers",
        "0",
        "--max-samples",
        str(args.max_samples),
        "--epochs-mse",
        "1",
        "--epochs-dgsea",
        "1",
        "--epochs-hybrid",
        "1",
        "--filter-invalid-smiles",
        "0",
        "--use-fp",
        "0",
        "--timeout",
        str(args.notebook_timeout),
    ]
    if args.run_checker:
        cmd.append("--check-conclusion")

    print("[smoke] running reduced notebook training check")
    subprocess.run(cmd, cwd=str(repo), check=True)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    print(f"[smoke] repo={repo}")
    check_core(repo)
    if args.run_notebook:
        run_reduced_notebook(repo, args)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
