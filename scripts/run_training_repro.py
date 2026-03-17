#!/usr/bin/env python3
"""Execute training reproducibility notebook with environment overrides."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exp3a training reproducibility notebook.")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/exp3a_release_repro.ipynb"),
    )
    p.add_argument(
        "--executed-notebook",
        type=Path,
        default=Path("notebooks/exp3a_release_repro.executed.ipynb"),
    )

    p.add_argument("--data-path", type=str, required=True, help="Path to L1000 parquet.")
    p.add_argument(
        "--model-dir",
        type=str,
        default="ChemBERTa",
        help="Local model directory (recommended) or HF model id.",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force local model loading only (no HF download).",
    )
    p.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow HuggingFace download when model dir does not exist locally.",
    )
    p.add_argument("--out-dir", type=str, default="runs/chemberta_exp3a")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--epochs-mse", type=int, default=5)
    p.add_argument("--epochs-dgsea", type=int, default=5)
    p.add_argument("--epochs-hybrid", type=int, default=4)
    p.add_argument("--pathway-key", type=str, default="P53_pathway")
    p.add_argument(
        "--filter-invalid-smiles",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to filter invalid SMILES with RDKit (1=yes, 0=no).",
    )
    p.add_argument(
        "--use-fp",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to compute Morgan fingerprints with RDKit (1=yes, 0=no).",
    )
    p.add_argument("--timeout", type=int, default=0, help="nbconvert timeout (0 = unlimited).")
    p.add_argument(
        "--check-conclusion",
        action="store_true",
        help="Run consistency checker after notebook execution.",
    )
    return p.parse_args()


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def ensure_rdkit_available(python_bin: str) -> None:
    cmd = [python_bin, "-c", "import rdkit"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(
            "RDKit is required by the current settings but is not available. "
            "Install RDKit, or rerun with --filter-invalid-smiles 0 and --use-fp 0.\n"
            f"Import error: {msg}"
        ) from exc


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    nb = (repo / args.notebook).resolve()
    out_nb = (repo / args.executed_notebook).resolve()

    if not nb.exists():
        raise FileNotFoundError(f"Notebook not found: {nb}")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    model_path = Path(args.model_dir)
    model_exists_local = model_path.exists()
    if not model_exists_local and not args.allow_hf_download:
        raise FileNotFoundError(
            f"Model path not found locally: {args.model_dir}. "
            "Provide a local model directory via --model-dir, or pass --allow-hf-download."
        )

    pub_results_dir = (repo / "results" / "training_repro").resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()

    if args.filter_invalid_smiles == 1 or args.use_fp == 1:
        ensure_rdkit_available(sys.executable)

    env = os.environ.copy()
    env.update(
        {
            "L1000_PARQUET": args.data_path,
            "CHEMBERTA_DIR": args.model_dir,
            "LOCAL_FILES_ONLY": "1" if (args.local_files_only or model_exists_local) else "0",
            "OUT_DIR": str(out_dir),
            "DEVICE": args.device,
            "SEED": str(args.seed),
            "BATCH_SIZE": str(args.batch_size),
            "NUM_WORKERS": str(args.num_workers),
            "MAX_SAMPLES": str(args.max_samples),
            "EPOCHS_MSE": str(args.epochs_mse),
            "EPOCHS_DGSEA": str(args.epochs_dgsea),
            "EPOCHS_HYBRID": str(args.epochs_hybrid),
            "PATHWAY_KEY": args.pathway_key,
            "FILTER_INVALID_SMILES": str(args.filter_invalid_smiles),
            "USE_FP": str(args.use_fp),
            "PUB_RESULTS_DIR": str(pub_results_dir),
        }
    )

    timeout = args.timeout if args.timeout > 0 else -1
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb),
        "--output",
        str(out_nb),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    run_cmd(cmd, cwd=repo, env=env)

    # Some notebook execution contexts use the notebook directory as cwd.
    # If relative export paths were used, recover artifacts into publish dir.
    core_name = "exp3a_table_test_models_core.csv"
    per_path_name = "exp3a_table_test_per_pathway.csv"
    core_pub = pub_results_dir / core_name
    per_path_pub = pub_results_dir / per_path_name
    if not core_pub.exists() or not per_path_pub.exists():
        alt_dir = (repo / "notebooks" / "results" / "training_repro").resolve()
        alt_core = alt_dir / core_name
        alt_per_path = alt_dir / per_path_name
        if alt_core.exists() and alt_per_path.exists():
            pub_results_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt_core, core_pub)
            shutil.copy2(alt_per_path, per_path_pub)
            print("[fixup] copied exports from notebooks/results/training_repro -> results/training_repro")

    if args.check_conclusion:
        check_cmd = [
            sys.executable,
            "scripts/check_conclusion_consistency.py",
            "--core-table",
            str(core_pub),
        ]
        run_cmd(check_cmd, cwd=repo, env=env)

    print("[done] executed notebook:", out_nb)
    print("[done] core table: results/training_repro/exp3a_table_test_models_core.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
