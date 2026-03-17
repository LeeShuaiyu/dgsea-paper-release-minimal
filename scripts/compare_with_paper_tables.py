#!/usr/bin/env python3
"""Compare reproduction core metrics against manuscript tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare reproduced core table with paper tables.")
    p.add_argument(
        "--core-table",
        type=Path,
        default=Path("results/training_repro/exp3a_table_test_models_core.csv"),
        help="Reproduced core metrics table.",
    )
    p.add_argument(
        "--paper-gene",
        type=Path,
        default=Path("results/paper_tables/exp3a_table4_gene_level_paper.csv"),
        help="Paper Table 4 (gene-level) csv.",
    )
    p.add_argument(
        "--paper-pathway",
        type=Path,
        default=Path("results/paper_tables/exp3a_table5_pathway_level_paper.csv"),
        help="Paper Table 5 (pathway-level) csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/training_repro"),
        help="Directory to write comparison csv files.",
    )
    return p.parse_args()


def normalize_paper(paper_gene: pd.DataFrame, paper_path: pd.DataFrame) -> pd.DataFrame:
    model_map = {
        "Baseline": "MSE-only",
        "DGSEA-only": "DGSEA-only",
        "Hybrid (ours)": "Hybrid(best)",
    }
    g = paper_gene.copy()
    p = paper_path.copy()
    g["model"] = g["objective"].map(model_map)
    p["model"] = p["objective"].map(model_map)
    merged = g.merge(p, on="model", how="inner", suffixes=("_gene", "_path"))
    out = pd.DataFrame(
        {
            "model": merged["model"],
            "ALL978_mean_r": merged["mean_r"],
            "Top50_mean_r": merged["top50_r"],
            "PATH_macro_corr": merged["correlation"],
            "PATH_macro_mse": merged["mse"],
            "PATH_macro_sign_acc": merged["sign_accuracy"],
        }
    )
    return out.set_index("model").sort_index()


def compute_hybrid_gains(df: pd.DataFrame) -> dict[str, float]:
    base = df.loc["MSE-only"]
    hyb = df.loc["Hybrid(best)"]
    return {
        "all978_gain": float(hyb["ALL978_mean_r"] - base["ALL978_mean_r"]),
        "top50_gain": float(hyb["Top50_mean_r"] - base["Top50_mean_r"]),
        "path_corr_gain": float(hyb["PATH_macro_corr"] - base["PATH_macro_corr"]),
        "path_sign_gain": float(hyb["PATH_macro_sign_acc"] - base["PATH_macro_sign_acc"]),
        "path_mse_reduction": float(base["PATH_macro_mse"] - hyb["PATH_macro_mse"]),
    }


def main() -> int:
    args = parse_args()
    for path in [args.core_table, args.paper_gene, args.paper_pathway]:
        if not path.exists():
            raise FileNotFoundError(path)

    repro = pd.read_csv(args.core_table).set_index("model").sort_index()
    paper_gene = pd.read_csv(args.paper_gene)
    paper_path = pd.read_csv(args.paper_pathway)
    paper = normalize_paper(paper_gene, paper_path)

    rows: list[dict[str, float | str]] = []
    metrics = ["ALL978_mean_r", "Top50_mean_r", "PATH_macro_corr", "PATH_macro_mse", "PATH_macro_sign_acc"]
    for model in ["MSE-only", "DGSEA-only", "Hybrid(best)"]:
        for metric in metrics:
            pv = float(paper.loc[model, metric])
            rv = float(repro.loc[model, metric])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "paper_value": pv,
                    "repro_value": rv,
                    "delta_repro_minus_paper": rv - pv,
                }
            )
    comp = pd.DataFrame(rows)
    gain = pd.DataFrame(
        [
            {"source": "paper", **compute_hybrid_gains(paper)},
            {"source": "repro", **compute_hybrid_gains(repro)},
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comp_path = args.out_dir / "comparison_paper_vs_repro_core.csv"
    gain_path = args.out_dir / "comparison_hybrid_gain_paper_vs_repro.csv"
    comp.to_csv(comp_path, index=False)
    gain.to_csv(gain_path, index=False)

    print("[compare] core:", comp_path)
    print("[compare] gain:", gain_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
