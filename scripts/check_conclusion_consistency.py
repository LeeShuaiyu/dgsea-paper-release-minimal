#!/usr/bin/env python3
"""Check whether trained results preserve paper-level conclusions.

Expected input: core comparison table with rows for MSE baseline and Hybrid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check conclusion-level consistency.")
    p.add_argument(
        "--core-table",
        type=Path,
        default=Path("results/training_repro/exp3a_table_test_models_core.csv"),
        help="Path to core comparison csv.",
    )
    p.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/conclusion_thresholds.json"),
        help="JSON file with threshold values.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/training_repro/conclusion_check.json"),
        help="Where to write machine-readable check result.",
    )
    return p.parse_args()


def _find_col(df: pd.DataFrame, names: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    raise KeyError(f"Cannot find any of columns: {names}")


def _pick_rows(df: pd.DataFrame, model_col: str) -> tuple[pd.Series, pd.Series]:
    names = df[model_col].astype(str).str.lower()
    idx_mse = names.str.contains("mse")
    idx_hyb = names.str.contains("hybrid")
    if not idx_mse.any():
        raise ValueError("Cannot find MSE row in model column.")
    if not idx_hyb.any():
        raise ValueError("Cannot find Hybrid row in model column.")
    base = df.loc[idx_mse].iloc[0]
    hyb = df.loc[idx_hyb].iloc[0]
    return base, hyb


def main() -> int:
    args = parse_args()
    if not args.core_table.exists():
        raise FileNotFoundError(f"Core table not found: {args.core_table}")
    if not args.thresholds.exists():
        raise FileNotFoundError(f"Threshold config not found: {args.thresholds}")

    thr = json.loads(args.thresholds.read_text(encoding="utf-8"))
    df = pd.read_csv(args.core_table)

    c_model = _find_col(df, ["model", "setting", "objective", "name"])
    c_all = _find_col(df, ["ALL978_mean_r", "all978_mean_r"])
    c_top = _find_col(df, ["Top50_mean_r", "top50_mean_r"])
    c_corr = _find_col(df, ["PATH_macro_corr", "path_macro_corr"])
    c_mse = _find_col(df, ["PATH_macro_mse", "path_macro_mse"])
    c_sign = _find_col(df, ["PATH_macro_sign_acc", "path_macro_sign_acc"])

    base, hyb = _pick_rows(df, c_model)

    delta = {
        "all978_gain": float(hyb[c_all] - base[c_all]),
        "top50_gain": float(hyb[c_top] - base[c_top]),
        "path_corr_gain": float(hyb[c_corr] - base[c_corr]),
        "path_sign_gain": float(hyb[c_sign] - base[c_sign]),
        "path_mse_reduction": float(base[c_mse] - hyb[c_mse]),
    }

    checks = {
        "path_corr_improved": delta["path_corr_gain"] >= float(thr["min_path_corr_gain"]),
        "path_sign_improved": delta["path_sign_gain"] >= float(thr["min_path_sign_gain"]),
        "path_mse_reduced": delta["path_mse_reduction"] >= float(thr["min_path_mse_reduction"]),
        "gene_not_degraded_all978": delta["all978_gain"] >= -float(thr["max_all978_drop"]),
        "gene_not_degraded_top50": delta["top50_gain"] >= -float(thr["max_top50_drop"]),
    }

    passed = all(checks.values())
    result = {
        "core_table": str(args.core_table),
        "thresholds": thr,
        "delta": delta,
        "checks": checks,
        "passed": passed,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[check] core table:", args.core_table)
    print("[check] deltas:", json.dumps(delta, indent=2))
    for k, v in checks.items():
        print(f"[check] {k}: {'PASS' if v else 'FAIL'}")
    print(f"[check] overall: {'PASS' if passed else 'FAIL'}")
    print("[check] report:", args.out_json)

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

