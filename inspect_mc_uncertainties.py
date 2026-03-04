#!/usr/bin/env python3
"""
Inspect systematic uncertainties across MC samples (no data, no signal) in the
bamboo output directory. It ranks systematics by their total impact on the
integrated yield of a chosen histogram (default: DL_ml_score).

Outputs: CSV and JSON with per-system total/relative impacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import uproot

# Defaults
HIST_NAME = "SR_GGF_resolved1b"
# HIST_NAME = "DL_resolved_1b_InvM_ee"
OUT_PREFIX = "mc_uncertainties"

# Identify files to skip (data and signals)
DATA_KEYWORDS = ["data_"]
SIGNAL_KEYWORDS = [
    "ggHH_",
    "qqHH",
    "ttH",
    "WplusH",
    "ZH",
    "GluGluHto2Wto2L2Nu",
    "VBFHto2Wto2L2Nu",
]


def is_data_or_signal(name: str) -> bool:
    return any(k in name for k in DATA_KEYWORDS) or any(k in name for k in SIGNAL_KEYWORDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank MC systematics by yield impact")
    parser.add_argument("--base-dir", type=Path, help="Directory with ROOT files")
    parser.add_argument("--hist", default=HIST_NAME, help="Histogram name to inspect")
    parser.add_argument("--out-prefix", default=OUT_PREFIX, help="Prefix for CSV/JSON outputs")
    parser.add_argument("--top", type=int, default=100, help="How many top systematics to print")
    return parser.parse_args()


def load_hist(path: Path, hist_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as f:
        h = f[hist_name]
        counts, edges = h.to_numpy()
    return counts.astype(float), edges.astype(float)


def collect_systematics(path: Path, hist_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with uproot.open(path) as f:
        if hist_name not in f:
            return {}
        nominal, _ = f[hist_name].to_numpy()
        for key in f.keys():
            k = key.split(";")[0]
            if "__" not in k:
                continue
            if not (k.endswith("up") or k.endswith("down")):
                continue
            base, var = k.split("__", 1)
            if base != hist_name:
                continue
            direction = "up" if var.endswith("up") else "down"
            syst = var[: -len(direction)]
            counts, _ = f[key].to_numpy()
            out.setdefault(syst, {})[direction] = counts.astype(float)
        out["nominal"] = {"nominal": nominal.astype(float)}
    return out


def aggregate_impacts(
    files: List[Path], hist_name: str
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    syst_diff_sum: Dict[str, float] = {}
    syst_nom_sum: Dict[str, float] = {}
    per_process: Dict[str, Dict[str, float]] = {}

    for path in files:
        systs = collect_systematics(path, hist_name)
        if not systs:
            continue
        nominal = systs.pop("nominal")["nominal"]
        nom_yield = float(np.clip(nominal, 0.0, None).sum())
        proc = path.name.replace(".root", "")

        for syst, updown in systs.items():
            up = updown.get("up")
            down = updown.get("down")
            if up is None and down is None:
                continue
            impacts = []
            for var in (up, down):
                if var is None:
                    continue
                impacts.append(float(np.clip(var, 0.0, None).sum()) - nom_yield)
            if not impacts:
                continue
            avg_abs = float(np.mean(np.abs(impacts)))
            syst_diff_sum[syst] = syst_diff_sum.get(syst, 0.0) + avg_abs
            syst_nom_sum[syst] = syst_nom_sum.get(syst, 0.0) + nom_yield
            per_process.setdefault(proc, {})[syst] = avg_abs

    return syst_diff_sum, syst_nom_sum, per_process


def rank_systematics(syst_diff_sum: Dict[str, float], syst_nom_sum: Dict[str, float]) -> List[dict]:
    rows = []
    for syst, diff in syst_diff_sum.items():
        nom = syst_nom_sum.get(syst, 0.0)
        rel = diff / nom if nom > 0 else 0.0
        rows.append({"syst": syst, "abs_diff": diff, "rel": rel})
    rows.sort(key=lambda r: r["rel"], reverse=True)
    return rows


def write_outputs(prefix: str, rows: List[dict]) -> None:
    csv_path = Path(f"{prefix}.csv")
    json_path = Path(f"{prefix}.json")

    with csv_path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["rank", "syst", "abs_diff", "rel"])
        writer.writeheader()
        for i, r in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": i,
                    "syst": r["syst"],
                    "abs_diff": f"{r['abs_diff']:.6f}",
                    "rel": f"{r['rel']:.6f}",
                }
            )

    with json_path.open("w") as h:
        json.dump(rows, h, indent=2)

    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


def main() -> None:
    args = parse_args()
    files = [p for p in args.base_dir.glob("*.root") if not is_data_or_signal(p.name)]
    if not files:
        raise RuntimeError("No MC files found (after filtering data/signals)")

    syst_diff_sum, syst_nom_sum, _ = aggregate_impacts(files, args.hist)
    ranked = rank_systematics(syst_diff_sum, syst_nom_sum)

    top = args.top
    print("Top systematics by relative yield impact:")
    for i, r in enumerate(ranked[:top], 1):
        print(f"{i:2d}. {r['syst']:<40} rel={r['rel']:.6f} abs={r['abs_diff']:.2f}")

    write_outputs(args.out_prefix, ranked)


if __name__ == "__main__":
    main()
