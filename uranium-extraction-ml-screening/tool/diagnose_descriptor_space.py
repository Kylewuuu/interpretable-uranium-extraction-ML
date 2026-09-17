from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/home/wyh/data/examples/U_logD")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose scale or calculation mismatches among training-ligand, "
            "complete-PubChem, and screened-candidate descriptor tables."
        )
    )
    parser.add_argument(
        "--training-descriptors",
        type=Path,
        default=BASE_DIR / "data/input/ligand_des.xlsx",
    )
    parser.add_argument(
        "--selected-summary",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/selected_descriptor_similarity_summary.json",
    )
    parser.add_argument(
        "--complete-cache",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/complete_pubchem_sample_selected_descriptors.csv",
    )
    parser.add_argument(
        "--screened-descriptor-dir",
        type=Path,
        default=BASE_DIR / "pre_data/input/pub_smi_des",
    )
    parser.add_argument(
        "--screened-sample-size", type=int, default=20_000
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/descriptor_space_diagnostic.csv",
    )
    return parser.parse_args()


def load_selected(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = list(summary["feature_selection"]["selected_descriptors"])
    if not selected:
        raise ValueError("No selected descriptors were found.")
    return selected


def sample_screened(
    directory: Path, selected: list[str], sample_size: int
) -> pd.DataFrame:
    files = sorted(directory.glob("desc_part_*.csv"))
    if not files:
        files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No descriptor CSV files were found in {directory}")

    rng = np.random.default_rng(RANDOM_STATE)
    reservoir = pd.DataFrame(columns=[*selected, "_priority"])
    processed = 0
    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=lambda column: column in selected,
            chunksize=100_000,
        ):
            missing = [name for name in selected if name not in chunk.columns]
            if missing:
                raise KeyError(f"{path} is missing descriptors: {missing}")
            chunk = chunk[selected].apply(pd.to_numeric, errors="coerce")
            chunk["_priority"] = rng.random(len(chunk))
            reservoir = pd.concat([reservoir, chunk], ignore_index=True)
            if len(reservoir) > sample_size:
                priorities = reservoir["_priority"].to_numpy(dtype=float)
                keep = np.argpartition(priorities, sample_size - 1)[:sample_size]
                reservoir = reservoir.iloc[keep].reset_index(drop=True)
            processed += len(chunk)
    print(f"Screened rows examined: {processed:,}")
    print(f"Screened rows sampled: {len(reservoir):,}")
    return reservoir.drop(columns="_priority")


def finite_frame(frame: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    result = frame[selected].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def quantile(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    return float(clean.quantile(value)) if len(clean) else float("nan")


def main() -> None:
    args = parse_args()
    selected = load_selected(args.selected_summary)

    training = finite_frame(
        pd.read_excel(args.training_descriptors, usecols=selected), selected
    )
    # Molecular descriptors are repeated across experimental records. This
    # produces one profile for each unique stored ligand descriptor vector.
    training_unique = training.drop_duplicates().reset_index(drop=True)
    complete = finite_frame(pd.read_csv(args.complete_cache), selected)
    screened = finite_frame(
        sample_screened(
            args.screened_descriptor_dir, selected, args.screened_sample_size
        ),
        selected,
    )

    rows: list[dict[str, float | str]] = []
    for name in selected:
        train_col = training_unique[name]
        complete_col = complete[name]
        screened_col = screened[name]
        train_mean = float(train_col.mean())
        train_std = float(train_col.std(ddof=0))
        if not np.isfinite(train_std) or train_std == 0:
            train_std = float("nan")

        complete_median = quantile(complete_col, 0.50)
        screened_median = quantile(screened_col, 0.50)
        rows.append(
            {
                "descriptor": name,
                "training_unique_count": int(training_unique[name].notna().sum()),
                "training_q01": quantile(train_col, 0.01),
                "training_median": quantile(train_col, 0.50),
                "training_q99": quantile(train_col, 0.99),
                "training_mean": train_mean,
                "training_std": train_std,
                "complete_q01": quantile(complete_col, 0.01),
                "complete_median": complete_median,
                "complete_q99": quantile(complete_col, 0.99),
                "screened_q01": quantile(screened_col, 0.01),
                "screened_median": screened_median,
                "screened_q99": quantile(screened_col, 0.99),
                "complete_median_abs_z": abs(
                    (complete_median - train_mean) / train_std
                ),
                "screened_median_abs_z": abs(
                    (screened_median - train_mean) / train_std
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["largest_median_abs_z"] = result[
        ["complete_median_abs_z", "screened_median_abs_z"]
    ].max(axis=1)
    result = result.sort_values(
        "largest_median_abs_z", ascending=False
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Training descriptor rows: {len(training):,}")
    print(f"Unique stored training profiles: {len(training_unique):,}")
    print(f"Complete-PubChem sample rows: {len(complete):,}")
    print("\nDescriptors producing the largest median standardized shifts:")
    columns = [
        "descriptor",
        "training_median",
        "complete_median",
        "screened_median",
        "complete_median_abs_z",
        "screened_median_abs_z",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(result[columns].head(12).to_string(index=False))
    print(f"\nFull diagnostic table: {args.output}")


if __name__ == "__main__":
    main()
