from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")


BASE_DIR = Path("/home/wyh/data/examples/U_logD")
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SCREENED = 1_178_295
PUBCHEM_WORKING_COUNT = 122_207_703
DEFAULT_SAMPLING_FRACTION = 0.0002  # 0.02%
AD_QUANTILE = 0.95

THREE_D_DESCRIPTORS = {
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "Asphericity",
    "Eccentricity",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Joint t-SNE visualization of development ligands, screened PubChem "
            "candidates, and the complete PubChem working collection before screening."
        )
    )
    parser.add_argument(
        "--training-descriptors",
        type=Path,
        default=BASE_DIR / "data/input/ligand_des.xlsx",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=BASE_DIR / "data/input/y.xlsx",
    )
    parser.add_argument(
        "--training-records",
        type=Path,
        default=BASE_DIR / "data/shuffled_total.xlsx",
    )
    parser.add_argument("--ligand-smiles-column", default="lig_SMI")
    parser.add_argument(
        "--selected-summary",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/selected_descriptor_similarity_summary.json",
    )
    parser.add_argument(
        "--pubchem-input",
        type=Path,
        default=BASE_DIR / "reviewer3_comment4/complete_pubchem_raw_sample.csv",
        help="The 122,207,703-record CID-SMILES working file (CSV or CSV.GZ).",
    )
    parser.add_argument(
        "--screened-descriptor-dir",
        type=Path,
        default=BASE_DIR / "pre_data/input/pub_smi_des",
        help="Directory containing descriptor CSV files for screened candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "reviewer3_comment4",
    )
    parser.add_argument("--n-screened", type=int, default=N_SCREENED)
    parser.add_argument(
        "--pubchem-working-count",
        type=int,
        default=PUBCHEM_WORKING_COUNT,
        help="Number of molecular records in the pre-screening PubChem working file.",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=DEFAULT_SAMPLING_FRACTION,
        help=(
            "Common sampling fraction applied independently to the complete "
            "pre-screening PubChem collection and the screened-candidate pool "
            "(default: 0.0002 = 0.02 percent)."
        ),
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=2.0,
        help=(
            "Raw complete-PubChem molecules sampled per requested plotted molecule; "
            "allows for invalid structures or failed 3D embedding."
        ),
    )
    parser.add_argument("--read-chunk-size", type=int, default=500_000)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached complete-PubChem sample descriptors.",
    )
    return parser.parse_args()


def load_selected_descriptors(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = list(summary["feature_selection"]["selected_descriptors"])
    if not selected:
        raise ValueError("No selected descriptors were found in the summary file.")
    return selected


def canonicalize_smiles(value: object) -> str:
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Invalid development-ligand SMILES: {value!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_development_reference(
    descriptor_path: Path,
    target_path: Path,
    record_path: Path,
    smiles_column: str,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    descriptors = pd.read_excel(descriptor_path, usecols=selected)
    targets = pd.read_excel(target_path)
    records = pd.read_excel(record_path, usecols=[smiles_column])
    if "logD" not in targets.columns:
        raise KeyError("The target file must contain a 'logD' column.")
    if not (len(descriptors) == len(targets) == len(records)):
        raise ValueError("Training descriptor, target, and record tables differ in length.")

    valid_indices = np.flatnonzero(
        pd.to_numeric(targets["logD"], errors="coerce").notna()
    )
    development_indices, _ = train_test_split(
        valid_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    development = descriptors.iloc[development_indices].to_numpy(dtype=float)
    means = np.nanmean(development, axis=0)
    missing = ~np.isfinite(development)
    if missing.any():
        development[missing] = np.take(means, np.where(missing)[1])

    ligand_ids = (
        records.iloc[development_indices][smiles_column]
        .map(canonicalize_smiles)
        .reset_index(drop=True)
    )
    grouped = pd.DataFrame(development, columns=selected)
    grouped.insert(0, "canonical_ligand", ligand_ids)
    reference = (
        grouped.groupby("canonical_ligand", sort=True)
        .median(numeric_only=True)
        .to_numpy(dtype=float)
    )
    return reference, means


def list_csv_files(directory: Path, preferred_pattern: str) -> list[Path]:
    files = sorted(directory.glob(preferred_pattern))
    if not files:
        files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files were found in {directory}")
    return files


def normalize_pubchem_columns(columns: list[str]) -> tuple[str, str]:
    mapping = {str(column).strip().upper(): str(column) for column in columns}
    if "CID" not in mapping or "SMILES" not in mapping:
        raise KeyError("The PubChem input must contain CID and SMILES columns.")
    return mapping["CID"], mapping["SMILES"]


def uniform_priority_sample(
    current: pd.DataFrame,
    new_rows: pd.DataFrame,
    priorities: np.ndarray,
    target_size: int,
) -> pd.DataFrame:
    additions = new_rows.copy()
    additions["_priority"] = priorities
    combined = pd.concat([current, additions], ignore_index=True)
    if len(combined) > target_size:
        keep = np.argpartition(
            combined["_priority"].to_numpy(), target_size - 1
        )[:target_size]
        combined = combined.iloc[keep].reset_index(drop=True)
    return combined


def sample_complete_pubchem(
    input_path: Path,
    raw_sample_size: int,
    chunk_size: int,
) -> pd.DataFrame:
    compression = "gzip" if input_path.suffix.lower() == ".gz" else None
    header = pd.read_csv(input_path, compression=compression, nrows=0)
    cid_column, smiles_column = normalize_pubchem_columns(list(header.columns))
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir = pd.DataFrame(columns=["CID", "SMILES", "_priority"])
    processed = 0
    usable = 0

    reader = pd.read_csv(
        input_path,
        compression=compression,
        usecols=[cid_column, smiles_column],
        dtype={cid_column: str, smiles_column: str},
        chunksize=chunk_size,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns={cid_column: "CID", smiles_column: "SMILES"})
        processed += len(chunk)
        chunk = chunk[["CID", "SMILES"]]
        chunk = chunk.dropna(subset=["CID", "SMILES"])
        usable += len(chunk)
        if len(chunk):
            reservoir = uniform_priority_sample(
                reservoir,
                chunk,
                rng.random(len(chunk)),
                raw_sample_size,
            )
        if chunk_number % 20 == 0:
            print(
                f"PubChem chunks={chunk_number:,}; processed={processed:,}; "
                f"usable rows={usable:,}"
            )

    if len(reservoir) < raw_sample_size:
        raise ValueError(
            f"Only {len(reservoir):,} complete-PubChem rows were available for sampling."
        )
    return reservoir.sort_values("_priority").drop(columns="_priority").reset_index(drop=True)


def calculate_selected_descriptors(
    sampled: pd.DataFrame,
    selected: list[str],
    requested_size: int,
) -> pd.DataFrame:
    base_names = [name.removeprefix("lig_") for name in selected]
    rows: list[dict[str, object]] = []
    for position, record in sampled.iterrows():
        smiles = str(record["SMILES"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=RANDOM_STATE) != 0:
                continue
            AllChem.MMFFOptimizeMolecule(mol)
            desc2d = CalcMolDescriptors(mol)
            result: dict[str, object] = {
                "CID": str(record["CID"]),
                "SMILES": smiles,
            }
            valid = True
            for full_name, base_name in zip(selected, base_names):
                if base_name in THREE_D_DESCRIPTORS:
                    value = getattr(Descriptors3D, base_name)(mol)
                else:
                    value = desc2d.get(base_name, np.nan)
                value = float(value)
                if not np.isfinite(value):
                    valid = False
                    break
                result[full_name] = value
            if valid:
                rows.append(result)
        except Exception:
            continue

        if (position + 1) % 500 == 0:
            print(
                f"Complete-PubChem descriptors: examined={position + 1:,}; "
                f"valid={len(rows):,}"
            )
        if len(rows) >= requested_size:
            break

    if len(rows) < requested_size:
        raise ValueError(
            f"Only {len(rows):,} valid complete-PubChem descriptor rows were obtained. "
            "Increase --oversample-factor and rerun with --force-recompute."
        )
    return pd.DataFrame(rows[:requested_size])


def load_or_create_complete_pubchem_descriptors(
    args: argparse.Namespace,
    selected: list[str],
    requested_size: int,
) -> np.ndarray:
    cache = args.output_dir / "complete_pubchem_sample_selected_descriptors.csv"
    if cache.exists() and not args.force_recompute:
        cached = pd.read_csv(cache)
        missing = [name for name in selected if name not in cached.columns]
        if not missing and len(cached) >= requested_size:
            return cached.iloc[:requested_size][selected].to_numpy(dtype=float)

    raw_size = int(np.ceil(requested_size * args.oversample_factor))
    sampled = sample_complete_pubchem(
        args.pubchem_input,
        raw_size,
        args.read_chunk_size,
    )
    sampled.to_csv(args.output_dir / "complete_pubchem_raw_sample.csv", index=False)
    calculated = calculate_selected_descriptors(sampled, selected, requested_size)
    calculated.to_csv(cache, index=False)
    return calculated[selected].to_numpy(dtype=float)


def sample_screened_descriptors(
    directory: Path,
    selected: list[str],
    imputation_means: np.ndarray,
    n_screened: int,
    sample_size: int,
) -> np.ndarray:
    files = list_csv_files(directory, "desc_part_*.csv")
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir_values = np.empty((0, len(selected)), dtype=float)
    reservoir_priorities = np.empty(0, dtype=float)
    processed = 0

    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=lambda name: name in selected,
            chunksize=100_000,
        ):
            remaining = n_screened - processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            missing_columns = [name for name in selected if name not in chunk.columns]
            if missing_columns:
                raise KeyError(f"{path} is missing descriptors: {missing_columns}")
            values = chunk[selected].to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            if missing.any():
                values[missing] = np.take(imputation_means, np.where(missing)[1])
            priorities = rng.random(len(values))
            reservoir_values = np.vstack([reservoir_values, values])
            reservoir_priorities = np.concatenate([reservoir_priorities, priorities])
            if len(reservoir_priorities) > sample_size:
                keep = np.argpartition(reservoir_priorities, sample_size - 1)[:sample_size]
                reservoir_values = reservoir_values[keep]
                reservoir_priorities = reservoir_priorities[keep]
            processed += len(values)
        if processed >= n_screened:
            break

    if processed != n_screened:
        raise ValueError(
            f"Expected {n_screened:,} screened descriptor rows but read {processed:,}."
        )
    if len(reservoir_values) != sample_size:
        raise ValueError(
            f"Requested {sample_size:,} screened candidates but sampled "
            f"{len(reservoir_values):,}."
        )
    return reservoir_values


def impute(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    if missing.any():
        result[missing] = np.take(means, np.where(missing)[1])
    return result


def proximity_statistics(
    reference: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, object]:
    scaler = StandardScaler().fit(reference)
    reference_z = scaler.transform(reference)
    loo = NearestNeighbors(n_neighbors=2).fit(reference_z)
    loo_distances, _ = loo.kneighbors(reference_z)
    loo_rms = loo_distances[:, 1] / np.sqrt(reference_z.shape[1])
    threshold = float(np.quantile(loo_rms, AD_QUANTILE))
    nearest = NearestNeighbors(n_neighbors=1).fit(reference_z)

    result: dict[str, object] = {
        "boundary_definition": (
            "95th percentile of leave-one-out nearest-neighbour RMS standardized "
            "distances among unique development ligands"
        ),
        "distance_threshold": threshold,
        "groups": {},
    }
    for name, values in groups.items():
        distances, _ = nearest.kneighbors(scaler.transform(values))
        rms = distances[:, 0] / np.sqrt(reference_z.shape[1])
        result["groups"][name] = {
            "sample_size": int(len(values)),
            "median_distance": float(np.median(rms)),
            "median_similarity": float(np.median(1.0 / (1.0 + rms))),
            "sample_inside_count": int(np.sum(rms <= threshold)),
            "sample_inside_fraction": float(np.mean(rms <= threshold)),
        }
    return result


def make_joint_tsne(
    reference: np.ndarray,
    complete_pubchem: np.ndarray,
    screened: np.ndarray,
    perplexity: float,
    output_dir: Path,
) -> None:
    scaler = StandardScaler().fit(reference)
    reference_z = np.clip(scaler.transform(reference), -8.0, 8.0)
    complete_pubchem_z = np.clip(scaler.transform(complete_pubchem), -8.0, 8.0)
    screened_z = np.clip(scaler.transform(screened), -8.0, 8.0)
    combined = np.vstack([complete_pubchem_z, screened_z, reference_z])

    effective_perplexity = min(perplexity, max(5.0, (len(combined) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=RANDOM_STATE,
        method="barnes_hut",
        angle=0.5,
        verbose=1,
    ).fit_transform(combined)

    n_complete_pubchem = len(complete_pubchem_z)
    n_screened = len(screened_z)
    complete_pubchem_xy = embedding[:n_complete_pubchem]
    screened_xy = embedding[n_complete_pubchem : n_complete_pubchem + n_screened]
    reference_xy = embedding[n_complete_pubchem + n_screened :]

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8}
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(
        complete_pubchem_xy[:, 0],
        complete_pubchem_xy[:, 1],
        s=5,
        color="#C9CED3",
        alpha=0.22,
        linewidths=0,
        label="Complete PubChem collection",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        screened_xy[:, 0],
        screened_xy[:, 1],
        s=7,
        color="#0072B2",
        alpha=0.40,
        linewidths=0,
        label="Screened PubChem candidates",
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        reference_xy[:, 0],
        reference_xy[:, 1],
        s=62,
        marker="*",
        color="#D55E00",
        edgecolors="white",
        linewidths=0.5,
        label="Development-set ligands",
        zorder=4,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("PubChem chemical space before and after screening", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(
        output_dir / "Figure_S_pubchem_screening_tSNE_v2.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Figure_S_pubchem_screening_tSNE_v2.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    labels = np.concatenate(
        [
            np.full(n_complete_pubchem, "complete_pubchem_sample", dtype=object),
            np.full(n_screened, "screened_candidate_sample", dtype=object),
            np.full(len(reference_z), "development_ligands_all", dtype=object),
        ]
    )
    pd.DataFrame(
        {"tSNE1": embedding[:, 0], "tSNE2": embedding[:, 1], "group": labels}
    ).to_csv(output_dir / "pubchem_screening_tSNE_v2_coordinates.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.sampling_fraction <= 1.0:
        raise ValueError("--sampling-fraction must be greater than 0 and no greater than 1.")
    if args.pubchem_working_count <= args.n_screened:
        raise ValueError("The PubChem working count must exceed the screened count.")

    complete_pubchem_sample_size = max(
        1, int(round(args.pubchem_working_count * args.sampling_fraction))
    )
    screened_sample_size = max(
        1, int(round(args.n_screened * args.sampling_fraction))
    )
    print(
        f"Common sampling fraction: {args.sampling_fraction:.6%}; "
        f"complete PubChem sample={complete_pubchem_sample_size:,}; "
        f"screened-candidate sample={screened_sample_size:,}"
    )

    selected = load_selected_descriptors(args.selected_summary)
    reference, imputation_means = build_development_reference(
        args.training_descriptors,
        args.target_file,
        args.training_records,
        args.ligand_smiles_column,
        selected,
    )
    complete_pubchem = load_or_create_complete_pubchem_descriptors(
        args,
        selected,
        complete_pubchem_sample_size,
    )
    complete_pubchem = impute(complete_pubchem, imputation_means)
    screened = sample_screened_descriptors(
        args.screened_descriptor_dir,
        selected,
        imputation_means,
        args.n_screened,
        screened_sample_size,
    )

    statistics = proximity_statistics(
        reference,
        {
            "complete_pubchem_sample": complete_pubchem,
            "screened_candidate_sample": screened,
        },
    )
    statistics.update(
        {
            "selected_descriptor_count": len(selected),
            "selected_descriptors": selected,
            "random_seed": RANDOM_STATE,
            "common_sampling_fraction": args.sampling_fraction,
            "pubchem_working_population": args.pubchem_working_count,
            "screened_candidate_population": args.n_screened,
            "complete_pubchem_sample_size": complete_pubchem_sample_size,
            "screened_candidate_sample_size": screened_sample_size,
            "unique_development_ligand_count": len(reference),
            "note": (
                "t-SNE is used only for visualization. Quantitative proximity is "
                "calculated in the original standardized descriptor space."
            ),
        }
    )
    with open(
        args.output_dir / "pubchem_screening_tSNE_v2_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(statistics, handle, indent=2, ensure_ascii=False)

    make_joint_tsne(
        reference,
        complete_pubchem,
        screened,
        args.perplexity,
        args.output_dir,
    )
    print(f"Selected descriptors: {len(selected)}")
    print(f"Unique development ligands plotted: {len(reference)}")
    print(f"Complete PubChem molecules plotted: {len(complete_pubchem)}")
    print(f"Screened PubChem candidates plotted: {len(screened)}")
    print(
        "Figure: "
        f"{args.output_dir / 'Figure_S_pubchem_screening_tSNE_v2.png'}"
    )
    print(
        "Statistics: "
        f"{args.output_dir / 'pubchem_screening_tSNE_v2_summary.json'}"
    )


if __name__ == "__main__":
    main()
