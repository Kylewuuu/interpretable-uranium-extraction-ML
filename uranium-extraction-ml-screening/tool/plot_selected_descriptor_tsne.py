from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")


BASE_DIR = Path("/home/kylewu/hotpot/examples/U_logD")
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_CANDIDATE_SAMPLE = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a t-SNE-style chemical-space plot from the important, "
            "nonredundant ligand descriptors selected in the similarity analysis."
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
        "--candidate-descriptor-dir",
        type=Path,
        default=BASE_DIR / "pre_data/input/pub_smi_des",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=BASE_DIR / "reviewer3_comment4",
    )
    parser.add_argument(
        "--candidate-sample-size",
        type=int,
        default=DEFAULT_CANDIDATE_SAMPLE,
    )
    parser.add_argument("--perplexity", type=float, default=40.0)
    return parser.parse_args()


def load_selected_descriptors(summary_path: Path) -> list[str]:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    names = summary["feature_selection"]["selected_descriptors"]
    if not names:
        raise ValueError("No selected descriptors were found in the summary file.")
    return list(names)


def build_training_reference(
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
        raise ValueError(
            "Training descriptor, target, and record tables have different lengths."
        )

    valid_indices = np.flatnonzero(pd.to_numeric(targets["logD"], errors="coerce").notna())
    development_indices, _ = train_test_split(
        valid_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    development = descriptors.iloc[development_indices].to_numpy(dtype=float)
    development_smiles = records.iloc[development_indices][smiles_column]
    means = np.nanmean(development, axis=0)
    missing = ~np.isfinite(development)
    if missing.any():
        development[missing] = np.take(means, np.where(missing)[1])

    canonical_ligands: list[str] = []
    for value in development_smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid training ligand SMILES: {value!r}")
        canonical_ligands.append(
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        )

    grouped = pd.DataFrame(development, columns=selected)
    grouped.insert(0, "canonical_ligand", canonical_ligands)
    # Use one robust descriptor position per canonical ligand. This prevents
    # repeated experimental records and small 3D-conformer differences from
    # producing hundreds of duplicate training points.
    reference = (
        grouped.groupby("canonical_ligand", sort=True)
        .median(numeric_only=True)
        .to_numpy(dtype=float)
    )
    return reference, means


def descriptor_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("desc_part_*.csv"))
    if not files:
        files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No candidate descriptor CSV files found in {directory}")
    return files


def sample_candidates(
    files: list[Path],
    analysis_dir: Path,
    selected: list[str],
    imputation_means: np.ndarray,
    requested_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    paired_files: list[tuple[Path, Path, int]] = []
    for descriptor_file in files:
        tag = descriptor_file.stem.rsplit("_", 1)[-1]
        similarity_file = analysis_dir / f"candidate_similarity_{tag}.csv"
        if similarity_file.exists():
            n_rows = len(pd.read_csv(similarity_file, usecols=["inside_descriptor_domain"]))
            paired_files.append((descriptor_file, similarity_file, n_rows))
    if not paired_files:
        raise FileNotFoundError(
            "No matching descriptor/similarity file pairs were found. Run the "
            "descriptor-similarity script first."
        )

    row_counts = np.asarray([item[2] for item in paired_files], dtype=int)
    total_rows = int(row_counts.sum())
    raw_allocations = requested_size * row_counts / total_rows
    allocations = np.floor(raw_allocations).astype(int)
    remainder = requested_size - int(allocations.sum())
    if remainder > 0:
        largest_remainders = np.argsort(-(raw_allocations - allocations))[:remainder]
        allocations[largest_remainders] += 1

    sampled_values: list[np.ndarray] = []
    for (descriptor_file, similarity_file, valid_rows), sample_n in zip(
        paired_files, allocations
    ):
        frame = pd.read_csv(descriptor_file, usecols=lambda name: name in selected)
        missing_columns = [name for name in selected if name not in frame.columns]
        if missing_columns:
            raise KeyError(f"{descriptor_file} is missing: {missing_columns}")

        if valid_rows > len(frame):
            raise ValueError(
                f"Similarity rows exceed descriptor rows: {similarity_file}"
            )
        # The last processed descriptor part may have been truncated when the
        # cumulative screening count reached exactly 1,178,295.
        frame = frame.iloc[:valid_rows].reset_index(drop=True)
        sample_n = min(int(sample_n), len(frame))
        positions = rng.choice(len(frame), size=sample_n, replace=False)
        values = frame.iloc[positions][selected].to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(imputation_means, np.where(missing)[1])
        sampled_values.append(values)

    candidates = np.vstack(sampled_values)
    if len(candidates) != requested_size:
        raise ValueError(
            f"Requested {requested_size:,} candidates but sampled {len(candidates):,}."
        )
    return candidates


def similarity_space_transform(
    training: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the same training-reference z-standardization as the quantitative
    nearest-neighbour analysis. For t-SNE visualization only, cap standardized
    coordinates at +/-8 so extreme domain-external values cannot numerically
    dominate the layout. The uncapped values remain used for all reported
    distances, similarities, and domain assignments.
    """
    scaler = StandardScaler().fit(training)
    return (
        np.clip(scaler.transform(training), -8.0, 8.0),
        np.clip(scaler.transform(candidates), -8.0, 8.0),
    )


def make_tsne(
    training: np.ndarray,
    candidates: np.ndarray,
    perplexity: float,
    output_dir: Path,
) -> None:
    training_scaled, candidate_scaled = similarity_space_transform(training, candidates)
    combined = np.vstack([training_scaled, candidate_scaled])
    labels = np.concatenate(
        [
            np.full(len(training_scaled), "Development-set ligand", dtype=object),
            np.full(len(candidate_scaled), "PubChem candidate", dtype=object),
        ]
    )

    effective_perplexity = min(perplexity, max(5.0, (len(combined) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="random",
        max_iter=1500,
        random_state=RANDOM_STATE,
        method="barnes_hut",
        angle=0.5,
        verbose=1,
    ).fit_transform(combined)

    n_training = len(training_scaled)
    training_xy = embedding[:n_training]
    candidate_xy = embedding[n_training:]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8}
    )
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    ax.scatter(
        candidate_xy[:, 0],
        candidate_xy[:, 1],
        s=6,
        color="#8FB8D8",
        alpha=0.28,
        linewidths=0,
        label="PubChem candidates (random sample)",
        rasterized=True,
    )
    ax.scatter(
        training_xy[:, 0],
        training_xy[:, 1],
        s=54,
        marker="*",
        color="#D6453D",
        edgecolors="white",
        linewidths=0.45,
        label="Development-set ligands",
        zorder=4,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Selected-descriptor chemical space", fontweight="bold")
    ax.legend(
        frameon=False,
        markerscale=1.3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(
        output_dir / "Figure_S_selected_descriptor_tSNE.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Figure_S_selected_descriptor_tSNE.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(
        {"tSNE1": embedding[:, 0], "tSNE2": embedding[:, 1], "group": labels}
    ).to_csv(output_dir / "selected_descriptor_tSNE_coordinates.csv", index=False)


def main() -> None:
    args = parse_args()
    summary_path = args.analysis_dir / "selected_descriptor_similarity_summary.json"
    selected = load_selected_descriptors(summary_path)
    training, imputation_means = build_training_reference(
        args.training_descriptors,
        args.target_file,
        args.training_records,
        args.ligand_smiles_column,
        selected,
    )
    candidates = sample_candidates(
        descriptor_files(args.candidate_descriptor_dir),
        args.analysis_dir,
        selected,
        imputation_means,
        args.candidate_sample_size,
    )
    make_tsne(
        training,
        candidates,
        args.perplexity,
        args.analysis_dir,
    )
    print(f"Selected descriptors: {len(selected)}")
    print(f"Unique development-set ligand profiles: {len(training)}")
    print(f"Candidate sample plotted: {len(candidates)}")
    print(f"Figure: {args.analysis_dir / 'Figure_S_selected_descriptor_tSNE.png'}")


if __name__ == "__main__":
    main()
