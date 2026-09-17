from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RDLogger.DisableLog("rdApp.*")


DEFAULT_BASE_DIR = Path("/home/kylewu/hotpot/examples/U_logD")
N_SCREENED = 1_178_295
RANDOM_STATE = 42
TEST_SIZE = 0.20
IMPORTANCE_THRESHOLD = 1e-4
CORRELATION_THRESHOLD = 0.90
AD_QUANTILE = 0.95
SCATTER_SAMPLE_SIZE = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Candidate-to-training similarity in the nonredundant, model-relevant "
            "ligand-descriptor space retained by feature engineering."
        )
    )
    parser.add_argument(
        "--training-descriptors",
        type=Path,
        default=DEFAULT_BASE_DIR / "data/input/ligand_des.xlsx",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=DEFAULT_BASE_DIR / "data/input/y.xlsx",
    )
    parser.add_argument(
        "--training-records",
        type=Path,
        default=DEFAULT_BASE_DIR / "data/shuffled_total.xlsx",
        help="Record-level table containing the ligand SMILES column.",
    )
    parser.add_argument("--ligand-smiles-column", default="lig_SMI")
    parser.add_argument(
        "--candidate-descriptor-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "pre_data/input/pub_smi_des",
    )
    parser.add_argument(
        "--candidate-id-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "pre_data/input/pub_smi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "reviewer3_comment4",
    )
    parser.add_argument("--n-screened", type=int, default=N_SCREENED)
    parser.add_argument("--scatter-sample-size", type=int, default=SCATTER_SAMPLE_SIZE)
    return parser.parse_args()


def part_tag(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def canonicalize_smiles(value: object) -> str:
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Invalid training ligand SMILES: {value!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_training_data(
    descriptor_path: Path,
    target_path: Path,
    record_path: Path,
    smiles_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    descriptors = pd.read_excel(descriptor_path)
    target_frame = pd.read_excel(target_path)
    records = pd.read_excel(record_path, usecols=[smiles_column])
    if "logD" not in target_frame.columns:
        raise KeyError("The target file must contain a 'logD' column.")
    if not (len(descriptors) == len(target_frame) == len(records)):
        raise ValueError("Descriptor, target, and record tables have different row counts.")

    descriptors = descriptors.select_dtypes(include=[np.number]).copy()
    target = pd.to_numeric(target_frame["logD"], errors="coerce")
    valid = target.notna()
    ligand_ids = records.loc[valid, smiles_column].map(canonicalize_smiles)
    return (
        descriptors.loc[valid].reset_index(drop=True),
        target.loc[valid].reset_index(drop=True),
        ligand_ids.reset_index(drop=True),
    )


def select_model_relevant_descriptors(
    descriptors: pd.DataFrame, target: pd.Series
) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the descriptor-block feature-engineering stages used before final
    recursive subset selection:
      1. split experimental records with random_state=42;
      2. mean imputation and removal of zero-variance descriptors;
      3. XGBoost normalized importance >= 1e-4;
      4. absolute-Pearson clustering at |r| >= 0.90, retaining the most
         important descriptor from each cluster.

    The final two-descriptor RFE subset is deliberately not used here: the aim is
    to characterize the broader nonredundant descriptor domain learned during
    feature engineering, rather than draw a two-axis applicability boundary.
    """
    indices = np.arange(len(descriptors))
    development_indices, _ = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    development = descriptors.iloc[development_indices].copy()
    y_development = target.iloc[development_indices].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    development_imputed = imputer.fit_transform(development)
    variances = np.nanvar(development_imputed, axis=0)
    nonconstant_mask = np.isfinite(variances) & (variances > 0)
    feature_names = development.columns[nonconstant_mask].tolist()
    x_development = development_imputed[:, nonconstant_mask]
    imputer_statistics = imputer.statistics_[nonconstant_mask]

    preliminary_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.6,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preliminary_model.fit(x_development, y_development)
    importances = np.asarray(preliminary_model.feature_importances_, dtype=float)
    importance_sum = float(importances.sum())
    if importance_sum <= 0:
        raise ValueError("The preliminary model returned no positive feature importance.")
    normalized_importance = importances / importance_sum
    important_mask = normalized_importance >= IMPORTANCE_THRESHOLD
    important_names = np.asarray(feature_names, dtype=object)[important_mask].tolist()
    important_values = normalized_importance[important_mask]
    important_matrix = x_development[:, important_mask]
    important_imputer_statistics = imputer_statistics[important_mask]
    if not important_names:
        raise ValueError("No descriptors passed the normalized-importance threshold.")

    if len(important_names) == 1:
        cluster_labels = np.ones(1, dtype=int)
    else:
        correlation = np.corrcoef(important_matrix, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlation, 1.0)
        distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
        hierarchy = linkage(squareform(distance, checks=False), method="average")
        cluster_labels = fcluster(
            hierarchy,
            t=1.0 - CORRELATION_THRESHOLD,
            criterion="distance",
        )

    selected_positions: list[int] = []
    for cluster in sorted(np.unique(cluster_labels)):
        members = np.flatnonzero(cluster_labels == cluster)
        representative = members[np.argmax(important_values[members])]
        selected_positions.append(int(representative))
    selected_positions.sort(key=lambda position: important_values[position], reverse=True)
    selected_names = [important_names[position] for position in selected_positions]

    audit = pd.DataFrame(
        {
            "descriptor": important_names,
            "normalized_importance": important_values,
            "correlation_cluster": cluster_labels,
            "retained_for_similarity": [
                position in set(selected_positions) for position in range(len(important_names))
            ],
        }
    ).sort_values(
        ["retained_for_similarity", "normalized_importance"],
        ascending=[False, False],
    )

    selected_matrix = important_matrix[:, selected_positions]
    selected_imputer_statistics = important_imputer_statistics[selected_positions]
    return (
        selected_names,
        audit,
        selected_matrix,
        selected_imputer_statistics,
        development_indices,
    )


def build_ligand_training_reference(
    selected_matrix: np.ndarray,
    development_ligand_ids: pd.Series,
) -> tuple[np.ndarray, StandardScaler]:
    frame = pd.DataFrame(selected_matrix)
    frame.insert(0, "canonical_ligand", development_ligand_ids.to_numpy())
    # A ligand may occur under many experimental records. Median descriptor
    # values provide one robust representative position for each canonical
    # ligand and prevent repeated records or conformer noise from overweighting it.
    reference = (
        frame.groupby("canonical_ligand", sort=True)
        .median(numeric_only=True)
        .to_numpy(dtype=float)
    )
    if len(reference) < 2:
        raise ValueError("At least two unique canonical training ligands are required.")
    scaler = StandardScaler().fit(reference)
    return reference, scaler


def training_domain_threshold(
    reference: np.ndarray, scaler: StandardScaler
) -> tuple[float, float, np.ndarray]:
    reference_z = scaler.transform(reference)
    neighbours = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(reference_z)
    distances, _ = neighbours.kneighbors(reference_z)
    # Root-mean-square standardized distance makes values comparable if the
    # selected descriptor count changes after rerunning feature engineering.
    loo_rms_distances = distances[:, 1] / np.sqrt(reference_z.shape[1])
    threshold = float(np.quantile(loo_rms_distances, AD_QUANTILE))
    similarity_threshold = 1.0 / (1.0 + threshold)
    return threshold, similarity_threshold, loo_rms_distances


def find_identity_file(identity_dir: Path, descriptor_file: Path) -> Path | None:
    tag = part_tag(descriptor_file)
    candidates = [
        identity_dir / f"CID_SMILES_part_{tag}.csv",
        identity_dir / f"pub_smi_part_{tag}.csv",
        identity_dir / descriptor_file.name,
    ]
    return next((path for path in candidates if path.exists()), None)


def analyze_candidates(
    args: argparse.Namespace,
    selected_names: list[str],
    imputer_statistics: np.ndarray,
    reference: np.ndarray,
    scaler: StandardScaler,
    distance_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, int]:
    descriptor_files = sorted(args.candidate_descriptor_dir.glob("*.csv"))
    if not descriptor_files:
        raise FileNotFoundError(
            f"No candidate descriptor CSV files found in {args.candidate_descriptor_dir}"
        )

    reference_z = scaler.transform(reference)
    neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(reference_z)
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    output_parts: list[pd.DataFrame] = []
    n_processed = 0
    n_inside = 0

    for descriptor_file in descriptor_files:
        if n_processed >= args.n_screened:
            break
        frame = pd.read_csv(descriptor_file, usecols=lambda name: name in selected_names)
        missing = [name for name in selected_names if name not in frame.columns]
        if missing:
            raise KeyError(f"{descriptor_file} is missing selected descriptors: {missing}")
        remaining = args.n_screened - n_processed
        if len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()

        values = frame[selected_names].to_numpy(dtype=float)
        missing_mask = ~np.isfinite(values)
        if missing_mask.any():
            values[missing_mask] = np.take(imputer_statistics, np.where(missing_mask)[1])
        values_z = scaler.transform(values)
        euclidean, nearest_indices = neighbours.kneighbors(values_z)
        rms_distance = euclidean[:, 0] / np.sqrt(len(selected_names))
        similarity = 1.0 / (1.0 + rms_distance)
        inside = rms_distance <= distance_threshold
        n_inside += int(inside.sum())
        all_distances.append(rms_distance.astype(np.float32))

        result = pd.DataFrame(
            {
                "nearest_training_profile": nearest_indices[:, 0],
                "descriptor_rms_distance": rms_distance,
                "descriptor_similarity": similarity,
                "inside_descriptor_domain": inside,
            }
        )
        identity_file = find_identity_file(args.candidate_id_dir, descriptor_file)
        if identity_file is not None:
            identities = pd.read_csv(identity_file, usecols=lambda name: name in {"CID", "SMILES"})
            identities = identities.iloc[: len(result)].reset_index(drop=True)
            if len(identities) != len(result):
                raise ValueError(f"Row-count mismatch between {descriptor_file} and {identity_file}")
            result = pd.concat([identities, result], axis=1)
        result.to_csv(
            args.output_dir / f"candidate_similarity_{part_tag(descriptor_file)}.csv",
            index=False,
        )
        output_parts.append(result)

        target_sample = max(
            1,
            int(np.ceil(args.scatter_sample_size * len(values_z) / args.n_screened)),
        )
        sample_size = min(target_sample, len(values_z))
        reservoir.append(values_z[rng.choice(len(values_z), sample_size, replace=False)])
        n_processed += len(frame)

    if n_processed != args.n_screened:
        raise ValueError(
            f"Expected {args.n_screened:,} candidate rows, but processed {n_processed:,}."
        )

    candidate_sample_z = np.vstack(reservoir)
    if len(candidate_sample_z) > args.scatter_sample_size:
        keep = rng.choice(len(candidate_sample_z), args.scatter_sample_size, replace=False)
        candidate_sample_z = candidate_sample_z[keep]
    distances = np.concatenate(all_distances)
    combined = pd.concat(output_parts, ignore_index=True)
    return combined, distances, candidate_sample_z, n_inside, n_processed


def make_figure(
    output_dir: Path,
    reference: np.ndarray,
    scaler: StandardScaler,
    candidate_sample_z: np.ndarray,
    candidate_distances: np.ndarray,
    similarity_threshold: float,
) -> None:
    reference_z = scaler.transform(reference)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(reference_z)
    reference_xy = pca.transform(reference_z)
    candidate_xy = pca.transform(candidate_sample_z)
    similarities = 1.0 / (1.0 + candidate_distances)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    density = axes[0].hexbin(
        candidate_xy[:, 0],
        candidate_xy[:, 1],
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0,
    )
    axes[0].scatter(
        reference_xy[:, 0],
        reference_xy[:, 1],
        marker="x",
        s=28,
        linewidth=1.0,
        color="#C43C39",
        label="Development-set ligands",
        zorder=3,
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("a  Selected-descriptor space", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    colorbar = fig.colorbar(density, ax=axes[0], pad=0.02)
    colorbar.set_label("Candidate density (log count)")

    axes[1].hist(similarities, bins=80, density=True, color="#75A9C9", edgecolor="none")
    axes[1].axvline(
        similarity_threshold,
        color="#C43C39",
        linestyle="--",
        linewidth=1.2,
        label="Training-derived 95% boundary",
    )
    axes[1].set_xlabel("Nearest-training descriptor similarity")
    axes[1].set_ylabel("Density")
    axes[1].set_title("b  Candidate similarity", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "Figure_S_selected_descriptor_similarity.png", dpi=600)
    fig.savefig(output_dir / "Figure_S_selected_descriptor_similarity.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    descriptors, target, ligand_ids = load_training_data(
        args.training_descriptors,
        args.target_file,
        args.training_records,
        args.ligand_smiles_column,
    )
    selected_names, audit, selected_development, imputer_statistics, development_indices = (
        select_model_relevant_descriptors(descriptors, target)
    )
    audit.to_csv(args.output_dir / "selected_descriptor_feature_engineering.csv", index=False)

    reference, scaler = build_ligand_training_reference(
        selected_development,
        ligand_ids.iloc[development_indices].reset_index(drop=True),
    )
    distance_threshold, similarity_threshold, loo_distances = training_domain_threshold(
        reference, scaler
    )
    results, candidate_distances, candidate_sample_z, n_inside, n_processed = analyze_candidates(
        args,
        selected_names,
        imputer_statistics,
        reference,
        scaler,
        distance_threshold,
    )
    results.to_csv(args.output_dir / "candidate_selected_descriptor_similarity.csv", index=False)
    summary = {
        "analysis": "nearest-development-ligand similarity in selected descriptor space",
        "record_split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "feature_selection": {
            "normalized_xgboost_importance_threshold": IMPORTANCE_THRESHOLD,
            "absolute_pearson_correlation_threshold": CORRELATION_THRESHOLD,
            "selected_descriptor_count": len(selected_names),
            "selected_descriptors": selected_names,
        },
        "distance": "nearest-neighbour RMS Euclidean distance after z-standardization",
        "similarity": "1 / (1 + distance)",
        "domain_boundary": {
            "definition": "95th percentile of leave-one-out nearest-neighbour distances among unique development-set ligand profiles",
            "distance_threshold": distance_threshold,
            "similarity_threshold": similarity_threshold,
            "training_loo_distance_median": float(np.median(loo_distances)),
        },
        "candidate_count": n_processed,
        "unique_development_ligand_count": len(reference),
        "candidate_inside_count": n_inside,
        "candidate_inside_fraction": n_inside / n_processed,
        "candidate_similarity_median": float(np.median(1.0 / (1.0 + candidate_distances))),
    }
    with open(args.output_dir / "selected_descriptor_similarity_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
