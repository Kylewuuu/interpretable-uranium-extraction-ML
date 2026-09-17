from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path("/home/wyh/data/examples/U_logD")
RANDOM_STATE = 42
N_SCREENED = 1_178_295
AD_QUANTILE = 0.95
SCRIPT_VERSION = "4.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a nested chemical-space visualization using the complete "
            "PubChem sample, applicability-domain-filtered screened candidates, "
            "and all unique ligands used by the final screening model."
        )
    )
    parser.add_argument(
        "--training-descriptors",
        type=Path,
        default=BASE_DIR / "data/input/ligand_des.xlsx",
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
        "--output-dir",
        type=Path,
        default=BASE_DIR / "reviewer3_comment4",
    )
    parser.add_argument("--n-screened", type=int, default=N_SCREENED)
    parser.add_argument("--ad-quantile", type=float, default=AD_QUANTILE)
    parser.add_argument(
        "--display-distance-multiplier",
        type=float,
        default=6.0,
        help=(
            "Maximum candidate-to-training distance shown, expressed as a "
            "multiple of the strict applicability-domain threshold. This changes "
            "only the visualization pool, not the quantitative AD definition."
        ),
    )
    parser.add_argument(
        "--core-fraction",
        type=float,
        default=0.35,
        help=(
            "Fraction of plotted candidates sampled from the strict AD core; "
            "the remainder is sampled from the neighbouring transition shell."
        ),
    )
    parser.add_argument(
        "--z-clip",
        type=float,
        default=5.0,
        help=(
            "Scale of the smooth tanh saturation applied after training-based "
            "z-standardization so that a single heavy-tailed descriptor cannot "
            "dominate the multivariate distance without collapsing distinct values."
        ),
    )
    parser.add_argument(
        "--max-domain-points",
        type=int,
        default=236,
        help=(
            "Number of applicability-domain screened candidates displayed. "
            "The default matches 0.02 percent of the complete screened pool."
        ),
    )
    parser.add_argument("--perplexity", type=float, default=50.0)
    parser.add_argument("--read-chunk-size", type=int, default=100_000)
    return parser.parse_args()


def load_selected(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = list(summary["feature_selection"]["selected_descriptors"])
    if not selected:
        raise ValueError("No selected descriptors were found.")
    return selected


def canonicalize(value: object) -> str:
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Invalid training-ligand SMILES: {value!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_training_ligands(
    descriptor_path: Path,
    record_path: Path,
    smiles_column: str,
    selected: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    descriptors = pd.read_excel(descriptor_path, usecols=selected)
    records = pd.read_excel(record_path, usecols=[smiles_column])
    if len(descriptors) != len(records):
        raise ValueError("Training descriptor and record tables differ in length.")

    descriptors = descriptors.apply(pd.to_numeric, errors="coerce")
    means = descriptors.mean(axis=0).to_numpy(dtype=float)
    values = descriptors.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(means, np.where(missing)[1])

    ligand_ids = records[smiles_column].map(canonicalize).reset_index(drop=True)
    grouped = pd.DataFrame(values, columns=selected)
    grouped.insert(0, "canonical_ligand", ligand_ids)
    grouped = grouped.groupby("canonical_ligand", sort=True).median(numeric_only=True)
    return (
        grouped.to_numpy(dtype=float),
        grouped.index.astype(str).tolist(),
        means,
    )


def list_descriptor_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("desc_part_*.csv"))
    if not files:
        files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No descriptor CSV files were found in {directory}")
    return files


def impute(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    if missing.any():
        result[missing] = np.take(means, np.where(missing)[1])
    return result


def transform_robust(
    values: np.ndarray,
    scaler: StandardScaler,
    saturation_scale: float,
) -> np.ndarray:
    standardized = scaler.transform(values)
    return saturation_scale * np.tanh(standardized / saturation_scale)


def build_domain_reference(
    reference: np.ndarray,
    clip_limit: float,
    quantile: float,
) -> tuple[StandardScaler, np.ndarray, float, NearestNeighbors]:
    scaler = StandardScaler().fit(reference)
    reference_z = transform_robust(reference, scaler, clip_limit)
    loo = NearestNeighbors(n_neighbors=2).fit(reference_z)
    loo_distances, _ = loo.kneighbors(reference_z)
    loo_rms = loo_distances[:, 1] / np.sqrt(reference_z.shape[1])
    threshold = float(np.quantile(loo_rms, quantile))
    nearest = NearestNeighbors(n_neighbors=1).fit(reference_z)
    return scaler, reference_z, threshold, nearest


def filter_screened_domain(
    directory: Path,
    selected: list[str],
    imputation_means: np.ndarray,
    scaler: StandardScaler,
    clip_limit: float,
    nearest: NearestNeighbors,
    strict_threshold: float,
    display_distance_multiplier: float,
    n_screened: int,
    chunk_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    domain_values: list[np.ndarray] = []
    domain_metadata: list[pd.DataFrame] = []
    processed = 0

    for path in list_descriptor_files(directory):
        row_offset = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda column: column in selected,
            chunksize=chunk_size,
        ):
            remaining = n_screened - processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            missing_columns = [name for name in selected if name not in chunk.columns]
            if missing_columns:
                raise KeyError(f"{path} is missing descriptors: {missing_columns}")

            raw = chunk[selected].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            raw = impute(raw, imputation_means)
            transformed = transform_robust(raw, scaler, clip_limit)
            distances, indices = nearest.kneighbors(transformed)
            rms = distances[:, 0] / np.sqrt(transformed.shape[1])
            display_threshold = strict_threshold * display_distance_multiplier
            inside = rms <= display_threshold

            if inside.any():
                domain_values.append(transformed[inside])
                local_rows = np.flatnonzero(inside) + row_offset
                domain_metadata.append(
                    pd.DataFrame(
                        {
                            "source_file": path.name,
                            "row_in_file": local_rows,
                            "global_screened_index": processed + np.flatnonzero(inside),
                            "nearest_training_ligand_index": indices[inside, 0],
                            "rms_distance": rms[inside],
                            "distance_over_strict_threshold": (
                                rms[inside] / strict_threshold
                            ),
                            "inside_strict_domain": rms[inside] <= strict_threshold,
                        }
                    )
                )

            processed += len(chunk)
            row_offset += len(chunk)
            if processed % 200_000 < len(chunk):
                inside_so_far = sum(len(values) for values in domain_values)
                print(
                    f"Screened rows processed={processed:,}; "
                    f"within display range={inside_so_far:,}",
                    flush=True,
                )
        if processed >= n_screened:
            break

    if processed != n_screened:
        raise ValueError(
            f"Expected {n_screened:,} screened rows but read {processed:,}."
        )
    if not domain_values:
        raise ValueError("No screened candidates fall within the requested display range.")
    return np.vstack(domain_values), pd.concat(domain_metadata, ignore_index=True)


def _grouped_sample_indices(
    metadata: pd.DataFrame,
    requested: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if requested <= 0 or len(metadata) == 0:
        return np.empty(0, dtype=int)
    if len(metadata) <= requested:
        return np.arange(len(metadata), dtype=int)

    groups = metadata.groupby("nearest_training_ligand_index").indices
    selected_indices: list[int] = []
    if requested >= len(groups):
        base_quota = max(1, requested // len(groups))
        for group_indices in groups.values():
            group_indices = np.asarray(group_indices, dtype=int)
            take = min(base_quota, len(group_indices))
            selected_indices.extend(
                rng.choice(group_indices, take, replace=False).tolist()
            )

    selected_set = set(selected_indices)
    remaining_slots = requested - len(selected_indices)
    if remaining_slots > 0:
        remaining = np.asarray(
            [index for index in range(len(metadata)) if index not in selected_set],
            dtype=int,
        )
        extra = rng.choice(
            remaining, min(remaining_slots, len(remaining)), replace=False
        )
        selected_indices.extend(extra.tolist())
    return np.asarray(selected_indices[:requested], dtype=int)


def stratified_core_shell_sample(
    values: np.ndarray,
    metadata: pd.DataFrame,
    max_points: int,
    core_fraction: float,
    display_distance_multiplier: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(values) <= max_points:
        result = metadata.copy()
        result["sampling_zone"] = np.where(
            result["inside_strict_domain"], "strict_core", "transition_shell"
        )
        return values, result

    rng = np.random.default_rng(RANDOM_STATE)
    core_global = np.flatnonzero(metadata["inside_strict_domain"].to_numpy(bool))
    shell_global = np.flatnonzero(~metadata["inside_strict_domain"].to_numpy(bool))

    desired_core = int(round(max_points * core_fraction))
    desired_shell = max_points - desired_core
    core_n = min(desired_core, len(core_global))
    shell_n = min(desired_shell, len(shell_global))

    unfilled = max_points - core_n - shell_n
    if unfilled > 0:
        core_extra = min(unfilled, len(core_global) - core_n)
        core_n += core_extra
        unfilled -= core_extra
    if unfilled > 0:
        shell_n += min(unfilled, len(shell_global) - shell_n)

    core_meta = metadata.iloc[core_global].reset_index(drop=True)
    selected_core = core_global[_grouped_sample_indices(core_meta, core_n, rng)]

    # Sample the transition region across three logarithmically spaced radial
    # bands. This prevents the plotted shell from being dominated by candidates
    # immediately outside the strict boundary and produces a continuous, honest
    # view of how the screened space broadens away from the training ligands.
    shell_meta = metadata.iloc[shell_global].reset_index(drop=True)
    shell_ratio = shell_meta["distance_over_strict_threshold"].to_numpy(float)
    radial_edges = np.geomspace(1.0, display_distance_multiplier, num=4)
    band_targets = np.full(3, shell_n // 3, dtype=int)
    band_targets[: shell_n % 3] += 1
    selected_shell_local: list[int] = []
    for band_index, (lower, upper) in enumerate(
        zip(radial_edges[:-1], radial_edges[1:])
    ):
        if band_index == 0:
            in_band = (shell_ratio > lower) & (shell_ratio <= upper)
        else:
            in_band = (shell_ratio > lower) & (shell_ratio <= upper)
        band_local = np.flatnonzero(in_band)
        if not len(band_local):
            continue
        band_meta = shell_meta.iloc[band_local].reset_index(drop=True)
        chosen_in_band = _grouped_sample_indices(
            band_meta, min(int(band_targets[band_index]), len(band_local)), rng
        )
        selected_shell_local.extend(band_local[chosen_in_band].tolist())

    # Refill any quota left by a sparse radial band using the remaining shell
    # candidates, without replacement.
    selected_shell_set = set(selected_shell_local)
    shell_unfilled = shell_n - len(selected_shell_local)
    if shell_unfilled > 0:
        remaining_shell = np.asarray(
            [
                index
                for index in range(len(shell_meta))
                if index not in selected_shell_set
            ],
            dtype=int,
        )
        if len(remaining_shell):
            extra = rng.choice(
                remaining_shell,
                min(shell_unfilled, len(remaining_shell)),
                replace=False,
            )
            selected_shell_local.extend(extra.tolist())

    selected_shell = shell_global[np.asarray(selected_shell_local[:shell_n], dtype=int)]
    selected = np.concatenate([selected_core, selected_shell])
    rng.shuffle(selected)

    result = metadata.iloc[selected].reset_index(drop=True).copy()
    result["sampling_zone"] = np.where(
        result["inside_strict_domain"], "strict_core", "transition_shell"
    )
    radial_edges = np.geomspace(1.0, display_distance_multiplier, num=4)
    ratios = result["distance_over_strict_threshold"].to_numpy(float)
    result["distance_band"] = np.select(
        [
            ratios <= 1.0,
            (ratios > 1.0) & (ratios <= radial_edges[1]),
            (ratios > radial_edges[1]) & (ratios <= radial_edges[2]),
            ratios > radial_edges[2],
        ],
        ["strict_core", "near_transition", "middle_transition", "outer_transition"],
        default="unclassified",
    )
    return values[selected], result


def make_nested_tsne(
    complete_z: np.ndarray,
    domain_z: np.ndarray,
    reference_z: np.ndarray,
    perplexity: float,
    output_dir: Path,
) -> None:
    # All three groups are embedded jointly. No coordinates are moved or mapped
    # after t-SNE. The blue points were selected quantitatively in the original
    # descriptor space before this visualization was calculated.
    combined = np.vstack([complete_z, domain_z, reference_z])
    n_components = min(15, combined.shape[1], len(combined) - 1)
    pca_values = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(
        combined
    )

    # Break exact numerical ties only. This perturbation is negligible relative
    # to the standardized descriptor scale and prevents zero t-SNE bandwidths.
    unique_rows = len(np.unique(np.round(pca_values, decimals=12), axis=0))
    duplicate_rows = len(pca_values) - unique_rows
    if duplicate_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        pca_values += rng.normal(0.0, 1.0e-7, size=pca_values.shape)

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
    ).fit_transform(pca_values)

    n_complete = len(complete_z)
    n_domain = len(domain_z)
    complete_xy = embedding[:n_complete]
    domain_xy = embedding[n_complete : n_complete + n_domain]
    reference_xy = embedding[n_complete + n_domain :]

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8}
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        complete_xy[:, 0],
        complete_xy[:, 1],
        s=6,
        color="#DCE6EE",
        alpha=0.30,
        linewidths=0,
        rasterized=True,
        label="PubChem collection",
        zorder=1,
    )
    ax.scatter(
        domain_xy[:, 0],
        domain_xy[:, 1],
        s=15,
        color="#5AA6C8",
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        label="Screened candidates",
        zorder=2,
    )
    ax.scatter(
        reference_xy[:, 0],
        reference_xy[:, 1],
        s=58,
        marker="*",
        color="#174A7E",
        edgecolors="white",
        linewidths=0.45,
        label="Training ligands",
        zorder=4,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "Model-relevant chemical space after screening",
        fontsize=15,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5)
    ax.margins(x=0.04, y=0.06)
    fig.tight_layout()
    fig.savefig(
        output_dir / "Figure_S_nested_pubchem_domain_tSNE_final.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Figure_S_nested_pubchem_domain_tSNE_final.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    all_coordinates = embedding
    labels = np.concatenate(
        [
            np.full(n_complete, "pubchem_collection", dtype=object),
            np.full(n_domain, "screened_candidates", dtype=object),
            np.full(len(reference_z), "training_ligands", dtype=object),
        ]
    )
    pd.DataFrame(
        {
            "tSNE1": all_coordinates[:, 0],
            "tSNE2": all_coordinates[:, 1],
            "group": labels,
        }
    ).to_csv(output_dir / "nested_pubchem_domain_tSNE_final_coordinates.csv", index=False)

    print(f"Duplicate rows tie-broken before t-SNE: {duplicate_rows:,}")


def main() -> None:
    args = parse_args()
    print(f"Nested-domain plotting script version: {SCRIPT_VERSION}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.ad_quantile < 1.0:
        raise ValueError("--ad-quantile must be between 0 and 1.")
    if args.z_clip <= 0:
        raise ValueError("--z-clip must be positive.")
    if args.display_distance_multiplier < 1.0:
        raise ValueError("--display-distance-multiplier must be at least 1.0.")
    if not 0.0 <= args.core_fraction <= 1.0:
        raise ValueError("--core-fraction must be between 0 and 1.")

    selected = load_selected(args.selected_summary)
    reference, ligand_ids, imputation_means = load_training_ligands(
        args.training_descriptors,
        args.training_records,
        args.ligand_smiles_column,
        selected,
    )
    scaler, reference_z, threshold, nearest = build_domain_reference(
        reference, args.z_clip, args.ad_quantile
    )

    display_z, display_metadata = filter_screened_domain(
        args.screened_descriptor_dir,
        selected,
        imputation_means,
        scaler,
        args.z_clip,
        nearest,
        threshold,
        args.display_distance_multiplier,
        args.n_screened,
        args.read_chunk_size,
    )
    display_metadata["nearest_training_ligand"] = display_metadata[
        "nearest_training_ligand_index"
    ].map(dict(enumerate(ligand_ids)))
    display_metadata.to_csv(
        args.output_dir / "screened_candidates_within_display_range.csv", index=False
    )

    strict_count = int(display_metadata["inside_strict_domain"].sum())
    domain_plot_z, domain_plot_metadata = stratified_core_shell_sample(
        display_z,
        display_metadata,
        args.max_domain_points,
        args.core_fraction,
        args.display_distance_multiplier,
    )
    domain_plot_metadata.to_csv(
        args.output_dir / "screened_candidates_plotted.csv", index=False
    )

    complete = pd.read_csv(args.complete_cache, usecols=selected)
    complete_raw = complete[selected].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    complete_raw = impute(complete_raw, imputation_means)
    complete_z = transform_robust(complete_raw, scaler, args.z_clip)

    make_nested_tsne(
        complete_z,
        domain_plot_z,
        reference_z,
        args.perplexity,
        args.output_dir,
    )

    summary = {
        "analysis": "sampled PubChem and model-applicability descriptor-domain t-SNE",
        "selected_descriptor_count": len(selected),
        "selected_descriptors": selected,
        "training_ligand_count": len(reference),
        "screened_input_count": args.n_screened,
        "screened_in_strict_domain_count": strict_count,
        "screened_in_strict_domain_fraction": strict_count / args.n_screened,
        "screened_within_display_range_count": len(display_z),
        "screened_within_display_range_fraction": len(display_z) / args.n_screened,
        "screened_points_plotted": len(domain_plot_z),
        "plotted_strict_core_count": int(
            domain_plot_metadata["inside_strict_domain"].sum()
        ),
        "plotted_transition_shell_count": int(
            (~domain_plot_metadata["inside_strict_domain"]).sum()
        ),
        "complete_pubchem_points_plotted": len(complete_z),
        "domain_boundary": (
            f"{args.ad_quantile:.0%} percentile of leave-one-out nearest-neighbour "
            "RMS distances among unique training ligands"
        ),
        "domain_distance_threshold": threshold,
        "display_distance_multiplier": args.display_distance_multiplier,
        "display_distance_threshold": threshold * args.display_distance_multiplier,
        "preprocessing": (
            "training-based z-standardization followed by smooth symmetric tanh "
            f"saturation with scale {args.z_clip:g}; the same transformed values "
            "are used for domain filtering and visualization"
        ),
        "visualization": (
            "PCA to at most 15 components followed by a joint t-SNE embedding "
            "of the PubChem sample, the stratified core-plus-transition sample "
            "of screened candidates, and all unique training ligands. Coordinates are not "
            "moved or interpolated after t-SNE. t-SNE is used only for qualitative "
            "visualization."
        ),
        "candidate_plot_sampling": (
            f"{args.core_fraction:.0%} of plotted candidates are requested from "
            "the strict applicability-domain core and the remainder from the "
            "transition region. The transition sample is distributed across three "
            "logarithmically spaced distance bands and grouped by nearest training "
            "ligand in the original transformed descriptor space."
        ),
        "transition_radial_edges_over_strict_threshold": np.geomspace(
            1.0, args.display_distance_multiplier, num=4
        ).tolist(),
        "random_seed": RANDOM_STATE,
    }
    with open(
        args.output_dir / "nested_pubchem_domain_tSNE_final_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Selected descriptors: {len(selected)}")
    print(f"Unique training ligands: {len(reference)}")
    print(f"Strict-domain screened candidates: {strict_count:,}")
    print(f"Strict-domain fraction: {strict_count / args.n_screened:.6%}")
    print(f"Candidates within display range: {len(display_z):,}")
    print(
        "Plotted composition: "
        f"core={int(domain_plot_metadata['inside_strict_domain'].sum()):,}; "
        f"transition={int((~domain_plot_metadata['inside_strict_domain']).sum()):,}"
    )
    print(f"Candidates plotted: {len(domain_plot_z):,}")
    print(
        "Figure: "
        f"{args.output_dir / 'Figure_S_nested_pubchem_domain_tSNE_final.png'}"
    )


if __name__ == "__main__":
    main()
