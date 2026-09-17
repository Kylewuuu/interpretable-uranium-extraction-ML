#!/usr/bin/env python3
"""Leakage-controlled validation for the uranium-extraction logD model.

This script was prepared for Reviewer 3, Comment 1. It reconstructs the three
feature blocks used in the manuscript and performs three complementary tests:

1. record-level holdout validation
   Individual records are randomly assigned to an 80/20 split with seed 42,
   reproducing the split unit used for the manuscript while keeping all
   data-adaptive processing inside the development subset.
2. condition-grouped holdout validation
   Records sharing the same ligand, solvent, and experimental-condition vector
   are kept in the same partition. This prevents identical model inputs from
   appearing in both training and test sets.
3. ligand-grouped holdout validation
   Ligand and solvent SMILES are canonicalized with RDKit before grouping. All
   records and alternate SMILES aliases for a canonical ligand are kept
   together, so every test ligand is unseen during feature selection,
   hyperparameter optimization, and model fitting.

For both analyses, the holdout set is created before any data-adaptive feature
selection. Feature-importance screening, correlation clustering, recursive
feature selection, and Optuna tuning are fitted using the development set only.

Expected input files in --input-dir:
    ligand_des.xlsx, sol_des.xlsx, ligand_mor.xlsx, sol_mor.xlsx,
    sol_envs.xlsx, y.xlsx

The metadata file can be either shuffled_total_metadata.zip (containing a CSV),
a CSV, or an XLSX file. It must contain the aligned columns record_index,
lig_SMI, Sol_SMI, c_lig, c_HNO3, c_Metal_mM, D, and logD.

Run in the original Python environment used for the manuscript, for example:

python reviewer3_q1_validation.py \
  --input-dir /home/kylewu/hotpot/examples/U_logD/data/input \
  --metadata /home/kylewu/hotpot/examples/U_logD/data/shuffled_total_metadata.zip \
  --output-dir /home/kylewu/hotpot/examples/U_logD/reviewer3_q1_validation \
  --analyses both --n-trials 100

Use --validate-only first to audit files without fitting models.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_val_score,
    train_test_split,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
IMPORTANCE_THRESHOLD = 1.0e-4
CORRELATION_THRESHOLD = 0.95
RECURSIVE_CANDIDATE_CAP = 60
RFA_TOLERANCE = 1.0e-4
RFA_VS_RFE_SCORE_TOLERANCE = 0.01

DESCRIPTOR_3D_NAMES = [
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
]

REQUIRED_INPUTS = [
    "ligand_des.xlsx",
    "sol_des.xlsx",
    "ligand_mor.xlsx",
    "sol_mor.xlsx",
    "sol_envs.xlsx",
    "y.xlsx",
]

REQUIRED_METADATA_COLUMNS = [
    "record_index",
    "lig_SMI",
    "Sol_SMI",
    "c_lig",
    "c_HNO3",
    "c_Metal_mM",
    "D",
    "logD",
]


@dataclass
class SplitSummary:
    analysis: str
    n_total: int
    n_train: int
    n_test: int
    test_fraction: float
    n_train_outer_groups: int
    n_test_outer_groups: int
    outer_group_overlap: int
    n_train_ligands: int
    n_test_ligands: int
    ligand_overlap: int
    train_logd_mean: float
    test_logd_mean: float
    train_logd_std: float
    test_logd_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled validation for Reviewer 3, Comment 1"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analyses",
        choices=["record", "condition", "ligand", "both", "all"],
        default="both",
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--skip-descriptor-repair",
        action="store_true",
        help="Leave missing descriptor rows as NaN instead of recovering 2D descriptors from SMILES.",
    )
    return parser.parse_args()


def require_model_packages():
    try:
        import optuna  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Model fitting requires xgboost and optuna. Run this script in the "
            "original manuscript environment (XGBoost 2.1.1; Optuna 3.1.1)."
        ) from exc


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(
                    f"Expected exactly one CSV in {path}; found {csv_members}"
                )
            with archive.open(csv_members[0]) as stream:
                metadata = pd.read_csv(stream)
    elif suffix == ".csv":
        metadata = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        metadata = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported metadata format: {path}")

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    return metadata


def canonicalize_smiles(
    smiles: pd.Series,
    label: str,
) -> pd.Series:
    """Return RDKit canonical isomeric SMILES for grouping identities."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required to canonicalize ligand and solvent identities "
            "before grouped validation."
        ) from exc

    canonical = []
    failures = []
    for row_idx, raw_smiles in smiles.items():
        molecule = Chem.MolFromSmiles(str(raw_smiles))
        if molecule is None:
            failures.append({"row": int(row_idx), "smiles": str(raw_smiles)})
            canonical.append(None)
        else:
            canonical.append(
                Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=True,
                )
            )
    if failures:
        raise ValueError(f"Failed to canonicalize {label} SMILES: {failures}")
    return pd.Series(canonical, index=smiles.index, name=f"{label}_canonical")


def read_inputs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    missing_files = [name for name in REQUIRED_INPUTS if not (input_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input files in {input_dir}: {missing_files}")

    return {
        "ligand_des": pd.read_excel(input_dir / "ligand_des.xlsx"),
        "sol_des": pd.read_excel(input_dir / "sol_des.xlsx"),
        "ligand_mor": pd.read_excel(input_dir / "ligand_mor.xlsx"),
        "sol_mor": pd.read_excel(input_dir / "sol_mor.xlsx"),
        "env": pd.read_excel(input_dir / "sol_envs.xlsx"),
        "target": pd.read_excel(input_dir / "y.xlsx"),
    }


def allclose_with_nan(left: pd.Series, right: pd.Series) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=True,
        )
    )


def audit_alignment(
    tables: Dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> Dict[str, object]:
    lengths = {name: len(frame) for name, frame in tables.items()}
    lengths["metadata"] = len(metadata)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input row counts are inconsistent: {lengths}")

    if not np.array_equal(
        metadata["record_index"].to_numpy(), np.arange(len(metadata))
    ):
        raise ValueError("record_index is not sequential; row alignment cannot be guaranteed")

    checks = {
        "c_lig": allclose_with_nan(metadata["c_lig"], tables["env"]["c_lig"]),
        "c_HNO3": allclose_with_nan(metadata["c_HNO3"], tables["env"]["c_HNO3"]),
        "c_Metal_mM": allclose_with_nan(
            metadata["c_Metal_mM"], tables["env"]["c_Metal_mM"]
        ),
        "D": allclose_with_nan(metadata["D"], tables["target"]["D"]),
        "logD": allclose_with_nan(metadata["logD"], tables["target"]["logD"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Metadata/input alignment failed for: {failed}")

    ligand_key = "lig_canonical" if "lig_canonical" in metadata else "lig_SMI"
    solvent_key = "sol_canonical" if "sol_canonical" in metadata else "Sol_SMI"
    input_key_columns = [
        ligand_key,
        solvent_key,
        "c_lig",
        "c_HNO3",
        "c_Metal_mM",
    ]
    complete_key_columns = input_key_columns + ["D", "logD"]

    ligand_aliases = {}
    if "lig_canonical" in metadata:
        for canonical, group in metadata.groupby("lig_canonical", sort=True):
            aliases = sorted(group["lig_SMI"].astype(str).unique().tolist())
            if len(aliases) > 1:
                ligand_aliases[str(canonical)] = aliases

    solvent_aliases = {}
    if "sol_canonical" in metadata:
        for canonical, group in metadata.groupby("sol_canonical", sort=True):
            aliases = sorted(group["Sol_SMI"].astype(str).unique().tolist())
            if len(aliases) > 1:
                solvent_aliases[str(canonical)] = aliases

    return {
        "row_counts": lengths,
        "alignment_checks": checks,
        "n_raw_ligand_smiles": int(metadata["lig_SMI"].nunique()),
        "n_canonical_ligands": int(
            metadata[ligand_key].nunique()
        ),
        "canonical_ligand_aliases": ligand_aliases,
        "n_raw_solvent_smiles": int(metadata["Sol_SMI"].nunique()),
        "n_canonical_solvents": int(
            metadata[solvent_key].nunique()
        ),
        "canonical_solvent_aliases": solvent_aliases,
        "n_missing_target": int(metadata["logD"].isna().sum()),
        "n_rows_with_any_missing_condition": int(
            metadata[["c_lig", "c_HNO3", "c_Metal_mM"]].isna().any(axis=1).sum()
        ),
        "n_exact_duplicate_rows_including_target": int(
            metadata.duplicated(complete_key_columns).sum()
        ),
        "n_unique_input_groups": int(
            metadata[input_key_columns].drop_duplicates().shape[0]
        ),
        "all_missing_ligand_descriptor_rows": np.where(
            tables["ligand_des"].isna().all(axis=1)
        )[0].astype(int).tolist(),
    }


def repair_missing_2d_descriptors(
    descriptor_df: pd.DataFrame,
    smiles: pd.Series,
    prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Recover 2D descriptors for rows lost because 3D embedding failed.

    The original generator attempted embedding before CalcMolDescriptors, so a
    3D failure produced an entirely empty row. Here only the 2D fields are
    recovered. The ten 3D fields remain missing and can be handled by XGBoost's
    native missing-value treatment.
    """
    missing_rows = np.where(descriptor_df.isna().all(axis=1))[0]
    report = {
        "requested_rows": missing_rows.astype(int).tolist(),
        "repaired_rows": [],
        "failed_rows": [],
        "note": "Only 2D descriptors are recovered; 3D fields remain NaN.",
    }
    if len(missing_rows) == 0:
        return descriptor_df, report

    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import CalcMolDescriptors
    except ImportError:
        warnings.warn(
            "RDKit is unavailable; missing descriptor rows were left as NaN."
        )
        report["failed_rows"] = missing_rows.astype(int).tolist()
        return descriptor_df, report

    repaired = descriptor_df.copy()
    three_d_columns = {f"{prefix}{name}" for name in DESCRIPTOR_3D_NAMES}

    for row_idx in missing_rows:
        try:
            mol = Chem.MolFromSmiles(str(smiles.iloc[row_idx]))
            if mol is None:
                raise ValueError("RDKit could not parse SMILES")
            mol = Chem.AddHs(mol)
            calculated = CalcMolDescriptors(mol)
            for raw_name, value in calculated.items():
                column = f"{prefix}{raw_name}"
                if column in repaired.columns and column not in three_d_columns:
                    repaired.at[row_idx, column] = value
            report["repaired_rows"].append(int(row_idx))
        except Exception as exc:  # retain a record of every unresolved row
            report["failed_rows"].append(
                {"row": int(row_idx), "error": f"{type(exc).__name__}: {exc}"}
            )
    return repaired, report


def sanitize_xgboost_values(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Replace values that XGBoost cannot represent with missing values.

    XGBoost's histogram data matrix uses float32-compatible feature values.
    A small number of RDKit descriptors (notably information-content terms)
    can return +/-inf or finite values beyond the float32 range for unusual
    structures. XGBoost handles NaN natively, so these exceptional cells are
    converted to NaN without estimating values from the holdout set.
    """
    numeric = frame.astype(float).copy()
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    invalid = np.isinf(values) | (
        np.abs(values) > np.finfo(np.float32).max
    )
    affected_columns = [
        str(numeric.columns[index])
        for index in np.where(invalid.any(axis=0))[0]
    ]
    report = {
        "n_replaced_cells": int(invalid.sum()),
        "n_affected_rows": int(invalid.any(axis=1).sum()),
        "n_affected_columns": int(invalid.any(axis=0).sum()),
        "affected_columns": affected_columns,
        "replacement": "NaN (handled natively by XGBoost)",
        "float32_max": float(np.finfo(np.float32).max),
    }
    if invalid.any():
        numeric = numeric.mask(invalid, np.nan)
    return numeric, report


def make_condition_groups(metadata: pd.DataFrame) -> np.ndarray:
    ligand_key = "lig_canonical" if "lig_canonical" in metadata else "lig_SMI"
    solvent_key = "sol_canonical" if "sol_canonical" in metadata else "Sol_SMI"
    columns = [ligand_key, solvent_key, "c_lig", "c_HNO3", "c_Metal_mM"]
    key = metadata[columns].copy()
    key = key.astype(object).where(key.notna(), "__MISSING__")
    return key.astype(str).agg("||".join, axis=1).to_numpy()


def balanced_ligand_holdout(
    ligand_groups: np.ndarray,
    target_fraction: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    search_iterations: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose approximately 20% of records while keeping ~20% of ligands.

    Selection uses group sizes only, never the target values.
    """
    labels, counts = np.unique(ligand_groups, return_counts=True)
    n_test_groups = max(1, int(round(target_fraction * len(labels))))
    rng = np.random.default_rng(random_state)
    target_n = target_fraction * len(ligand_groups)

    best_indices = None
    best_distance = np.inf
    for _ in range(search_iterations):
        candidate = rng.choice(len(labels), size=n_test_groups, replace=False)
        distance = abs(counts[candidate].sum() - target_n)
        if distance < best_distance:
            best_distance = distance
            best_indices = candidate.copy()
            if distance == 0:
                break

    test_labels = set(labels[best_indices])
    test_mask = np.array([label in test_labels for label in ligand_groups])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def make_outer_split(
    analysis: str,
    metadata: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if analysis == "record":
        outer_groups = np.arange(len(metadata))
        train_idx, test_idx = train_test_split(
            outer_groups,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
    elif analysis == "condition":
        outer_groups = make_condition_groups(metadata)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(
            splitter.split(metadata, metadata["logD"], groups=outer_groups)
        )
    elif analysis == "ligand":
        ligand_key = (
            "lig_canonical" if "lig_canonical" in metadata else "lig_SMI"
        )
        outer_groups = metadata[ligand_key].astype(str).to_numpy()
        train_idx, test_idx = balanced_ligand_holdout(outer_groups)
    else:
        raise ValueError(analysis)
    return train_idx, test_idx, outer_groups


def make_inner_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
    group_aware: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not group_aware:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return list(splitter.split(X, y))
    unique_groups = np.unique(groups)
    if len(unique_groups) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups=groups))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, y))


def build_selection_estimator(random_state: int = RANDOM_STATE):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )


def correlation_representatives(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    importance: Dict[str, float],
    threshold: float = CORRELATION_THRESHOLD,
) -> List[str]:
    names = list(feature_names)
    if len(names) <= 1:
        return names

    numeric = X[names].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.mean()).fillna(0.0)
    corr = np.corrcoef(numeric.to_numpy(), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    distance = 1.0 - np.abs(corr)
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)
    hierarchy = linkage(condensed, method="average")
    labels = fcluster(hierarchy, t=1.0 - threshold, criterion="distance")

    representatives = []
    for label in sorted(np.unique(labels)):
        members = [names[i] for i in np.where(labels == label)[0]]
        representatives.append(
            max(members, key=lambda name: (importance.get(name, 0.0), name))
        )
    return representatives


def subset_cv_score(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> float:
    if not features:
        return -np.inf
    scores = cross_val_score(
        build_selection_estimator(),
        X[list(features)],
        y,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        error_score="raise",
    )
    return float(np.mean(scores))


def select_feature_block(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    block_name: str,
) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    """Fit all feature-selection steps using development data only."""
    estimator = build_selection_estimator()
    estimator.fit(X_train, y_train)
    raw_importance = estimator.feature_importances_
    importance = dict(zip(X_train.columns, raw_importance))

    retained = [
        name
        for name, value in importance.items()
        if value >= IMPORTANCE_THRESHOLD
    ]
    if not retained:
        retained = [max(importance, key=importance.get)]

    representatives = correlation_representatives(
        X_train, retained, importance, threshold=CORRELATION_THRESHOLD
    )

    if len(representatives) > RECURSIVE_CANDIDATE_CAP:
        representatives = sorted(
            representatives,
            key=lambda name: importance[name],
            reverse=True,
        )[:RECURSIVE_CANDIDATE_CAP]

    if len(representatives) <= 2:
        selected = representatives
        decision = "all retained representatives (<=2 features)"
        rfe_score = subset_cv_score(X_train, y_train, selected, cv_splits)
        rfa_score = rfe_score
        rfe_features = selected
        rfa_features = selected
    else:
        rfe_estimator = build_selection_estimator()
        rfecv = RFECV(
            estimator=rfe_estimator,
            step=max(1, len(representatives) // 20),
            min_features_to_select=1,
            cv=list(cv_splits),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        rfecv.fit(X_train[representatives], y_train)
        rfe_features = list(np.array(representatives)[rfecv.support_])
        rfe_score = subset_cv_score(
            X_train, y_train, rfe_features, cv_splits
        )

        rfa_estimator = build_selection_estimator()
        rfa = SequentialFeatureSelector(
            rfa_estimator,
            n_features_to_select="auto",
            tol=RFA_TOLERANCE,
            direction="forward",
            scoring="neg_mean_absolute_error",
            cv=list(cv_splits),
            n_jobs=-1,
        )
        rfa.fit(X_train[representatives], y_train)
        rfa_features = list(np.array(representatives)[rfa.get_support()])
        rfa_score = subset_cv_score(
            X_train, y_train, rfa_features, cv_splits
        )

        if (
            rfa_score >= rfe_score - RFA_VS_RFE_SCORE_TOLERANCE
            and len(rfa_features) < len(rfe_features)
        ):
            selected = rfa_features
            decision = "RFA"
        else:
            selected = rfe_features
            decision = "RFE"

    importance_table = pd.DataFrame(
        {
            "block": block_name,
            "feature": list(importance.keys()),
            "preliminary_importance": list(importance.values()),
        }
    ).sort_values("preliminary_importance", ascending=False)
    importance_table["passed_importance_threshold"] = importance_table[
        "feature"
    ].isin(retained)
    importance_table["correlation_representative"] = importance_table[
        "feature"
    ].isin(representatives)
    importance_table["selected"] = importance_table["feature"].isin(selected)

    report = {
        "block": block_name,
        "n_raw": int(X_train.shape[1]),
        "n_after_importance": int(len(retained)),
        "n_after_correlation": int(len(representatives)),
        "n_selected": int(len(selected)),
        "selected_features": list(selected),
        "selection_decision": decision,
        "rfe_features": list(rfe_features),
        "rfa_features": list(rfa_features),
        "rfe_cv_neg_mae": float(rfe_score),
        "rfa_cv_neg_mae": float(rfa_score),
        "importance_threshold": IMPORTANCE_THRESHOLD,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "recursive_candidate_cap": RECURSIVE_CANDIDATE_CAP,
    }
    return selected, importance_table, report


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_trials: int,
) -> Tuple[Dict[str, object], object]:
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1.0e-4, 1.0e-1, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1.0e-5, 1.0e2, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0e-5, 1.0e2, log=True
            ),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=list(cv_splits),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            error_score="raise",
        )
        return -float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )
    model = XGBRegressor(**best_params)
    return best_params, model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def grouped_cv_metrics(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X.iloc[train_idx], y[train_idx])
        prediction = fold_model.predict(X.iloc[valid_idx])
        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(valid_idx)),
                **evaluate_predictions(y[valid_idx], prediction),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(
    analysis: str,
    blocks: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
) -> Dict[str, object]:
    analysis_dir = output_dir / analysis
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx, outer_groups = make_outer_split(analysis, metadata)
    y = metadata["logD"].to_numpy(dtype=float)
    inner_groups = outer_groups[train_idx]

    ligand_key = "lig_canonical" if "lig_canonical" in metadata else "lig_SMI"
    ligand_train = set(metadata.iloc[train_idx][ligand_key])
    ligand_test = set(metadata.iloc[test_idx][ligand_key])
    group_train = set(outer_groups[train_idx])
    group_test = set(outer_groups[test_idx])

    split_summary = SplitSummary(
        analysis=analysis,
        n_total=len(metadata),
        n_train=len(train_idx),
        n_test=len(test_idx),
        test_fraction=len(test_idx) / len(metadata),
        n_train_outer_groups=len(group_train),
        n_test_outer_groups=len(group_test),
        outer_group_overlap=len(group_train & group_test),
        n_train_ligands=len(ligand_train),
        n_test_ligands=len(ligand_test),
        ligand_overlap=len(ligand_train & ligand_test),
        train_logd_mean=float(y[train_idx].mean()),
        test_logd_mean=float(y[test_idx].mean()),
        train_logd_std=float(y[train_idx].std(ddof=1)),
        test_logd_std=float(y[test_idx].std(ddof=1)),
    )

    # Identical group-aware folds are reused across feature blocks and tuning.
    dummy_train = blocks["experiment"].iloc[train_idx]
    cv_splits = make_inner_splits(
        dummy_train,
        y[train_idx],
        inner_groups,
        n_splits=N_SPLITS,
        group_aware=(analysis != "record"),
    )

    selected_all: List[str] = []
    importance_tables = []
    selection_reports = []
    for block_name in ["experiment", "descriptor", "fingerprint"]:
        block_train = blocks[block_name].iloc[train_idx].reset_index(drop=True)
        selected, table, report = select_feature_block(
            block_train,
            y[train_idx],
            cv_splits,
            block_name,
        )
        selected_all.extend(selected)
        importance_tables.append(table)
        selection_reports.append(report)

    combined = pd.concat(
        [blocks["experiment"], blocks["descriptor"], blocks["fingerprint"]],
        axis=1,
    )
    X_train = combined.iloc[train_idx][selected_all].reset_index(drop=True)
    X_test = combined.iloc[test_idx][selected_all].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    best_params, model = optimize_xgboost(
        X_train, y_train, cv_splits, n_trials=n_trials
    )
    cv_table = grouped_cv_metrics(model, X_train, y_train, cv_splits)

    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_metrics = evaluate_predictions(y_train, train_prediction)
    test_metrics = evaluate_predictions(y_test, test_prediction)

    predictions = pd.DataFrame(
        {
            "record_index": metadata.iloc[test_idx]["record_index"].to_numpy(),
            "lig_SMI": metadata.iloc[test_idx]["lig_SMI"].to_numpy(),
            "lig_canonical": metadata.iloc[test_idx][ligand_key].to_numpy(),
            "Sol_SMI": metadata.iloc[test_idx]["Sol_SMI"].to_numpy(),
            "experimental_logD": y_test,
            "predicted_logD": test_prediction,
        }
    )

    if analysis == "ligand":
        partition = np.full(len(metadata), "train", dtype=object)
        partition[test_idx] = "test"
        inventory_source = metadata.assign(partition=partition)
        inventory_rows = []
        for canonical, group in inventory_source.groupby(ligand_key, sort=True):
            inventory_rows.append(
                {
                    "lig_canonical": canonical,
                    "partition": group["partition"].iloc[0],
                    "n_records": int(len(group)),
                    "n_raw_smiles_aliases": int(group["lig_SMI"].nunique()),
                    "raw_smiles_aliases": " || ".join(
                        sorted(group["lig_SMI"].astype(str).unique())
                    ),
                }
            )
        pd.DataFrame(inventory_rows).to_csv(
            analysis_dir / "ligand_split_inventory.csv", index=False
        )

    pd.concat(importance_tables, ignore_index=True).to_csv(
        analysis_dir / "feature_selection_audit.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "feature": feature,
                "block": next(
                    name for name, block in blocks.items() if feature in block.columns
                ),
            }
            for feature in selected_all
        ]
    ).to_csv(analysis_dir / "selected_features.csv", index=False)
    cv_table.to_csv(analysis_dir / "development_cv_folds.csv", index=False)
    predictions.to_csv(analysis_dir / "holdout_predictions.csv", index=False)
    joblib.dump(model, analysis_dir / "final_model.joblib")

    with open(analysis_dir / "best_params.json", "w", encoding="utf-8") as stream:
        json.dump(best_params, stream, indent=2, ensure_ascii=False)
    with open(
        analysis_dir / "feature_selection_report.json", "w", encoding="utf-8"
    ) as stream:
        json.dump(selection_reports, stream, indent=2, ensure_ascii=False)
    with open(analysis_dir / "split_summary.json", "w", encoding="utf-8") as stream:
        json.dump(asdict(split_summary), stream, indent=2, ensure_ascii=False)

    cv_summary = {
        metric: {
            "mean": float(cv_table[metric].mean()),
            "sd": float(cv_table[metric].std(ddof=1)),
        }
        for metric in ["R2", "MAE", "RMSE"]
    }
    result = {
        "analysis": analysis,
        "split": asdict(split_summary),
        "n_selected_features": len(selected_all),
        "selected_features": selected_all,
        "development_cv": cv_summary,
        "train_metrics": train_metrics,
        "holdout_metrics": test_metrics,
        "best_params": best_params,
    }
    with open(analysis_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


def software_versions() -> Dict[str, str]:
    versions = {
        "Python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name, display_name in [
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "XGBoost"),
        ("optuna", "Optuna"),
        ("shap", "SHAP"),
    ]:
        try:
            module = __import__(module_name)
            versions[display_name] = str(module.__version__)
        except ImportError:
            versions[display_name] = "not installed"
    try:
        from rdkit import rdBase

        versions["RDKit"] = rdBase.rdkitVersion
    except ImportError:
        versions["RDKit"] = "not installed"
    return versions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    metadata["lig_canonical"] = canonicalize_smiles(
        metadata["lig_SMI"], label="lig"
    )
    metadata["sol_canonical"] = canonicalize_smiles(
        metadata["Sol_SMI"], label="sol"
    )
    tables = read_inputs(args.input_dir)
    audit = audit_alignment(tables, metadata)

    descriptor_repair = {"skipped": True}
    if not args.skip_descriptor_repair:
        repaired, descriptor_repair = repair_missing_2d_descriptors(
            tables["ligand_des"], metadata["lig_SMI"], prefix="lig_"
        )
        tables["ligand_des"] = repaired

    valid_target = metadata["logD"].notna().to_numpy()
    excluded_record_indices = metadata.loc[
        ~valid_target, "record_index"
    ].astype(int).tolist()

    metadata = metadata.loc[valid_target].reset_index(drop=True)
    for name in tables:
        tables[name] = tables[name].loc[valid_target].reset_index(drop=True)

    raw_blocks = {
        "experiment": tables["env"].astype(float),
        "descriptor": pd.concat(
            [tables["ligand_des"], tables["sol_des"]], axis=1
        ).astype(float),
        "fingerprint": pd.concat(
            [tables["ligand_mor"], tables["sol_mor"]], axis=1
        ).astype(float),
    }
    blocks = {}
    value_sanitization = {}
    for block_name, block in raw_blocks.items():
        blocks[block_name], value_sanitization[block_name] = (
            sanitize_xgboost_values(block)
        )

    if len(set(sum((list(block.columns) for block in blocks.values()), []))) != sum(
        block.shape[1] for block in blocks.values()
    ):
        raise ValueError("Feature names are not unique across the three blocks")

    audit.update(
        {
            "excluded_missing_target_record_indices": excluded_record_indices,
            "n_records_after_target_exclusion": len(metadata),
            "feature_block_shapes_after_exclusion": {
                name: list(block.shape) for name, block in blocks.items()
            },
            "descriptor_repair": descriptor_repair,
            "xgboost_value_sanitization": value_sanitization,
        }
    )
    with open(args.output_dir / "data_audit.json", "w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
    with open(
        args.output_dir / "software_versions.json", "w", encoding="utf-8"
    ) as stream:
        json.dump(software_versions(), stream, indent=2, ensure_ascii=False)

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.validate_only:
        print("Input validation completed; no model was fitted.")
        return

    require_model_packages()
    if args.analyses == "both":
        analyses = ["condition", "ligand"]
    elif args.analyses == "all":
        analyses = ["record", "condition", "ligand"]
    else:
        analyses = [args.analyses]
    results = []
    for analysis in analyses:
        print(f"\n===== Running {analysis}-grouped validation =====")
        results.append(
            run_analysis(
                analysis,
                blocks,
                metadata,
                args.output_dir,
                n_trials=args.n_trials,
            )
        )

    with open(args.output_dir / "all_results.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    summary_rows = []
    for result in results:
        row = {
            "analysis": result["analysis"],
            "n_train": result["split"]["n_train"],
            "n_test": result["split"]["n_test"],
            "test_fraction": result["split"]["test_fraction"],
            "n_selected_features": result["n_selected_features"],
        }
        for prefix, metrics in [
            ("train", result["train_metrics"]),
            ("holdout", result["holdout_metrics"]),
        ]:
            for metric, value in metrics.items():
                row[f"{prefix}_{metric}"] = value
        for metric, values in result["development_cv"].items():
            row[f"cv_{metric}_mean"] = values["mean"]
            row[f"cv_{metric}_sd"] = values["sd"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "validation_summary.csv", index=False
    )
    print(f"\nCompleted. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
