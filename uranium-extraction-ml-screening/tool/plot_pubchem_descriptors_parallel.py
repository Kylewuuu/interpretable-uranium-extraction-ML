from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D


RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path("/home/wyh/data/examples/U_logD")
RANDOM_STATE = 42
DEFAULT_REQUESTED_SIZE = 24_442
DEFAULT_WORKERS = 32

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

_SELECTED: list[str] = []
_BASE_NAMES: list[str] = []
_DESC_FUNCTIONS: dict[str, object] = {}
_MAX_HEAVY_ATOMS = 80
_MAX_TOTAL_ATOMS = 250
_EMBED_TIMEOUT_SECONDS = 20
_MMFF_MAX_ITERS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel, checkpointed calculation of the selected PubChem ligand "
            "descriptors required by plot_pubchem_screening_tsne_v2.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "reviewer3_comment4/complete_pubchem_raw_sample.csv",
        help="Existing random sample containing CID and SMILES columns.",
    )
    parser.add_argument(
        "--selected-summary",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/selected_descriptor_similarity_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/complete_pubchem_sample_selected_descriptors.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR
        / "reviewer3_comment4/complete_pubchem_descriptor_checkpoint.csv",
    )
    parser.add_argument(
        "--requested-size", type=int, default=DEFAULT_REQUESTED_SIZE
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-heavy-atoms", type=int, default=80)
    parser.add_argument("--max-total-atoms", type=int, default=250)
    parser.add_argument("--embed-timeout-seconds", type=int, default=20)
    parser.add_argument("--mmff-max-iters", type=int, default=200)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing checkpoint and start again from the raw sample.",
    )
    return parser.parse_args()


def load_selected_descriptors(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = list(summary["feature_selection"]["selected_descriptors"])
    if not selected:
        raise ValueError("No selected descriptors were found in the summary file.")
    return selected


def normalize_columns(columns: list[str]) -> tuple[str, str]:
    mapping = {str(column).strip().upper(): str(column) for column in columns}
    if "CID" not in mapping or "SMILES" not in mapping:
        raise KeyError("The input CSV must contain CID and SMILES columns.")
    return mapping["CID"], mapping["SMILES"]


def init_worker(
    selected: list[str],
    max_heavy_atoms: int,
    max_total_atoms: int,
    embed_timeout_seconds: int,
    mmff_max_iters: int,
) -> None:
    global _SELECTED
    global _BASE_NAMES
    global _DESC_FUNCTIONS
    global _MAX_HEAVY_ATOMS
    global _MAX_TOTAL_ATOMS
    global _EMBED_TIMEOUT_SECONDS
    global _MMFF_MAX_ITERS

    _SELECTED = selected
    _BASE_NAMES = [name.removeprefix("lig_") for name in selected]
    _DESC_FUNCTIONS = dict(Descriptors._descList)
    _MAX_HEAVY_ATOMS = max_heavy_atoms
    _MAX_TOTAL_ATOMS = max_total_atoms
    _EMBED_TIMEOUT_SECONDS = embed_timeout_seconds
    _MMFF_MAX_ITERS = mmff_max_iters
    RDLogger.DisableLog("rdApp.*")


def failed_result(
    sample_index: int, cid: str, smiles: str, status: str
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "CID": cid,
        "SMILES": smiles,
        "status": status,
    }


def calculate_one(record: tuple[int, str, str]) -> dict[str, object]:
    sample_index, cid, smiles = record
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return failed_result(sample_index, cid, smiles, "invalid_smiles")
        if mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
            return failed_result(sample_index, cid, smiles, "too_many_heavy_atoms")

        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() > _MAX_TOTAL_ATOMS:
            return failed_result(sample_index, cid, smiles, "too_many_total_atoms")

        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_STATE
        params.numThreads = 1
        params.maxIterations = 200
        if hasattr(params, "timeout"):
            params.timeout = _EMBED_TIMEOUT_SECONDS
        if AllChem.EmbedMolecule(mol, params) != 0:
            return failed_result(sample_index, cid, smiles, "embedding_failed")

        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return failed_result(sample_index, cid, smiles, "missing_mmff_parameters")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=_MMFF_MAX_ITERS)

        result: dict[str, object] = {
            "sample_index": sample_index,
            "CID": cid,
            "SMILES": smiles,
            "status": "ok",
        }
        for full_name, base_name in zip(_SELECTED, _BASE_NAMES):
            if base_name in THREE_D_DESCRIPTORS:
                function = getattr(Descriptors3D, base_name)
            else:
                function = _DESC_FUNCTIONS.get(base_name)
                if function is None:
                    return failed_result(
                        sample_index, cid, smiles, f"unknown_descriptor:{base_name}"
                    )
            value = float(function(mol))
            if not np.isfinite(value):
                return failed_result(
                    sample_index, cid, smiles, f"nonfinite_descriptor:{base_name}"
                )
            result[full_name] = value
        return result
    except Exception as exc:
        return failed_result(
            sample_index, cid, smiles, f"error:{type(exc).__name__}"
        )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_checkpoint(path: Path, restart: bool) -> pd.DataFrame:
    if restart or not path.exists():
        return pd.DataFrame()
    checkpoint = pd.read_csv(path, dtype={"CID": str, "SMILES": str})
    if "sample_index" not in checkpoint.columns or "status" not in checkpoint.columns:
        raise ValueError(f"Checkpoint has an unexpected format: {path}")
    checkpoint["sample_index"] = pd.to_numeric(
        checkpoint["sample_index"], errors="raise"
    ).astype(int)
    checkpoint = checkpoint.drop_duplicates("sample_index", keep="last")
    return checkpoint.sort_values("sample_index").reset_index(drop=True)


def build_final_output(
    checkpoint: pd.DataFrame,
    selected: list[str],
    requested_size: int,
    output_path: Path,
) -> pd.DataFrame:
    valid = checkpoint.loc[checkpoint["status"] == "ok"].copy()
    missing = [name for name in selected if name not in valid.columns]
    if missing:
        raise KeyError(f"Checkpoint is missing selected descriptors: {missing}")
    valid = valid.sort_values("sample_index").head(requested_size)
    if len(valid) < requested_size:
        raise ValueError(
            f"Only {len(valid):,} valid rows are available; "
            f"{requested_size:,} are required."
        )
    final = valid[["CID", "SMILES", *selected]].copy()
    atomic_write_csv(final, output_path)
    return final


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.requested_size < 1:
        raise ValueError("--requested-size must be at least 1.")

    selected = load_selected_descriptors(args.selected_summary)
    raw = pd.read_csv(args.input, dtype=str)
    cid_column, smiles_column = normalize_columns(list(raw.columns))
    raw = raw.rename(columns={cid_column: "CID", smiles_column: "SMILES"})
    raw = raw[["CID", "SMILES"]].dropna().reset_index(drop=True)
    raw.insert(0, "sample_index", np.arange(len(raw), dtype=int))

    checkpoint = load_checkpoint(args.checkpoint, args.restart)
    processed_indices = (
        set(checkpoint["sample_index"].astype(int)) if len(checkpoint) else set()
    )
    valid_count = (
        int((checkpoint["status"] == "ok").sum()) if len(checkpoint) else 0
    )
    if valid_count >= args.requested_size:
        final = build_final_output(
            checkpoint, selected, args.requested_size, args.output
        )
        print(f"Output already complete: {args.output} ({len(final):,} rows)")
        return

    pending = [
        (int(row.sample_index), str(row.CID), str(row.SMILES))
        for row in raw.itertuples(index=False)
        if int(row.sample_index) not in processed_indices
    ]
    if not pending:
        raise ValueError(
            "No unprocessed rows remain, but the requested valid count was not reached."
        )

    print(f"Selected descriptors: {len(selected)}")
    print(f"Raw sample rows: {len(raw):,}")
    print(f"Resuming processed rows: {len(checkpoint):,}")
    print(f"Resuming valid rows: {valid_count:,}")
    print(f"Pending rows: {len(pending):,}")
    print(f"Worker processes: {args.workers}")
    print(f"Target valid rows: {args.requested_size:,}")

    rows = checkpoint.to_dict("records") if len(checkpoint) else []
    completed_this_run = 0
    last_saved_count = len(rows)
    started = time.monotonic()

    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(
            selected,
            args.max_heavy_atoms,
            args.max_total_atoms,
            args.embed_timeout_seconds,
            args.mmff_max_iters,
        ),
        maxtasksperchild=100,
    )

    reached_target = False
    try:
        iterator = pool.imap_unordered(calculate_one, pending, chunksize=1)
        for result in iterator:
            rows.append(result)
            completed_this_run += 1
            if result["status"] == "ok":
                valid_count += 1

            if (
                len(rows) - last_saved_count >= args.checkpoint_every
                or valid_count >= args.requested_size
            ):
                checkpoint = pd.DataFrame(rows).drop_duplicates(
                    "sample_index", keep="last"
                )
                atomic_write_csv(checkpoint, args.checkpoint)
                last_saved_count = len(rows)

            if completed_this_run % 100 == 0 or valid_count >= args.requested_size:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = completed_this_run / elapsed
                print(
                    f"Processed this run={completed_this_run:,}; "
                    f"total checkpointed={len(rows):,}; valid={valid_count:,}/"
                    f"{args.requested_size:,}; rate={rate:.2f} molecules/s",
                    flush=True,
                )

            if valid_count >= args.requested_size:
                reached_target = True
                pool.terminate()
                break

        if not reached_target:
            pool.close()
    except KeyboardInterrupt:
        print("Interrupted; saving checkpoint before exit...", flush=True)
        pool.terminate()
        checkpoint = pd.DataFrame(rows).drop_duplicates("sample_index", keep="last")
        atomic_write_csv(checkpoint, args.checkpoint)
        raise
    finally:
        pool.join()

    checkpoint = pd.DataFrame(rows).drop_duplicates("sample_index", keep="last")
    atomic_write_csv(checkpoint, args.checkpoint)
    final = build_final_output(checkpoint, selected, args.requested_size, args.output)

    status_counts = checkpoint["status"].value_counts().to_dict()
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "requested_valid_rows": args.requested_size,
        "output_rows": len(final),
        "workers": args.workers,
        "max_heavy_atoms": args.max_heavy_atoms,
        "max_total_atoms": args.max_total_atoms,
        "embed_timeout_seconds": args.embed_timeout_seconds,
        "mmff_max_iters": args.mmff_max_iters,
        "status_counts": status_counts,
    }
    summary_path = args.output.with_name(
        "complete_pubchem_descriptor_parallel_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Descriptor cache: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
