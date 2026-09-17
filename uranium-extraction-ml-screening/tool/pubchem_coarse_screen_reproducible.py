from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors
from rdkit.Chem.EState import EState_VSA


RDLogger.DisableLog("rdApp.*")

# The allowed set covers the nonmetals commonly encountered in neutral organic
# extractants. Metals and disconnected salts are excluded separately.
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

# The uranium dataset is overwhelmingly amide based and also contains TBP.
# Therefore, at least one amide or phosphoryl group is required.
AMIDE_SMARTS = "[CX3](=[OX1])[NX3]"
PHOSPHORYL_SMARTS = "[PX4,PX5](=[OX1])"

# Preserve the screening-set size reported in the manuscript. When more eligible
# molecules are available, a deterministic hash-based sample is used so that the
# same PubChem snapshot and seed always produce the same working set.
DEFAULT_TARGET_COUNT = 1_178_295
DEFAULT_EXPECTED_INPUT_COUNT = 122_207_703
DEFAULT_SEED = 42

_AMIDE_QUERY = None
_PHOSPHORYL_QUERY = None
_BOUNDS: dict[str, tuple[float, float]] = {}
_SEED = DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible coarse screening of a PubChem CID-SMILES snapshot for "
            "uranium-extractant virtual screening."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/kylewu/hotpot/examples/CID_SMILES.csv"),
        help="PubChem CID-SMILES CSV or CSV.GZ file.",
    )
    parser.add_argument(
        "--training-descriptors",
        type=Path,
        default=Path(
            "/home/kylewu/hotpot/examples/U_logD/data/input/ligand_des.xlsx"
        ),
        help="Training ligand descriptor table used to derive numerical bounds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/kylewu/hotpot/examples/U_logD/pubchem_screening"),
    )
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Download/snapshot date of the PubChem CID-SMILES file (YYYY-MM-DD).",
    )
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument(
        "--expected-input-count",
        type=int,
        default=DEFAULT_EXPECTED_INPUT_COUNT,
        help=(
            "Expected number of records in the PubChem snapshot. The script stops "
            "if the actual count differs; pass 0 to disable this check."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--map-chunk-size", type=int, default=500)
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def normalize_columns(columns: Iterable[str]) -> dict[str, str]:
    mapping = {str(column).strip().upper(): str(column) for column in columns}
    if "CID" not in mapping or "SMILES" not in mapping:
        raise KeyError("The input file must contain CID and SMILES columns.")
    return mapping


def derive_training_bounds(path: Path) -> dict[str, tuple[float, float]]:
    """
    Use the observed range of unique training ligands as the descriptor boundary.
    These bounds are determined independently of PubChem prediction values.
    """
    feature_columns = [
        "lig_HeavyAtomCount",
        "lig_MolWt",
        "lig_MolLogP",
        "lig_AvgIpc",
        "lig_VSA_EState5",
    ]
    frame = pd.read_excel(path, usecols=feature_columns)
    unique_ligands = frame.drop_duplicates().dropna()
    if unique_ligands.empty:
        raise ValueError("No complete training-ligand descriptor rows were available.")

    bounds = {
        "heavy_atoms": (
            float(unique_ligands["lig_HeavyAtomCount"].min()),
            99.0,
        ),
        "mol_wt": (
            float(unique_ligands["lig_MolWt"].min()),
            float(unique_ligands["lig_MolWt"].max()),
        ),
        "mol_logp": (
            float(unique_ligands["lig_MolLogP"].min()),
            float(unique_ligands["lig_MolLogP"].max()),
        ),
        "avg_ipc": (
            float(unique_ligands["lig_AvgIpc"].min()),
            float(unique_ligands["lig_AvgIpc"].max()),
        ),
        "vsa_estate5": (
            float(unique_ligands["lig_VSA_EState5"].min()),
            float(unique_ligands["lig_VSA_EState5"].max()),
        ),
    }
    return bounds


def init_worker(bounds: dict[str, tuple[float, float]], seed: int) -> None:
    global _AMIDE_QUERY, _PHOSPHORYL_QUERY, _BOUNDS, _SEED
    _AMIDE_QUERY = Chem.MolFromSmarts(AMIDE_SMARTS)
    _PHOSPHORYL_QUERY = Chem.MolFromSmarts(PHOSPHORYL_SMARTS)
    _BOUNDS = bounds
    _SEED = seed


def bounded(value: float, key: str) -> bool:
    lower, upper = _BOUNDS[key]
    return bool(np.isfinite(value) and lower <= value <= upper)


def deterministic_key(cid: str, canonical_smiles: str) -> int:
    digest = hashlib.sha256(
        f"{_SEED}|{cid}|{canonical_smiles}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def evaluate_record(record: tuple[str, str]):
    cid, smiles = record
    passed: list[str] = ["input_records"]

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return passed, None
    passed.append("valid_smiles")

    if len(Chem.GetMolFrags(mol)) != 1:
        return passed, None
    passed.append("single_component")

    if Chem.GetFormalCharge(mol) != 0:
        return passed, None
    passed.append("neutral_molecule")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not elements.issubset(ALLOWED_ELEMENTS):
        return passed, None
    passed.append("allowed_elements")

    heavy_atoms = int(mol.GetNumHeavyAtoms())
    if not bounded(float(heavy_atoms), "heavy_atoms"):
        return passed, None
    passed.append("heavy_atom_count")

    if not (
        mol.HasSubstructMatch(_AMIDE_QUERY)
        or mol.HasSubstructMatch(_PHOSPHORYL_QUERY)
    ):
        return passed, None
    passed.append("amide_or_phosphoryl_motif")

    # The training descriptor workflow calculated descriptors after adding H atoms.
    mol_h = Chem.AddHs(mol)
    mol_wt = float(Descriptors.MolWt(mol_h))
    if not bounded(mol_wt, "mol_wt"):
        return passed, None
    passed.append("molecular_weight")

    mol_logp = float(Crippen.MolLogP(mol_h))
    if not bounded(mol_logp, "mol_logp"):
        return passed, None
    passed.append("hydrophobicity_mollogp")

    avg_ipc = float(GraphDescriptors.AvgIpc(mol_h))
    if not bounded(avg_ipc, "avg_ipc"):
        return passed, None
    passed.append("avg_ipc")

    vsa_estate5 = float(EState_VSA.VSA_EState5(mol_h))
    if not bounded(vsa_estate5, "vsa_estate5"):
        return passed, None
    passed.append("vsa_estate5")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    result = (
        deterministic_key(str(cid), canonical_smiles),
        str(cid),
        canonical_smiles,
        heavy_atoms,
        mol_wt,
        mol_logp,
        avg_ipc,
        vsa_estate5,
    )
    passed.append("eligible_after_rule_filters")
    return passed, result


def input_batches(path: Path, chunk_size: int):
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        column_map = normalize_columns(reader.fieldnames or [])
        cid_column = column_map["CID"]
        smiles_column = column_map["SMILES"]
        batch: list[tuple[str, str]] = []
        for row in reader:
            batch.append((row[cid_column], row[smiles_column]))
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch


def prepare_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS eligible (
            selection_key INTEGER NOT NULL,
            cid TEXT NOT NULL,
            smiles TEXT NOT NULL,
            heavy_atoms INTEGER NOT NULL,
            mol_wt REAL NOT NULL,
            mol_logp REAL NOT NULL,
            avg_ipc REAL NOT NULL,
            vsa_estate5 REAL NOT NULL
        )
        """
    )
    return connection


def export_working_set(
    connection: sqlite3.Connection, output_path: Path, target_count: int
) -> int:
    eligible_count = int(connection.execute("SELECT COUNT(*) FROM eligible").fetchone()[0])
    if eligible_count < target_count:
        raise ValueError(
            f"Only {eligible_count:,} molecules passed the rules; this is fewer than "
            f"the requested working-set size of {target_count:,}."
        )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_eligible_selection_key "
        "ON eligible(selection_key)"
    )
    query = (
        "SELECT cid, smiles, heavy_atoms, mol_wt, mol_logp, avg_ipc, vsa_estate5 "
        "FROM eligible ORDER BY selection_key ASC LIMIT ?"
    )
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "CID",
                "SMILES",
                "HeavyAtomCount",
                "MolWt",
                "MolLogP",
                "AvgIpc",
                "VSA_EState5",
            ]
        )
        exported = 0
        cursor = connection.execute(query, (target_count,))
        while True:
            rows = cursor.fetchmany(100_000)
            if not rows:
                break
            writer.writerows(rows)
            exported += len(rows)
    return exported


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bounds = derive_training_bounds(args.training_descriptors)
    database_path = args.output_dir / "eligible_candidates.sqlite"
    connection = prepare_database(database_path)
    counts: Counter[str] = Counter()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(bounds, args.seed),
    ) as pool:
        for batch_number, batch in enumerate(
            input_batches(args.input, args.chunk_size), start=1
        ):
            accepted_rows = []
            for passed, result in pool.map(
                evaluate_record, batch, chunksize=args.map_chunk_size
            ):
                counts.update(passed)
                if result is not None:
                    accepted_rows.append(result)

            if accepted_rows:
                connection.executemany(
                    "INSERT INTO eligible VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    accepted_rows,
                )
                connection.commit()

            print(
                f"Batch {batch_number}: processed={counts['input_records']:,}, "
                f"eligible={counts['eligible_after_rule_filters']:,}"
            )

    actual_input_count = int(counts["input_records"])
    if args.expected_input_count and actual_input_count != args.expected_input_count:
        connection.close()
        raise ValueError(
            f"The input contains {actual_input_count:,} records, but "
            f"{args.expected_input_count:,} were expected. Check the PubChem snapshot "
            "before reporting the screening counts."
        )

    output_path = args.output_dir / f"CID_SMILES_screened_{args.target_count}.csv"
    exported = export_working_set(connection, output_path, args.target_count)
    eligible_count = int(counts["eligible_after_rule_filters"])
    connection.close()

    ordered_stages = [
        "input_records",
        "valid_smiles",
        "single_component",
        "neutral_molecule",
        "allowed_elements",
        "heavy_atom_count",
        "amide_or_phosphoryl_motif",
        "molecular_weight",
        "hydrophobicity_mollogp",
        "avg_ipc",
        "vsa_estate5",
        "eligible_after_rule_filters",
    ]
    summary = {
        "pubchem_snapshot_date": args.snapshot_date,
        "input_file": str(args.input),
        "expected_input_count": args.expected_input_count,
        "actual_input_count": actual_input_count,
        "allowed_elements": sorted(ALLOWED_ELEMENTS),
        "structure_rules": {
            "single_component": True,
            "formal_charge": 0,
            "amide_smarts": AMIDE_SMARTS,
            "phosphoryl_smarts": PHOSPHORYL_SMARTS,
            "motif_rule": "amide OR phosphoryl",
        },
        "descriptor_bounds_from_unique_training_ligands": {
            key: {"minimum": value[0], "maximum": value[1]}
            for key, value in bounds.items()
        },
        "stage_counts": {stage: int(counts[stage]) for stage in ordered_stages},
        "working_set": {
            "selection_method": (
                "Deterministic SHA-256 ordering of CID and canonical SMILES after "
                "all rule-based filters"
            ),
            "random_seed": args.seed,
            "eligible_count_before_fixed_set_selection": eligible_count,
            "retained_count": exported,
        },
        "shap_fingerprint_rule": (
            "No hard SHAP-associated fingerprint-bit filter was applied; the selected "
            "Morgan bits were retained as prediction-model inputs."
        ),
    }
    with open(
        args.output_dir / "pubchem_screening_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "Step": ordered_stages + ["fixed_working_screening_set"],
            "Remaining_count": [counts[stage] for stage in ordered_stages] + [exported],
        }
    ).to_csv(args.output_dir / "pubchem_screening_stage_counts.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nFinal working set: {output_path}")


if __name__ == "__main__":
    main()
