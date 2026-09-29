"""Create the development-v2 causal producer bundle; never run or score outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.records import Observation
from airproof.v6_mechanistic_input_producer import produce_provenance_complete_bundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True,
                        help="Explicit causal arrays plus calibration_truth only")
    parser.add_argument("--observations-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = np.load(args.source_npz, allow_pickle=False)
    forbidden = {"truth", "evaluation_truth", "attack", "attack_flag", "attack_label", "event_mask", "corrupted"}
    if forbidden & set(source.files):
        raise ValueError("source NPZ contains scoring fields outside calibration_truth")
    required = {
        "public_center", "prediction_epochs", "public_center_available_at", "transitions",
        "transition_latest_input_at", "physical_forcing",
        "physical_forcing_latest_input_at", "previous_public_center",
        "calibration_truth", "calibration_truth_available_at",
    }
    if set(source.files) != required:
        raise ValueError(f"source NPZ schema mismatch: expected exactly {sorted(required)}")
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.observations_jsonl.read_text(
        encoding="utf-8").splitlines() if line]
    observations = []
    for row in rows:
        if any(name in row for name in forbidden):
            raise ValueError("observation source contains scoring labels")
        observations.append(Observation(**row, corrupted=False))
    source_hashes = {
        str(args.source_npz): sha256(args.source_npz),
        str(args.observations_jsonl): sha256(args.observations_jsonl),
        str(args.metadata_json): sha256(args.metadata_json),
        "airproof/v6_mechanistic_input_producer.py": sha256(
            ROOT / "airproof/v6_mechanistic_input_producer.py"),
    }
    manifest = produce_provenance_complete_bundle(
        args.output_dir, observations=observations, public_center=source["public_center"],
        prediction_epochs=source["prediction_epochs"],
        public_center_available_at=source["public_center_available_at"],
        transitions=source["transitions"],
        transition_latest_input_at=source["transition_latest_input_at"],
        physical_forcing=source["physical_forcing"],
        physical_forcing_latest_input_at=source["physical_forcing_latest_input_at"],
        previous_public_center=source["previous_public_center"],
        calibration_half_open=tuple(metadata["calibration_half_open"]),
        evaluation_half_open=tuple(metadata["evaluation_half_open"]),
        calibration_truth={metadata["calibration_world_id"]: source["calibration_truth"]},
        calibration_truth_available_at={metadata["calibration_world_id"]:
            source["calibration_truth_available_at"]},
        calibration_split_id=metadata["calibration_split_id"],
        evaluation_split_id=metadata["evaluation_split_id"],
        public_model_id=metadata["public_model_id"],
        sensor_model_id=metadata["sensor_model_id"],
        source_roles=metadata["source_roles"],
        calibration_pair_context=metadata["calibration_pair_context"],
        development_seed_id=metadata["development_seed_id"],
        cross_fit_folds=int(metadata.get("cross_fit_folds", 2)),
        source_hashes=source_hashes,
    )
    print(json.dumps({
        "producer_namespace": manifest["producer_namespace"],
        "bundle_root_sha256": manifest["bundle_root_sha256"],
        "authorizes_outcome_execution": False,
    }, indent=2))


if __name__ == "__main__":
    main()
