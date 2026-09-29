"""Build the registered multi-world v2 input campaign without efficacy outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import (
    build_calibration_world,
    build_development_world,
    build_global_calibration,
    expected_physical_operator,
    finalize_campaign_bundle,
    load_registration,
    save_csr,
    sha256_file,
    verify_campaign_bundle,
    write_json,
)


def _task(kind: str, root: str, stage: str, seed: int):
    if kind == "calibration":
        return kind, build_calibration_world(root, stage, seed)
    if kind == "development":
        return kind, build_development_world(root, stage, seed)
    raise ValueError("unknown input-build role")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    final = ROOT / registration["producer"]["campaign_bundle"]
    if args.verify:
        manifest = verify_campaign_bundle(ROOT, final)
        print(json.dumps({"verified": True, "manifest_sha256": sha256_file(final / "manifest.json"),
                          "payloads": len(manifest["payload_sha256"])}, indent=2))
        return
    if not 1 <= args.workers <= int(registration["runtime"]["workers_max"]):
        raise SystemExit("workers outside registered range")
    if final.exists():
        raise SystemExit("final input namespace already exists; overwrite prohibited; use --verify")
    stage = Path(str(final) + ".building")
    registration_sha = sha256_file(ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json")
    if stage.exists():
        state_path = stage / "build_state.json"
        if not state_path.is_file() or json.loads(state_path.read_text()).get("registration_sha256") != registration_sha:
            raise SystemExit("stale staging namespace has a different registration; preserve and inspect it")
    else:
        stage.mkdir(parents=True)
        write_json(stage / "build_state.json", {"registration_sha256": registration_sha,
            "role": "resumable input generation; zero efficacy outcomes"})
        mechanism = stage / "mechanism"
        mechanism.mkdir()
        save_csr(mechanism / "physical_operator.npz",
                 expected_physical_operator(int(registration["scale"]["grid_side"])))

    tasks = []
    for seed in registration["world_roles"]["calibration"]["seeds"]:
        if not (stage / "calibration" / str(seed) / "summary.json").is_file():
            tasks.append(("calibration", str(ROOT), str(stage), int(seed)))
    for seed in registration["world_roles"]["development"]["seeds"]:
        if not (stage / "prediction" / str(seed) / "summary.json").is_file():
            tasks.append(("development", str(ROOT), str(stage), int(seed)))
    finished = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_task, *task) for task in tasks]
            for future in as_completed(futures):
                kind, summary = future.result()
                finished.append((kind, summary))
                write_json(stage / "progress.json", {"newly_completed": len(finished),
                    "submitted": len(tasks), "last_role": kind, "last_seed": summary["seed"],
                    "efficacy_outcomes": 0})
    calibration = [json.loads(path.read_text()) for path in
        sorted((stage / "calibration").glob("*/summary.json"))]
    development = [json.loads(path.read_text()) for path in
        sorted((stage / "prediction").glob("*/summary.json"))]
    if (len(calibration) != len(registration["world_roles"]["calibration"]["seeds"])
            or len(development) != len(registration["world_roles"]["development"]["seeds"])):
        raise RuntimeError("input campaign is incomplete after worker completion")
    build_global_calibration(ROOT, stage)
    finalize_campaign_bundle(ROOT, stage, calibration, development)
    (stage / "build_state.json").unlink()
    # Recreate the manifest after removing the transient state so the inventory is exact.
    (stage / "manifest.json").unlink()
    finalize_campaign_bundle(ROOT, stage, calibration, development)
    stage.rename(final)
    manifest = verify_campaign_bundle(ROOT, final)
    print(json.dumps({"completed": True, "calibration_worlds": len(calibration),
        "development_worlds": len(development),
        "manifest_sha256": sha256_file(final / "manifest.json"),
        "payloads": len(manifest["payload_sha256"]), "scientific_outcomes": 0}, indent=2))


if __name__ == "__main__":
    main()
