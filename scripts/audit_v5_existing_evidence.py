"""Read-only historical evidence inventory and archive exposure audit (no RPC)."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v5_audit import sha256_file


def artifact(name):
    p = ROOT / name
    return {"path": name, "exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None,
            "sha256": sha256_file(p) if p.is_file() else None}


def main():
    roots = ["airproof", "configs", "data", "docs", "reports", "results", "scripts", "source_paper"]
    found = subprocess.run(["rg", "--files", *roots], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    found += [p.name for p in ROOT.iterdir() if p.is_file()]
    extensions = {".json", ".jsonl", ".yaml", ".yml", ".py", ".md", ".csv", ".tex", ".log", ".txt"}
    pattern = re.compile(r"\bepa\b|epa_|beijing|6ea6b823d7eee44e4961f678e1c8c23b0f7de776d6d39a78bcb56ab30ae3d9c3", re.I)
    scanned, matches, unreadable = [], [], []
    for name in sorted(set(found)):
        name = name.replace("\\", "/")
        if "/v5/" in name or "V5_" in name or "v5_" in name or Path(name).suffix not in extensions:
            continue
        p = ROOT / name
        try:
            raw = p.read_bytes()
            content = raw.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            unreadable.append({"path": name, "reason": type(exc).__name__})
            continue
        scanned.append({"path": name, "sha256": sha256_file(p)})
        if pattern.search(name) or pattern.search(content):
            lines = [{"line": i, "text": line[:1000]} for i, line in enumerate(content.splitlines(), 1) if pattern.search(line)]
            matches.append({"path": name, "sha256": sha256_file(p), "matched_lines": lines})
    historical = []
    for name in ("data/external/epa/epa_transfer_preflight.json", "results/fullpaper30h_20260902/results/epa_transfer.json"):
        data = json.loads((ROOT / name).read_text())
        start = data["train_rows"] + data["purge_hours"]
        historical.append({**artifact(name), "dataset": "EPA", "fit_indices": [0, data["train_rows"]],
                           "test_indices": [start, start + data["test_steps"]], "interval_convention": "half-open"})
    import pandas as pd
    frame = pd.read_parquet(ROOT / "data/external/epa/epa_bay_area_pm25_2024.parquet")
    pivot = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    candidates = []
    for start, stop in ((7944, 8280), (8448, 8784)):
        support = pivot.iloc[start:stop].notna().mean()
        candidates.append({"indices": [start, stop], "start_utc": pivot.index[start].isoformat(),
            "last_utc": pivot.index[stop-1].isoformat(), "hours": stop-start,
            "stations_at_least_80pct": support[support >= .8].index.tolist(),
            "station_coverage": support.to_dict(), "overlap_known_prior_test": any(start < h["test_indices"][1] and stop > h["test_indices"][0] for h in historical),
            "status": "candidate-unscored-in-inventoried-artifacts-not-an-unseen-dataset"})
    archive = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope_roots": roots,
        "scope": "all listed source/config/report/result/document text, including historical source snapshots; binary data identified separately",
        "exclusions": ["v5 work created after registration", "dependency environments", "temporary test directories"],
        "unreadable": unreadable, "scanned_files": scanned, "matching_files": matches,
        "historical_epa": historical, "epa_dataset": artifact("data/external/epa/epa_bay_area_pm25_2024.parquet"),
        "epa_hours": len(pivot), "epa_stations": len(pivot.columns), "epa_candidate_windows": candidates,
        "beijing": {"status": "historically-exposed-no-new-pollution-holdout-claimed", "fit_fraction": .8,
                    "prior_index_tests": [[28057,28393],[34050,34722],[34728,35064]],
                    "seasonal_test_dates": ["2016-06-01", "2017-01-01"],
                    "source": artifact("configs/transfer/beijing_seasonal_v4.yaml")},
        "rules": ["Fresh channel seeds do not create independent pollution archives.",
                  "Both EPA windows must share fit/validation ending before index 7944 with purge; no fitting on first test before second.",
                  "Support inspection uses missingness only; archive has previously been downloaded and used."]}
    paths = ["results/fullpaper30h_20260902/results/e6_hardhat_local.json",
             "reports/v4_validation/sepolia_30_anchors_20260903/analysis.json",
             "reports/v4_validation/sepolia_30_anchors_20260903/anchors.json",
             "contracts/src/AirProofAnchor.sol", "docs/V4_REPRODUCTION_AND_EVIDENCE.md",
             "reports/v4_primary/core_final_7600_7629_20260903/analysis.json"]
    hardhat = json.loads((ROOT / paths[0]).read_text())
    anchors = json.loads((ROOT / paths[2]).read_text())
    reuse = {"created_utc": archive["created_utc"], "artifacts": [artifact(p) for p in paths],
        "hardhat": {"rows": len(hardhat["anchors"]), "expected": 30, "role": "historical-local-EVM-feasibility"},
        "sepolia": {"rows": len(anchors), "expected": 30, "all_success_readback": all(r["status"] == 1 and r["readback_verified"] and not r["reverted"] for r in anchors),
                    "unique_transactions": len({r["transaction_hash"] for r in anchors}), "time_blocks": len({r["time_block"] for r in anchors}),
                    "role": "historical-public-network-feasibility-not-v5-delay-or-finality"},
        "no_new_public_transactions": True, "v4_primary_relabelled_as_v5": False,
        "reuse_gate": "Evidence remains attached to its original mechanism, inputs, resource contract, target and clock. Changed integrated v5 needs new evidence.",
        "local_v5_audit": artifact("reports/v5/audit_same_stream/analysis.json")}
    for name, data in (("archive_usage.json", archive), ("reuse_manifest.json", reuse)):
        (ROOT / "configs/v5" / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"scanned_files": len(scanned), "matching_files": len(matches), "unreadable": unreadable,
                      "epa_windows": candidates, "hardhat": reuse["hardhat"], "sepolia": reuse["sepolia"]}, indent=2))


if __name__ == "__main__":
    main()
