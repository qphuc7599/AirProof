"""Run the paired local audit matrix; never sends a blockchain transaction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v5_audit import FAULTS, POLICIES, run_same_stream_audit, sha256_file, summarize_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v5/audit_same_stream")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise FileExistsError("preserve completed results; select a new output directory")
    started = time.perf_counter()
    manifest = {"version": "v5-audit-1", "batch_sizes": [32,128,512,2048], "trials": args.trials,
        "faults": FAULTS, "policies": POLICIES, "expected_rows": 4 * 8 * 3 * args.trials,
        "fault_epoch": 10, "observation_horizon": 60, "gossip_available_epoch": 18,
        "independent_anchor_read_epoch": 12, "receipt_deadline_epoch": 20,
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [ROOT/"airproof/v5_audit.py", ROOT/"airproof/audit.py", Path(__file__)]},
        "limitations": ["local immutable-anchor abstraction, not consensus implementation or finality measurement",
            "anchor advantage assumes honest checkpoint anchored before attack and an available independent read path",
            "co-partition of gossip and anchor would remove the measured partition advantage",
            "fault families are constructed scenarios; 100 payload/key trials are not 100 independent network worlds",
            "logical detection delays are specified by fault/communication schedule; CPU timing measured separately",
            "receipt/payload timeout is an availability or service-obligation violation, not a Merkle nonmembership proof",
            "signed logs also detect equivocation once conflicting signed checkpoints are exchanged",
            "no policy recovers unavailable raw payloads", "costs are actual canonical local protocol encodings, excluding IP/TLS/EVM wire overhead"]}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = run_same_stream_audit(trials=args.trials)
    with (args.output/"results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, allow_nan=False)+"\n")
    historical = {}
    for label, name in [("hardhat_local", "results/fullpaper30h_20260902/results/e6_hardhat_local.json"),
                        ("sepolia_analysis", "reports/v4_validation/sepolia_30_anchors_20260903/analysis.json")]:
        path = ROOT/name
        data = json.loads(path.read_text(encoding="utf-8"))
        historical[label] = {"path": name, "sha256": sha256_file(path), "same_workload": False,
            "role": "retained external feasibility evidence; not substituted for local benchmark cost or delay",
            "measured": data if label != "hardhat_local" else {k:v for k,v in data.items() if k != "anchors"},
            "anchor_rows": len(data.get("anchors", [])) if label == "hardhat_local" else None}
    summary = {"rows": len(rows), "complete": len(rows)==manifest["expected_rows"],
        "all_clean_controls_valid": all(r["clean_control_valid"] for r in rows),
        "paired_payload_invariant": all(len({r["stream_sha256"] for r in rows if r["batch_size"]==size and r["trial"]==trial})==1 for size in manifest["batch_sizes"] for trial in range(args.trials)),
        "groups": summarize_audit(rows), "historical_anchor_evidence": historical,
        "elapsed_seconds": time.perf_counter()-started,
        "results_sha256": sha256_file(args.output/"results.jsonl"), "limitations": manifest["limitations"]}
    (args.output/"analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["rows","complete","all_clean_controls_valid","paired_payload_invariant","elapsed_seconds"]}))


if __name__ == "__main__": main()
