"""Recover exact-support public comparisons from historical DP aggregates only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "reports/v4_validation/privacy_full_grid_7500_7529"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_public_score(row, *, steps, groups, burn_in):
    """A subset of a finite domain equals it iff cardinality equals its size."""
    full_support = (steps - burn_in) * groups
    if row["scheduled_epochs"] != steps or row["release_count"] != full_support:
        return None
    if row["scheduled_group_opportunities"] != full_support:
        raise ValueError("inconsistent scheduled support")
    if row["release_clock"] != "acquisition_epoch_plus_fixed_collection_deadline":
        raise ValueError("unexpected historical clock")
    score = row["reference_group_rmse"]
    if score is None or not math.isfinite(score) or score < 0:
        raise ValueError("invalid public score")
    return score


def check_scoring_invariants():
    row = {"scheduled_epochs": 4, "release_count": 6, "scheduled_group_opportunities": 6,
           "release_clock": "acquisition_epoch_plus_fixed_collection_deadline", "reference_group_rmse": 2.0}
    assert matched_public_score(row, steps=4, groups=2, burn_in=1) == 2
    assert matched_public_score({**row, "release_count": 5}, steps=4, groups=2, burn_in=1) is None
    assert matched_public_score({**row, "scheduled_epochs": 2}, steps=4, groups=2, burn_in=1) is None
    try:
        matched_public_score({**row, "scheduled_group_opportunities": 5}, steps=4, groups=2, burn_in=1)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent domain accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v5/dp_public_reanalysis")
    args = parser.parse_args()
    check_scoring_invariants()
    import yaml
    cfg = yaml.safe_load((HISTORY / "base_configuration.yaml").read_text())
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    rows = [json.loads(line) for line in (HISTORY / "results.jsonl").read_text().splitlines()]
    if len(rows) != manifest["total_rows"]:
        raise ValueError("historical matrix incomplete")
    steps, groups, burn = (cfg["world"][k] for k in ("steps", "groups", "burn_in_steps"))
    output, seen, public_by_world = [], set(), {}
    group_rows = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("seed", "cell", "epsilon_user_max", "scheduled_epochs", "k_min", "private_eligibility", "release_mode"))
        if key in seen:
            raise ValueError("duplicate historical row")
        seen.add(key)
        world = (row["seed"], row["cell"])
        prior = public_by_world.setdefault(world, row["reference_group_rmse"])
        if prior != row["reference_group_rmse"]:
            raise ValueError("public aggregate changes across settings in same world")
        public = matched_public_score(row, steps=steps, groups=groups, burn_in=burn)
        recovered = public is not None
        result = {k: row[k] for k in ("seed", "cell", "epsilon_user_max", "scheduled_epochs", "k_min", "private_eligibility", "release_mode", "release_count", "release_rmse", "config_hash", "source_hash")}
        result.update({"exact_matched_public_rmse": public,
            "dp_minus_public_rmse": row["release_rmse"] - public if recovered else None,
            "matched_support_recoverable": recovered,
            "reason": "entire-post-burn-in-time-group-domain" if recovered else "per-epoch-public-errors-and-emission-mask-not-persisted",
            "public_scoring_clock": "reference-at-acquisition-epoch-delayed-to-same-plus-24-closure" if recovered else None,
            "public_newer_at_closure_used": False,
            "mechanism": "historical-v4-reserved-logical-release-not-integrated-v5"})
        output.append(result)
        if recovered:
            group_rows[(row["cell"], row["epsilon_user_max"], row["k_min"], row["private_eligibility"], row["release_mode"])].append(result)
    summaries = []
    for key, items in sorted(group_rows.items()):
        summaries.append(dict(zip(("cell", "epsilon_user_max", "k_min", "private_eligibility", "release_mode"), key)) | {
            "worlds": len(items), "mean_dp_rmse": statistics.mean(r["release_rmse"] for r in items),
            "mean_public_rmse": statistics.mean(r["exact_matched_public_rmse"] for r in items),
            "mean_paired_rmse_difference": statistics.mean(r["dp_minus_public_rmse"] for r in items),
            "worlds_dp_better": sum(r["dp_minus_public_rmse"] < 0 for r in items),
            "scope": "descriptive-historical-reanalysis-no-new-confirmation"})
    artifacts = [HISTORY / p for p in ("results.jsonl", "worlds.jsonl", "manifest.json", "base_configuration.yaml", "protocol.yaml", "runner.py", "source_snapshot/airproof/privacy_grid.py", "source_snapshot/airproof/continual_release.py")]
    provenance = {"historical_source_hash": manifest["source_hash"], "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
        "script_sha256": digest(Path(__file__)), "reran_grid": False, "reran_world_or_backbone": False,
        "available_artifacts": sorted(str(p.relative_to(HISTORY)) for p in HISTORY.iterdir() if p.is_file()),
        "saved_time_group_predictions_or_statistics": False,
        "support_proof": "Historical score mask is an emitted subset of the post-burn-in domain. Full cadence plus release_count=(steps-burn)*groups proves equality. Saved all-domain reference RMSE is then exactly matched.",
        "clock": "Public estimate for acquisition t is held until t+24, matching the DP release closure; no t+24 public observation is substituted.",
        "sparse_or_suppressed_warning": "All-epoch reference_group_rmse is not matched support for these rows and is not used as such.",
        "historical_prespecified_claim": "epsilon8, cadence28, k20, severe_clean, no gate is unavailable for matched-public scoring without regenerating unsaved arrays"}
    summary = {"historical_rows": len(rows), "historical_physical_worlds": len(public_by_world),
        "recoverable_rows": sum(r["matched_support_recoverable"] for r in output),
        "unrecoverable_rows": sum(not r["matched_support_recoverable"] for r in output),
        "scoring_invariants_pass": True, "groups": summaries, "provenance": provenance}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in output))
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: summary[k] for k in ("historical_rows", "historical_physical_worlds", "recoverable_rows", "unrecoverable_rows", "scoring_invariants_pass")}))
    print(json.dumps([r for r in summaries if r["epsilon_user_max"] == 8 and r["k_min"] == 20 and not r["private_eligibility"]], indent=2))


if __name__ == "__main__":
    main()
