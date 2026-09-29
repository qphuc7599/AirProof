"""Freeze eight-world residual radii, then evaluate separate integrated worlds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v5_uncertainty import _radii

METHODS = ("AP", "SQ", "PUBLIC", "HUBER")
CLOCKS = ("live", "reconstructed")
LEVELS = (.9, .95)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected_manifest(path):
    """Parse JSON with its native numeric semantics; YAML remains supported."""
    text = Path(path).read_text()
    return json.loads(text) if Path(path).suffix.lower() == '.json' else yaml.safe_load(text)


def identity(manifest):
    return {key: manifest[key] for key in ("source_hash", "protocol_hash", "base_hash", "selected", "overrides")}


def label_for(cell, group):
    return f"cell={cell}|group={group}"


def calibration_label(group):
    return f"group={group}"


def fit_radii(error_blocks, stratum_blocks, *, minimum=100):
    """Only calibration residuals enter this function; evaluation has no fit path."""
    pooled = np.concatenate([np.asarray(x, float).ravel() for x in error_blocks])
    if pooled.size < 20 or not np.isfinite(pooled).all():
        raise ValueError("at least20 finite calibration residuals required")
    strata = {}
    for label, blocks in sorted(stratum_blocks.items()):
        errors = np.concatenate([np.asarray(x, float).ravel() for x in blocks])
        strata[label] = {"n": int(errors.size), "fallback": errors.size < minimum,
                         "radii": None if errors.size < minimum else _radii(errors, LEVELS).tolist()}
    return {"levels": LEVELS, "minimum_stratum_support": minimum, "pooled_n": int(pooled.size),
            "pooled_radii": _radii(pooled, LEVELS).tolist(), "strata": strata}


def apply_radii(calibrator, predictions, cell, groups):
    prediction = np.asarray(predictions, float)
    radius = np.broadcast_to(calibrator["pooled_radii"], (*prediction.shape, 2)).copy()
    for group in np.unique(groups):
        entry = calibrator["strata"].get(calibration_label(group))
        if entry is not None and entry["radii"] is not None:
            radius[:, groups == group] = entry["radii"]
    return np.maximum(prediction[..., None]-radius, 0), prediction[..., None]+radius


def world_bootstrap_indices(world_count, draws=4000, seed=20260907):
    if world_count < 2 or draws < 1:
        raise ValueError("at least two worlds and one bootstrap draw required")
    return np.random.default_rng(seed).integers(world_count, size=(draws, world_count))


def bootstrap_summary(world_rows, world_seeds, indices):
    """Resample whole worlds with common draws; no row/time/station iid resampling."""
    ordered = {seed: i for i, seed in enumerate(world_seeds)}
    groups = {}
    for row in world_rows:
        key = (row["method"], row["clock"], row["stratum"], row["level"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        counts = np.zeros(len(world_seeds))
        totals = np.zeros((len(world_seeds), 3))
        for row in rows:
            i = ordered[row["seed"]]
            counts[i] += row["n"]
            totals[i] += np.array([row["coverage"], row["mean_width"], row["mean_interval_score"]])*row["n"]
        denominator = counts[indices].sum(1)
        sums = totals[indices].sum(1)
        estimates = np.divide(sums, denominator[:, None], out=np.full_like(sums, np.nan), where=denominator[:, None] > 0)
        point = totals.sum(0)/counts.sum()
        bounds = np.nanquantile(estimates, (.025, .975), axis=0)
        output.append({"method": key[0], "clock": key[1], "stratum": key[2], "level": key[3],
                       "n": int(counts.sum()), "supported_worlds": int(np.count_nonzero(counts)),
                       "metrics": {name: {"estimate": float(point[j]), "lower95": float(bounds[0, j]), "upper95": float(bounds[1, j])}
                                   for j, name in enumerate(("coverage", "mean_width", "mean_interval_score"))}})
    return output


def read_arrays(campaign, seed, cell, method, burn, expected_source):
    path = campaign/"jobs"/str(seed)/cell/f"{method}.npz"
    with np.load(path, allow_pickle=False) as archive:
        # Deliberately do not read counts or derive contexts from private arrivals.
        truth = np.asarray(archive["truth"][burn:], float)
        groups = np.asarray(archive["cell_groups"])
        predictions = {clock: np.asarray(archive[clock][burn:], float) for clock in CLOCKS}
    if truth.ndim != 2 or groups.shape != (truth.shape[1],) or truth.size == 0:
        raise ValueError("invalid retained prediction dimensions/burn")
    if not np.isfinite(truth).all() or any(p.shape != truth.shape or not np.isfinite(p).all() or np.any(p < 0) for p in predictions.values()):
        raise ValueError("finite truth and finite nonnegative matched predictions required")
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["seed"] != seed or metadata["method"] != method or metadata["cell"] != cell:
        raise ValueError("prediction metadata identity mismatch")
    if metadata.get("source_hash") != expected_source:
        raise ValueError("prediction source hash differs from campaign")
    return truth, groups, predictions, path, metadata


def freeze(campaign, selected_manifest, output, *, burn=48, training_manifest=None):
    manifest = json.loads((campaign/"manifest.json").read_text())
    selected = load_selected_manifest(selected_manifest)
    locked_estimator = selected.get("estimator", selected)
    if manifest["stage"] != "calibration" or len(set(manifest["seeds"])) != 8 or len(manifest["seeds"]) != 8:
        raise ValueError("exactly eight unique calibration worlds required")
    if manifest["selected"] != locked_estimator:
        raise ValueError("calibration estimator differs from supplied lock")
    if training_manifest is None:
        raise ValueError("training/development manifest required to audit disjoint calibration")
    training_seeds = json.loads(training_manifest.read_text())["seeds"]
    if set(training_seeds)&set(manifest["seeds"]):
        raise ValueError("training and calibration worlds overlap")
    output.mkdir(parents=True, exist_ok=False)
    calibrators, inputs = {}, {}
    # Process one method/clock at a time to bound working memory.
    for method in METHODS:
        for clock in CLOCKS:
            errors, strata = [], {}
            for seed in manifest["seeds"]:
                for cell in manifest["cells"]:
                    truth, groups, predictions, path, _ = read_arrays(campaign, seed, cell, method, burn, manifest["source_hash"])
                    error = predictions[clock]-truth
                    errors.append(error.ravel())
                    for group in np.unique(groups):
                        strata.setdefault(calibration_label(group), []).append(error[:, groups == group].ravel())
                    inputs[str(path.resolve())] = sha(path)
            calibrators[f"{method}|{clock}"] = fit_radii(errors, strata)
    frozen = {"calibrators": calibrators, "calibration_seeds": manifest["seeds"],
              "training_seeds": training_seeds, "burn_in": burn, "identity": identity(manifest),
              "selected_manifest_sha256": sha(selected_manifest), "campaign_manifest_sha256": sha(campaign/"manifest.json"),
              "training_manifest_sha256": None if training_manifest is None else sha(training_manifest),
              "input_npz_sha256": inputs, "source_sha256": sha(__file__),
              "public_strata": "fixed geographic group only; pooled across experimental cells; no attack label or private counts",
              "age_context": "unavailable as verified public causal input in current npz; not used",
              "claim": "frozen empirical residual quantiles; no exchangeability or posterior guarantee"}
    path = output/"frozen_quantiles.json"
    path.write_text(json.dumps(frozen, indent=2, allow_nan=False))
    (output/"freeze_identity.json").write_text(json.dumps({"frozen_quantiles_sha256": sha(path)}, indent=2))
    print(json.dumps({"event": "frozen", "calibration_worlds": 8, "quantile_sha256": sha(path)}))


def evaluate(campaign, frozen_path, output, *, draws=4000):
    frozen = json.loads(frozen_path.read_text())
    lock = json.loads((frozen_path.parent/"freeze_identity.json").read_text())
    if sha(frozen_path) != lock["frozen_quantiles_sha256"]:
        raise ValueError("frozen quantiles changed")
    manifest = json.loads((campaign/"manifest.json").read_text())
    if manifest["stage"] not in ("validation", "confirmation"):
        raise ValueError("only validation or confirmation evaluation is supported")
    if identity(manifest) != frozen["identity"]:
        raise ValueError("evaluation source/protocol/base/estimator identity differs from calibration")
    seeds = manifest["seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds)&set(frozen["calibration_seeds"]+frozen["training_seeds"]):
        raise ValueError("evaluation worlds duplicate or overlap calibration/training")
    indices = world_bootstrap_indices(len(seeds), draws)
    output.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], {}
    for seed in seeds:
        for cell in manifest["cells"]:
            for method in METHODS:
                truth, groups, predictions, path, metadata = read_arrays(campaign, seed, cell, method, frozen["burn_in"], manifest["source_hash"])
                inputs[str(path.resolve())] = sha(path)
                for clock in CLOCKS:
                    lower, upper = apply_radii(frozen["calibrators"][f"{method}|{clock}"], predictions[clock], cell, groups)
                    masks = {"pooled": np.ones(truth.shape, bool), f"cell={cell}": np.ones(truth.shape, bool)}
                    for group in np.unique(groups):
                        masks[label_for(cell, group)] = np.broadcast_to(groups == group, truth.shape)
                        masks[f"group={group}"] = np.broadcast_to(groups == group, truth.shape)
                    threshold = metadata.get("metrics", {}).get("event_threshold")
                    if threshold is not None:
                        masks[f"cell={cell}|event"] = truth >= threshold
                    for label, mask in masks.items():
                        target = truth[mask]
                        if not target.size:
                            continue
                        for index, level in enumerate(LEVELS):
                            low, high = lower[..., index][mask], upper[..., index][mask]
                            score = high-low+2/(1-level)*(np.maximum(low-target, 0)+np.maximum(target-high, 0))
                            rows.append({"seed": seed, "cell": cell, "method": method, "clock": clock,
                                         "stratum": label, "level": level, "n": target.size,
                                         "coverage": float(np.mean((low <= target)&(target <= high))),
                                         "mean_width": float(np.mean(high-low)), "mean_interval_score": float(np.mean(score))})
    summary = {"stage": manifest["stage"], "worlds": len(seeds), "draws": draws,
               "bootstrap_unit": "whole world; common draws across all cells, times, methods, clocks and groups",
               "bootstrap_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "frozen_quantiles_sha256": sha(frozen_path), "campaign_manifest_sha256": sha(campaign/"manifest.json"),
               "input_npz_sha256": inputs, "results": bootstrap_summary(rows, seeds, indices),
               "scope": "descriptive dependent-data interval performance; world bootstrap conditions on fitted calibration"}
    (output/"world_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"event": "evaluated", "worlds": len(seeds), "rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("freeze")
    fit.add_argument("--campaign", type=Path, required=True)
    fit.add_argument("--selected-manifest", type=Path, required=True)
    fit.add_argument("--training-manifest", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--burn-in", type=int, default=48)
    test = sub.add_parser("evaluate")
    test.add_argument("--campaign", type=Path, required=True)
    test.add_argument("--frozen", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.campaign, args.selected_manifest, args.output_dir, burn=args.burn_in, training_manifest=args.training_manifest)
    else:
        evaluate(args.campaign, args.frozen, args.output_dir, draws=args.draws)


if __name__ == "__main__":
    main()
