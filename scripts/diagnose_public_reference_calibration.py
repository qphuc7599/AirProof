"""Select spatial reference interpolation using public burn-in only; score afterwards."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.interpolate import RBFInterpolator

from airproof.config import load_config
from airproof.simulator import generate_world


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/v3/primary_full_horizon.yaml")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7007
    world = generate_world(config, seed)
    side, burn = config["world"]["grid_side"], config["world"]["burn_in_steps"]
    coordinates = np.indices((side, side)).reshape(2, -1).T / (side - 1)
    cells = sorted({o.cell for o in world.reference_observations})
    locations = coordinates[cells]
    values = np.asarray([[o.value for o in world.reference_observations if o.epoch == t]
                         for t in range(config["world"]["steps"])])
    specifications = [{"kernel": "thin_plate_spline", "smoothing": 0.001}]
    specifications += [{"kernel": "gaussian", "epsilon": epsilon, "smoothing": 0.001,
                        "degree": 0} for epsilon in (1.0, 2.0, 3.0, 4.0, 6.0)]
    validation = []
    for specification in specifications:
        losses = []
        for held in range(len(cells)):
            keep = np.arange(len(cells)) != held
            predictions = RBFInterpolator(
                locations[keep], values[:burn, keep].T, **specification
            )(locations[held:held + 1])[0]
            losses.extend((predictions - values[:burn, held]) ** 2)
        validation.append(float(np.sqrt(np.mean(losses))))
    selected = int(np.argmin(validation))
    # The selection above has no access to synthetic truth or to post-burn-in targets.
    results = []
    for index in sorted({0, selected}):
        field = np.maximum(RBFInterpolator(locations, values.T, **specifications[index])(
            coordinates).T, 0.0)
        results.append({"specification": specifications[index], "selected": index == selected,
                        "public_validation_rmse": validation[index],
                        "post_burn_truth_rmse": float(np.sqrt(np.mean(
                            (field[burn:] - world.truth[burn:]) ** 2)))})
    report = {"seed": seed, "role": "public-reference backbone diagnostic only",
              "validation": list(zip(specifications, validation)), "results": results}
    output = root / f"reports/v4_validation/public_reference_calibration_{seed}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
