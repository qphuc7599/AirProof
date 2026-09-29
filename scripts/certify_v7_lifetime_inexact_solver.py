#!/usr/bin/env python3
"""Build the hash-bound M4 inexact-solver certificate without rerunning models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from airproof.v6_covariance_forcing_inputs import write_json
from airproof.v7_lifetime_inexact import build_inexact_solver_certificate

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    "reports/v7/reviewer_revision/lifetime_retained_diagnostic/"
    "inexact_solver_certificate.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_inexact_solver_certificate(ROOT)
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "certificate_sha256": result["certificate_sha256"],
                "maximum_projected_gradient_inf": result[
                    "inexact_solver_certificate"
                ]["maximum_projected_gradient_inf"],
                "per_solve_l2_error_upper": result["inexact_solver_certificate"][
                    "per_solve_l2_error_upper"
                ],
                "inexact_uniform_active_window_l2_bound": result[
                    "inexact_solver_certificate"
                ]["inexact_uniform_active_window_l2_bound"],
                "inexact_sum_active_window_l2_bound": result[
                    "inexact_solver_certificate"
                ]["inexact_sum_active_window_l2_bound"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
