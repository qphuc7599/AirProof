from pathlib import Path

from airproof.campaign import build_jobs
from airproof.config import load_config
from airproof.experiment import run_method, run_privacy_replay
from airproof.simulator import generate_world


def test_airproof24h_registered_counts(tmp_path: Path):
    jobs = build_jobs(Path("configs/airproof24h/campaign.yaml").resolve(), tmp_path)
    expected_rows = {
        "e1": 450,
        "e2a": 540,
        "e2l": 96,
        "e2_poison": 48,
        "e3": 192,
        "e4": 120,
        "e5": 40,
    }
    assert {
        study: sum(len(job.methods) * len(job.variants) for job in items)
        for study, items in jobs.items()
    } == expected_rows
    assert len(jobs["e2l"]) == 16


def test_fullpaper30h_registered_counts_and_scale_cases(tmp_path: Path):
    jobs = build_jobs(Path("configs/fullpaper30h/campaign.yaml").resolve(), tmp_path)
    expected_rows = {
        "e1": 270,
        "e2a": 96,
        "e2l": 64,
        "e2_poison": 32,
        "e3": 192,
        "e4": 72,
        "e5": 40,
    }
    assert {
        study: sum(len(job.methods) * len(job.variants) for job in items)
        for study, items in jobs.items()
    } == expected_rows
    assert len(jobs["e5"]) == 40
    assert len({job.job_id for job in jobs["e5"]}) == 40
    full_scale = next(
        job for job in jobs["e5"] if job.job_id.startswith("agents10000_grid64_full672")
    )
    overrides = full_scale.variants[0]["overrides"]
    assert overrides["world.agents"] == 10000
    assert overrides["world.grid_side"] == 64
    assert overrides["world.steps"] == 672


def test_privacy_replay_matches_end_to_end_release_layer():
    config = load_config("configs/smoke.yaml")
    world = generate_world(config, 77)
    full = run_method(world, config, "airproof")
    replay = run_privacy_replay(world, config, "airproof")
    keys = (
        "release_count",
        "suppressed_release_count",
        "release_rmse",
        "declined_privacy_users",
        "max_composed_user_epsilon",
        "privacy_accountant_events",
    )
    assert {key: full["metrics"][key] for key in keys} == {
        key: replay["metrics"][key] for key in keys
    }


def test_residual_release_mode_runs_with_public_fixed_replay_baseline():
    config = load_config("configs/q1_small.yaml")
    config["privacy"]["release_mode"] = "residual"
    config["privacy"]["residual_clip"] = 10.0
    config["privacy"]["public_baseline_value"] = config["world"]["baseline"]
    world = generate_world(config, 123)
    replay = run_privacy_replay(world, config, "airproof")
    assert replay["metrics"]["privacy_release_mode"] == "residual"
    assert replay["metrics"]["privacy_transcript_scope"].startswith("residual-")


def test_registered_outage_duration_changes_realized_run_lengths():
    short = load_config("configs/smoke.yaml")
    short["world"].update({"steps": 240, "agents": 80})
    short["network"].update(
        {"availability": 0.5, "outage_median_hours": 1, "outage_p95_hours": 2}
    )
    long = load_config("configs/smoke.yaml")
    long["world"].update({"steps": 240, "agents": 80})
    long["network"].update(
        {"availability": 0.5, "outage_median_hours": 12, "outage_p95_hours": 36}
    )
    short_world = generate_world(short, 91)
    long_world = generate_world(long, 91)
    assert (
        long_world.metadata["realized_off_p95_steps"]
        > short_world.metadata["realized_off_p95_steps"]
    )
