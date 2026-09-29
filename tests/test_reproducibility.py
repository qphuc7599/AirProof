from airproof.config import config_hash, load_config
from airproof.experiment import run_experiment
from airproof.simulator import generate_world


def test_world_is_reproducible():
    config = load_config("configs/smoke.yaml")
    first = generate_world(config, 9)
    second = generate_world(config, 9)
    assert (first.truth == second.truth).all()
    assert first.observations == second.observations
    assert first.metadata == second.metadata


def test_run_ids_and_scientific_metrics_are_reproducible():
    config = load_config("configs/smoke.yaml")
    first = run_experiment(config, [2], ["airproof"])[0]
    second = run_experiment(config, [2], ["airproof"])[0]
    assert first["run_id"] == second["run_id"]
    assert first["config_hash"] == config_hash(config)
    nondeterministic = {
        "runtime_seconds",
        "peak_memory_mb",
        "python_peak_alloc_mb",
        "epoch_update_p50_seconds",
        "epoch_update_p95_seconds",
        "epoch_update_max_seconds",
    }
    for key, value in first["metrics"].items():
        if key not in nondeterministic:
            assert value == second["metrics"][key]


def test_burn_in_is_recorded_and_excluded_from_scoring_horizon():
    config = load_config("configs/smoke.yaml")
    config["world"]["burn_in_steps"] = 8
    result = run_experiment(config, [3], ["airproof"])[0]
    assert result["metrics"]["burn_in_steps"] == 8
    assert result["metrics"]["scoring_steps"] == config["world"]["steps"] - 8
