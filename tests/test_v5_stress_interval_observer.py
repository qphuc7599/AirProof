import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from airproof.v5_estimator import EstimatorConfig
from airproof.v5_experiment import method_specs

PATH = Path(__file__).resolve().parents[1]/"reports/v5/tools/run_stress_with_intervals.py"
SPEC = importlib.util.spec_from_file_location("stress_observer", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_observer_preserves_returned_objects_and_single_calls():
    selected = EstimatorConfig(regularization_multiplier=.1)
    truth = np.arange(12, dtype=float).reshape(6, 2)+10
    public = np.full_like(truth, 12.)
    world = SimpleNamespace(seed=99, truth=truth, cell_groups=np.array([0, 1]),
                            metadata={"v5_stress": {"case": {"name": "fixture"}}})
    cfg = {"world": {"burn_in_steps": 2}}
    result_objects, returned_objects = [], []
    fake = SimpleNamespace()
    original_rows = [{"unchanged": True}]
    def estimate(*args, **kwargs):
        result = SimpleNamespace(live=np.vstack((public+1, public[:1])), reconstructed=np.vstack((public+2, public[:1])))
        result_objects.append(result)
        return result
    def score(world, public, operators, cfg, selected, event_mask):
        for spec in method_specs(selected, "severe_clean", factorial=False):
            if spec["estimator"] is not None:
                returned_objects.append(fake.estimate_public_field(public, None, [], spec["estimator"]))
        return original_rows
    fake.estimate_public_field, fake.score_case = estimate, score
    calibrator = {"pooled_radii": [2., 3.], "strata": {}}
    frozen = {"burn_in": 2, "calibrators": {f"{method}|{clock}": copy.deepcopy(calibrator)
              for method in ("AP", "SQ", "PUBLIC", "HUBER") for clock in ("live", "reconstructed")}}
    frozen_copy = copy.deepcopy(frozen)
    emitted = []
    observer = MODULE.StressObserver(fake, frozen, emitted.append)
    observer.install()
    result = fake.score_case(world, public, None, cfg, selected, np.zeros_like(truth, bool))
    assert result is original_rows
    assert observer.solver_calls == 3 and observer.cases == 1 and observer.rows == 8
    assert all(a is b for a, b in zip(result_objects, returned_objects))
    assert frozen == frozen_copy
    thresholds = [row["events"]["threshold"] for row in emitted]
    assert thresholds == [float(np.quantile(truth[:2], .95))]*8
    widths = [row["intervals"]["pooled"]["0.9"]["mean_width"] for row in emitted]
    emitted.clear()
    mutated = truth.copy(); mutated[2:] += 1000
    world.truth = mutated
    result2 = fake.score_case(world, public, None, cfg, selected, np.zeros_like(truth, bool))
    assert result2 is original_rows
    assert [row["events"]["threshold"] for row in emitted] == thresholds
    assert [row["intervals"]["pooled"]["0.9"]["mean_width"] for row in emitted] == widths
    assert frozen == frozen_copy
    observer.restore()
    assert fake.score_case is score and fake.estimate_public_field is estimate


def test_event_metrics_use_true_event_support_and_false_positives():
    result = MODULE.event_metrics(np.array([1., 9., 10., 1.]), np.array([9., 9., 1., 1.]), 5.)
    assert result["recall"] == .5 and result["false_positive_rate"] == .5
    assert result["true_positive"] == result["false_positive"] == 1


def test_huber_numerical_repair_preserves_objective_and_other_methods():
    from dataclasses import asdict, replace
    calls = []
    returned = SimpleNamespace(live=np.zeros((2, 1)), reconstructed=np.zeros((2, 1)))
    def estimate(*args, **kwargs):
        calls.append(args[3])
        return returned
    module = SimpleNamespace(score_case=lambda *args: None, estimate_public_field=estimate)
    observer = MODULE.StressObserver(module, {}, lambda row: None, huber_iterations=200)
    quadratic = EstimatorConfig()
    huber = replace(quadratic, loss="huber", input_clip=False, output_cap=False)
    observer.active = {"configs": {"AP": quadratic, "HUBER": huber}, "predictions": {}}
    assert observer.estimate(None, None, [], quadratic) is returned
    assert observer.estimate(None, None, [], huber) is returned
    assert calls[0] is quadratic
    assert asdict(calls[1]) == {**asdict(huber), "max_irls": 200}
    assert huber.max_irls == 20 and observer.solver_calls == 2
