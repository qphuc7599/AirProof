"""Fit a causal ESN--TNN on the authors' public 44-station archive.

Run with .venv-ml/Scripts/python.exe. No GPU or Lightning dependency is needed.
This is a declared one-step benchmark, not a reproduction of the paper's tables.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import rdata
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.esntnn import (DESNConfig, DeepEchoStateEnsemble, convex_weight,
                            make_transformer, station_windows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_tnn(model, data, esn, indices, *, history, mean, scale, batch_size=256):
    source, decoder = station_windows(data, esn, indices, history=history, mean=mean, scale=scale)
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(source), batch_size):
            values.append(model(torch.from_numpy(source[start : start + batch_size]),
                                torch.from_numpy(decoder[start : start + batch_size])).numpy())
    return np.maximum(0, np.concatenate(values).reshape(len(indices), data.shape[1]) * scale + mean)


def block_ratio_interval(candidate, reference, truth, *, block, draws=4000):
    # Resample full spatial vectors together, not stations/epochs as independent.
    a, b = np.mean((candidate - truth)**2, axis=1), np.mean((reference - truth)**2, axis=1)
    rng = np.random.default_rng(20260903 + block)
    samples = []
    for _ in range(draws):
        starts = rng.integers(len(a), size=int(np.ceil(len(a) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(a)).ravel()[:len(a)]
        samples.append(np.sqrt(a[indices].mean() / b[indices].mean()))
    return {"ratio": float(np.sqrt(a.mean() / b.mean())),
            "lower95": float(np.quantile(samples, .025)),
            "upper95": float(np.quantile(samples, .975)), "block_epochs": block,
            "draws": draws, "method": "circular time-block bootstrap of complete spatial vectors"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/external/baselines/esntnn_author/Data")
    parser.add_argument("--seed", type=int, default=7800)
    parser.add_argument("--ensemble", type=int, default=100)
    parser.add_argument("--units", type=int, default=500)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--history", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-steps", type=int, default=500)
    args = parser.parse_args()
    if min(args.steps, args.history, args.check_every, args.patience, args.minimum_steps) < 1 or args.lr <= 0:
        raise ValueError("positive training settings required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    data_path = args.data_dir / "APFour.RData"
    location_path = args.data_dir / "AirPollutionLocations.RData"
    data = np.asarray(rdata.read_rda(str(data_path))["newpoll"], dtype=float).T
    locations = np.asarray(rdata.read_rda(str(location_path))["pollution.locs"], dtype=float)
    if not np.isfinite(data).all() or np.any(data < 0) or locations.shape != (data.shape[1], 2):
        raise ValueError("archive does not satisfy the complete nonnegative data contract")
    total = len(data)
    esn_end, tnn_end, test_start = int(total * .5), int(total * .7), int(total * .85)
    if args.history >= esn_end or test_start - tnn_end < 20:
        raise ValueError("insufficient nested fitting/validation history")
    cfg = DESNConfig(units=args.units, ensemble=args.ensemble)
    manifest = {
        "role": "contemporary-public-predictor-one-step-adaptation",
        "source_paper": "https://doi.org/10.1093/jrsssc/qlaf007",
        "author_repository": "https://github.com/Env-an-Stat-group/25.Bonas.JRSSC",
        "author_commit": "6f05f241ffe175b26a4d081091c30bfd4cfb10a2",
        "data_sha256": sha(data_path), "locations_sha256": sha(location_path),
        "module_sha256": sha(ROOT / "airproof/esntnn.py"), "runner_sha256": sha(Path(__file__)),
        "stations": data.shape[1], "epochs": total, "source_interval_hours": 4,
        "time_axis_note": "ordered author archive has 912 epochs, while article describes 870; no timestamps supplied; not an exact table replication",
        "split_half_open": {"DESN_fit": [0, esn_end], "TNN_fit": [esn_end, tnn_end],
                            "model_selection": [tnn_end, test_start], "untouched_test": [test_start, total]},
        "forecast_input": "public archive observations up to target t-1, never target t",
        "PCA_fit": "DESN training prefix only, unlike author code's combined train/future PCA",
        "future_observations": "each is used as an input only after its acquisition time in rolling one-step evaluation",
        "DESN": asdict(cfg), "TNN": {"channels": 4, "heads": 2, "encoder_layers": 1,
                 "decoder_layers": 1, "feedforward": 16, "dropout": .1, "history": args.history,
                 "max_optimizer_steps": args.steps, "lr": args.lr, "batch_size": 64,
                 "optimizer": "Adam", "loss": "MSE in frozen training units",
                 "checkpoint_interval": args.check_every, "patience_checks": args.patience,
                 "minimum_steps": args.minimum_steps, "seed": args.seed},
        "adaptations": ["causal one-step horizon", "separate out-of-sample TNN fit prefix",
                        "training-only PCA", "bounded optimizer budget and early stopping",
                        "validation-only convex ensemble weight", "nonnegative outputs for every predictor"],
        "torch": torch.__version__, "python": sys.version, "device": "cpu", "threads": 1,
        "created_unix": time.time(),
    }
    dump(output / "manifest.json", manifest)

    def progress(done, maximum):
        if done % 10 == 0 or done == maximum:
            print(json.dumps({"event": "DESN_fit", "members": done, "total": maximum,
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)

    esn_model = DeepEchoStateEnsemble(cfg, seed=args.seed).fit(data[:esn_end], progress=progress)
    with (output / "desn_model.pkl").open("wb") as handle:
        pickle.dump(esn_model, handle, protocol=5)
    prefix = data[:test_start]
    esn_prefix = esn_model.predict_series(prefix)
    training_times = np.arange(esn_end, tnn_end)
    validation_times = np.arange(tnn_end, test_start)
    mean, scale = esn_model.mean, esn_model.scale
    source, decoder = station_windows(prefix, esn_prefix, training_times,
                                      history=args.history, mean=mean, scale=scale)
    targets = ((prefix[training_times] - mean) / scale).reshape(-1, 1, 1).astype(np.float32)
    source, decoder, targets = map(torch.from_numpy, (source, decoder, targets))
    model = make_transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    best_loss, best_step, best_state, stale = float("inf"), 0, None, 0
    log = []
    for step in range(1, args.steps + 1):
        indices = rng.integers(len(source), size=64)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source[indices], decoder[indices])
        loss = torch.mean((prediction - targets[indices])**2)
        if not torch.isfinite(loss):
            raise ArithmeticError("non-finite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % args.check_every == 0 or step == args.steps:
            validation_prediction = predict_tnn(model, prefix, esn_prefix, validation_times,
                history=args.history, mean=mean, scale=scale)
            validation_mse = float(np.mean((validation_prediction - prefix[validation_times])**2))
            scheduler.step(validation_mse)
            if validation_mse < best_loss:
                best_loss, best_step = validation_mse, step
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            entry = {"event": "TNN_validation", "step": step, "training_mse_scaled": float(loss.detach()),
                     "validation_mse": validation_mse, "best_step": best_step,
                     "elapsed_seconds": round(time.perf_counter() - started, 1)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
            dump(output / "training_log.json", log)
            if step >= args.minimum_steps and stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, output / "tnn_state.pt")
    validation_prediction = predict_tnn(model, prefix, esn_prefix, validation_times,
        history=args.history, mean=mean, scale=scale)
    weight = convex_weight(esn_prefix[validation_times], validation_prediction, prefix[validation_times])
    validation_mix = (1 - weight) * esn_prefix[validation_times] + weight * validation_prediction
    selection = {
        "selected_before_test_evaluation_unix": time.time(), "best_checkpoint_step": best_step,
        "completed_optimizer_steps": step, "tnn_convex_weight": weight,
        "validation_rmse": {"DESN": float(np.sqrt(np.mean((esn_prefix[validation_times] - prefix[validation_times])**2))),
                            "TNN": float(np.sqrt(best_loss)),
                            "ESN_TNN": float(np.sqrt(np.mean((validation_mix - prefix[validation_times])**2)))},
        "DESN_artifact_sha256": sha(output / "desn_model.pkl"),
        "TNN_artifact_sha256": sha(output / "tnn_state.pt"),
        "no_test_parameter_updates": True,
    }
    dump(output / "selection_locked.json", selection)

    # First evaluation of the held-out targets occurs only after the lock above.
    esn_all = esn_model.predict_series(data)
    test_times = np.arange(test_start, total)
    tnn_test = predict_tnn(model, data, esn_all, test_times, history=args.history, mean=mean, scale=scale)
    truth = data[test_times]
    predictions = {"persistence": data[test_times - 1], "DESN": esn_all[test_times],
                   "TNN": tnn_test, "ESN_TNN": (1 - weight) * esn_all[test_times] + weight * tnn_test}
    metrics = {name: {"rmse": float(np.sqrt(np.mean((pred - truth)**2))),
                      "mae": float(np.mean(np.abs(pred - truth)))} for name, pred in predictions.items()}
    np.savez_compressed(output / "heldout_predictions.npz", truth=truth, target_indices=test_times,
                        coordinates_lat_lon=locations, **predictions)
    summary = {"role": manifest["role"], "test_epochs": len(test_times), "stations": data.shape[1],
               "metrics": metrics, "selection": selection,
               "ESN_TNN_vs_persistence": [block_ratio_interval(predictions["ESN_TNN"], predictions["persistence"],
                                         truth, block=block) for block in (6, 12, 24)],
               "wrapper_evaluation": "not included in this predictor-only artifact",
               "elapsed_seconds": time.perf_counter() - started, "finished_unix": time.time()}
    dump(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "metrics": metrics, "tnn_weight": weight,
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
