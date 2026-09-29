"""One resumable process pool; source-bound jobs and complete matrix accounting."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from airproof.config import load_config, config_hash, with_overrides
from airproof.experiment import _v4_public_components
from airproof.simulator import generate_world
from airproof.v5_estimator import EstimatorConfig
from airproof.v5_experiment import cell_configuration, method_specs, development_specs, evaluate_prepared, release_evidence
from airproof.v5_transport import generate_transport_trace, simulate_transport
from airproof.v5_evidence import build_run_evidence
from airproof.v5_evidence import attach_gateway_receipt
from airproof.v5_wire_payload import encode_observation_payload
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(canonical(value), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_inventory():
    paths = sorted([*ROOT.glob("airproof/*.py"), *ROOT.glob("scripts/*v5*.py"),
                    ROOT / "configs/v5/protocol.yaml", ROOT / "configs/v5/claim_families.yaml",ROOT / "configs/v5/runtime.json"])
    result = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return result, config_hash(result)


def fingerprint_records(records):
    # Transport policy cannot inspect payload, sigma, quality, or corruption flags.
    digest = hashlib.sha256()
    for r in records:
        digest.update(f"{r.user_id}|{r.epoch}|{r.cell}|{r.group}|{r.size_bytes}|{r.nullifier}\n".encode())
    return digest.hexdigest()


def audit_prediction(path, *, steps, cells, groups, lag, identity, source_hash, config_digest):
    warnings = []
    with np.load(path, allow_pickle=False) as archive:
        expected = {"live": (steps, cells), "reconstructed": (steps, cells),
                    "truth": (steps, cells), "counts": (steps+lag, groups), "cell_groups": (cells,)}
        for name, shape in expected.items():
            if name not in archive or archive[name].shape != shape:
                raise ValueError(f"{path}: missing/wrong shape {name}")
            value = archive[name]
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{path}: nonfinite/non-numeric {name}")
        if np.any(archive["counts"] < 0) or np.any(archive["counts"] != np.floor(archive["counts"])):
            raise ValueError(f"{path}: invalid contributor counts")
        if np.any(archive["cell_groups"] < 0) or np.any(archive["cell_groups"] >= groups):
            raise ValueError(f"{path}: invalid cell group domain")
        for key, expected_value in {"execution_identity": identity, "source_hash": source_hash, "config_hash": config_digest}.items():
            if key not in archive:
                warnings.append(f"legacy prediction missing embedded {key}")
            elif archive[key].shape != () or str(archive[key].item()) != expected_value:
                raise ValueError(f"{path}: embedded {key} mismatch")
    return warnings


def prediction_ready(path,cfg,identity,source_hash):
    if not path.exists(): return False
    audit_prediction(path,steps=cfg["world"]["steps"],cells=cfg["world"]["grid_side"]**2,groups=cfg["world"]["groups"],lag=cfg["twin"].get("fixed_lag",6),identity=identity,source_hash=source_hash,config_digest=config_hash(cfg))
    return True

def cached_value(directory, key, factory):
    """Trusted local cache; digest includes every declared dependency identity."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{key}.pkl.gz"; checksum=directory/f"{key}.sha256"
    if path.exists() and checksum.exists():
        with path.open("rb") as handle: digest=hashlib.file_digest(handle,"sha256").hexdigest()
        if digest != checksum.read_text(): raise ValueError(f"corrupted cache: {path}")
        with gzip.open(path,"rb") as handle: return pickle.load(handle)
    value=factory(); tmp=path.with_suffix(".tmp")
    with gzip.open(tmp,"wb",compresslevel=1) as handle: pickle.dump(value,handle,protocol=5)
    tmp.replace(path)
    with path.open("rb") as handle: digest=hashlib.file_digest(handle,"sha256").hexdigest()
    checksum.write_text(digest)
    return value


def run_world(job):
    started = time.perf_counter(); process = psutil.Process()
    output = Path(job["output"]); rows = []; peak = process.memory_info().rss
    transport_cache = {}; public_cache = {}; resumed = False
    for cell in job["cells"]:
        cfg = cell_configuration(job["base"], cell, job["overrides"])
        cfg["transport"] = job["protocol"]["transport"]
        cfg["scale"] = {**job["protocol"]["scale"], "acquisition_epochs": cfg["world"]["steps"]}
        cell_dir = output / "jobs" / str(job["seed"]) / cell
        specs = development_specs() if job["stage"] == "development" else method_specs(
            EstimatorConfig(**job["selected"]), cell, factorial=job["factorial"])
        identity = config_hash({"source": job["source_hash"], "config": cfg,
            "seed": job["seed"], "stage": job["stage"],
            "methods": [{**s, "estimator": asdict(s["estimator"]) if s["estimator"] else None} for s in specs]})
        existing = cell_dir / "outcomes.json"
        if existing.exists():
            old = json.loads(existing.read_text())
            if old["execution_identity"] != identity:
                raise ValueError(f"refuse resume mismatched identity: {existing}")
            if len(old["rows"]) != len(specs) or {r["method"] for r in old["rows"]} != {s["method"] for s in specs} or any(r.get("execution_identity") != identity for r in old["rows"]):
                raise ValueError("completed cell nested row identity/matrix mismatch")
            if not job["save_predictions"] or all(prediction_ready(cell_dir/f"{s['method']}.npz",cfg,identity,job["source_hash"]) for s in specs):
                resumed=True; rows.extend(old["rows"]); continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        cache_dir=output/"cache"/str(job["seed"])
        world_key=config_hash({"kind":"world","source":job["source_hash"],"config":cfg,"seed":job["seed"]})
        world = cached_value(cache_dir,world_key,lambda:generate_world(cfg,job["seed"]))
        pub_hash = hashlib.sha256(world.truth.tobytes() + repr(world.reference_observations).encode()).hexdigest()
        weather_hash=hashlib.sha256(pickle.dumps(world.public_meteorology,protocol=5)).hexdigest()
        public_key=config_hash({"kind":"public","data":pub_hash,"source":job["source_hash"],
            "twin":cfg["twin"],"baseline":cfg["world"].get("baseline",12),"weather":weather_hash})
        if public_key not in public_cache:
            public_cache[public_key] = cached_value(cache_dir,public_key,lambda:_v4_public_components(world,cfg))
        public, operators, public_diagnostics = public_cache[public_key]
        trace = generate_transport_trace(cfg, job["seed"])
        records_hash = fingerprint_records(world.observations)
        routes = {"airproof_deadline"} | ({"direct"} if any(not s["relay"] for s in specs) else set())
        transports = {}
        for policy in sorted(routes):
            key = config_hash({"trace": trace.trace_hash, "records": records_hash,
                "transport": cfg["transport"], "policy": policy, "source": job["source_hash"]})
            if key not in transport_cache:
                transport_cache[key] = cached_value(cache_dir,key,
                    lambda:simulate_transport(trace,world.observations,cfg,policy=policy))
            transports[policy] = transport_cache[key]
        release, release_arrays = release_evidence(world, public, transports["airproof_deadline"].release_arrivals, cfg,
            dense_calibration=job["stage"]=="calibration")
        np.savez_compressed(cell_dir / "release.npz", **release_arrays)
        write_json(cell_dir / "release.json", release)
        # One explicitly labeled protocol regression on the same accepted stream.
        # Issuance is real; neither all-message receipts nor their network return is asserted.
        accepted=transports["airproof_deadline"].raw_arrivals
        first=min((r for r in world.observations if r.nullifier in accepted),
            key=lambda r:(accepted[r.nullifier],r.nullifier),default=None)
        if first is not None and not (cell_dir/"receipt_interface_sample.json").exists():
            key=Ed25519PrivateKey.generate()
            attachment=attach_gateway_receipt(first,int(accepted[first.nullifier]),key,
                accepted_payload=encode_observation_payload(first))
            write_json(cell_dir/"receipt_interface_sample.json",{
                "scope":"one local interface regression; not receipt-network delivery or full cryptographic execution",
                "selection_rule":"earliest first gateway arrival, then nullifier",
                "gateway_public_key_hex":key.public_key().public_bytes_raw().hex(),
                "source_hash":job["source_hash"],"trace_hash":trace.trace_hash,"attachment":asdict(attachment)})
        selections = {}; cell_rows = []
        for spec in specs:
            row_path=cell_dir/f"{spec['method']}.json"
            if row_path.exists():
                previous=json.loads(row_path.read_text())
                if previous["execution_identity"]!=identity: raise ValueError(f"method resume identity mismatch:{row_path}")
                if not job["save_predictions"] or prediction_ready(cell_dir/f"{spec['method']}.npz",cfg,identity,job["source_hash"]):
                    resumed=True; cell_rows.append(previous); continue
            row, arrays = evaluate_prepared(world, public, operators, cfg, transports, spec,
                strata=job["stage"] != "development", selection_cache=selections)
            row.update(stage=job["stage"], cell=cell, execution_identity=identity,
                source_hash=job["source_hash"], trace_hash=trace.trace_hash,
                data_hash=pub_hash, record_transport_hash=records_hash)
            evidence=build_run_evidence(job["source_hash"],row["config_hash"],pub_hash,trace.trace_hash,
                spec["method"],job["seed"],"hourly frozen live and acquisition-aligned final reconstruction",
                canonical(row["metrics"]),{"feasible_floor":row["metrics"]["feasible_floor_violations"],
                    "solver_failure_rate":row["metrics"]["solver_failure_rate"],
                    "transport_tokens":row["transport"]["token_violations"]})
            row["run_evidence_id"]=evidence.evidence_id
            write_json(cell_dir/f"{spec['method']}.evidence.json",asdict(evidence))
            cell_rows.append(row)
            # Calibration/validation predictions are retained once per method.
            if job["save_predictions"]:
                np.savez_compressed(cell_dir / f"{spec['method']}.npz",
                    live=arrays["live"].astype(np.float32), reconstructed=arrays["reconstructed"].astype(np.float32),
                    counts=arrays["counts"], truth=world.truth.astype(np.float32), cell_groups=world.cell_groups,
                    execution_identity=np.array(identity), source_hash=np.array(job["source_hash"]), config_hash=np.array(config_hash(cfg)))
            write_json(row_path, row)
            peak = max(peak, process.memory_info().rss)
        write_json(existing, {"execution_identity": identity, "rows": cell_rows,
            "public_diagnostics": public_diagnostics, "release": release})
        rows.extend(cell_rows)
    original=output/"worlds"/f"{job['seed']}.json"
    if resumed and original.exists():
        original_result=json.loads(original.read_text())
        if original_result["rows"]==canonical(rows): return original_result
    return {"seed": job["seed"], "rows": rows, "elapsed_seconds": time.perf_counter()-started,
            "sampled_peak_rss_mb": peak / 2**20,"runtime_from_complete_fresh_world":not resumed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "development", "calibration", "validation", "confirmation", "stress"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--selected-config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    protocol = load_config(ROOT / "configs/v5/protocol.yaml")
    base = load_config(ROOT / protocol["historical_configuration"])
    if args.stage in ("validation", "confirmation", "calibration") and not args.selected_config:
        parser.error("selected configuration required after development")
    selected = asdict(EstimatorConfig())
    selected_manifest = None
    if args.selected_config:
        selected_manifest = load_config(args.selected_config)
        selected = selected_manifest.get("estimator", selected_manifest)
    if args.stage == "confirmation":
        if not selected_manifest.get("selection_passed") or not selected_manifest.get("confirmation_lock"):
            parser.error("confirmation requires passed selection and prospective lock")
        if args.smoke or not 30 <= len(args.seeds or []) <= 100:
            parser.error("confirmation requires 30..100 locked worlds at original scale")
        if args.seeds != selected_manifest["confirmation_lock"]["seeds"]:
            parser.error("confirmation seeds must match prospective lock exactly")
    low, high = protocol["seed_namespaces"][args.stage]
    seeds = args.seeds or list(range(low, high+1))
    if any(s < low or s > high for s in seeds) or len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique and within stage namespace")
    cells = args.cells or (list(protocol["primary"]["cells"]) if args.stage != "preflight" else ["severe_clean"])
    if args.stage == "confirmation" and cells != protocol["primary"]["cells"]:
        parser.error("cannot omit primary cells")
    overrides = ({"world.agents": 32, "world.grid_side": 4, "world.steps": 48,
        "world.burn_in_steps": 12, "world.regulatory_stations": 4,
        "twin.reference_calibration_epochs": 12, "attack.start_epoch": 24} if args.smoke else {})
    if args.smoke and args.stage != "preflight":
        parser.error("reduced scale only permitted in preflight")
    inventory, source_hash = source_inventory()
    if args.stage == "confirmation" and source_hash != selected_manifest["confirmation_lock"]["source_hash"]:
        parser.error("source differs from prospective confirmation lock")
    if args.stage == "confirmation" and (config_hash(protocol) != selected_manifest["confirmation_lock"].get("protocol_hash")
            or config_hash(base) != selected_manifest["confirmation_lock"].get("base_hash")):
        parser.error("protocol/base differs from prospective confirmation lock")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "controller.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error("controller lock exists; inspect process before removing stale lock")
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        manifest = {"stage": args.stage, "source_hash": source_hash, "source_files": inventory,
            "seeds": seeds, "cells": cells, "selected": selected, "overrides": overrides,
            "protocol_hash": config_hash(protocol), "base_hash": config_hash(base)}
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("campaign manifest changed; use separate output directory")
        write_json(manifest_path, manifest)
        snapshot = output / "source_snapshot"
        for name in inventory:
            dst = snapshot / name; dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(ROOT / name, dst)
        if args.workers < 1 or args.workers > 11:
            raise ValueError("worker count must be 1..11")
        if psutil.virtual_memory().available < 3 * 2**30:
            raise RuntimeError("less than 3 GiB available before campaign")
        jobs = [dict(output=str(output), seed=seed, cells=cells, source_hash=source_hash,
            protocol=protocol, base=base, overrides=overrides, stage=args.stage, selected=selected,
            factorial=args.stage in ("validation", "confirmation", "preflight"),
            save_predictions=args.save_predictions) for seed in seeds]
        completed = []; failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_world, job): job["seed"] for job in jobs}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result(); completed.append(result)
                    write_json(output / "worlds" / f"{seed}.json", result)
                    print(f"completed seed={seed} rows={len(result['rows'])} seconds={result['elapsed_seconds']:.1f} rss_mb={result['sampled_peak_rss_mb']:.0f}", flush=True)
                except Exception:
                    failure = {"seed": seed, "traceback": traceback.format_exc()}; failures.append(failure)
                    write_json(output / "failures" / f"{seed}.json", failure)
                    print(f"FAILED seed={seed}; retained failure artifact", flush=True)
                write_json(output / "progress.json", {"complete_seeds": sorted(r["seed"] for r in completed), "failures": failures})
        rows = [row for result in completed for row in result["rows"]]
        expected = sum(len(development_specs()) if args.stage == "development" else
            len(method_specs(EstimatorConfig(**selected), cell, factorial=jobs[0]["factorial"])) for cell in cells) * len(seeds)
        write_json(output / "outcomes.json", {"rows": rows, "expected_rows": expected,
            "complete": len(rows) == expected and not failures, "failures": failures})
        if failures: raise SystemExit(1)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
