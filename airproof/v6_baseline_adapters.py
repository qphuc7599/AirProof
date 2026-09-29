"""Versioned architecture adapters for matched v6 resource-contract runs.

These adapters define comparable information paths.  They are not reproductions
of the external systems that motivated the four architecture families.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from .records import Observation
from .v6_experiment import integrated_transport
from .v6_transport import TransportResult, TransportTrace, simulate_transport


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "v6" / "whole_system_baseline_adapters_v1.json"


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    family: str
    transport_policy: str
    raw_channel: bool
    protected_release_channel: bool
    accountability_return: bool
    prediction_input: str
    selection: str
    fidelity: str
    external_reproduction_claim_allowed: bool
    threat_model: tuple[str, ...]
    exclusions: tuple[str, ...]


@dataclass(frozen=True)
class AdapterRun:
    spec: AdapterSpec
    transport: TransportResult
    selected_nullifiers: tuple[str, ...]
    collector: object | None = None


def load_adapter_specs(path: str | Path = REGISTRY_PATH) -> Mapping[str, AdapterSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported baseline-adapter schema")
    common = payload.get("common_contract", {})
    required = {"transport_trace", "capacity", "clock", "packet_sizes", "denominators"}
    if not required.issubset(common):
        raise ValueError("baseline registry omits a common-contract field")
    specs = {}
    for adapter_id, item in payload["adapters"].items():
        if item.get("external_reproduction_claim_allowed") is not False:
            raise ValueError("v6 architecture adapters may not claim external reproduction")
        specs[adapter_id] = AdapterSpec(
            adapter_id=adapter_id,
            family=item["family"],
            transport_policy=item["transport_policy"],
            raw_channel=bool(item["raw_channel"]),
            protected_release_channel=bool(item["protected_release_channel"]),
            accountability_return=bool(item["accountability_return"]),
            prediction_input=item["prediction_input"],
            selection=item["selection"],
            fidelity=item["fidelity"],
            external_reproduction_claim_allowed=False,
            threat_model=tuple(item["threat_model"]),
            exclusions=tuple(item["exclusions"]),
        )
    return specs


def run_baseline_adapter(
    adapter_id: str,
    trace: TransportTrace,
    observations: Iterable[Observation],
    config: dict,
    *,
    specs: Mapping[str, AdapterSpec] | None = None,
) -> AdapterRun:
    """Run one adapter on a caller-supplied trace and immutable observations.

    Estimation is deliberately outside this function.  ``selected_nullifiers``
    is the only raw-evidence interface exposed to a downstream matched estimator;
    the privacy-only adapter exposes none.  For that adapter, each input record
    must represent an already-created protected release at its real source and
    acquisition slot; this function does not create or certify differential
    privacy.
    """
    registry = load_adapter_specs() if specs is None else specs
    if adapter_id not in registry:
        raise KeyError(f"unknown v6 baseline adapter: {adapter_id}")
    spec = registry[adapter_id]
    records = tuple(observations)
    if spec.accountability_return:
        result, collector = integrated_transport(
            trace, records, config, policy=spec.transport_policy, fairness=False
        )
        selected = tuple(r.nullifier for r in collector.selected)
        return AdapterRun(spec, result, selected, collector)
    result = simulate_transport(
        trace,
        records,
        config,
        spec.transport_policy,
        release=spec.protected_release_channel,
        raw=spec.raw_channel,
    )
    if spec.prediction_input == "timely_raw":
        lag = int(config.get("transport", {}).get("useful_lag_epochs", 6))
        selected = tuple(
            r.nullifier for r in records
            if r.nullifier in result.raw_arrivals
            and result.raw_arrivals[r.nullifier] <= r.epoch + lag
        )
    elif spec.prediction_input in {"protected_release_only", "public_only"}:
        selected = ()
    else:
        raise ValueError(f"unsupported prediction interface: {spec.prediction_input}")
    return AdapterRun(spec, result, selected)
