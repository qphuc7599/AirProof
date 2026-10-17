# AirProof experiment runbook

This guide covers the final numerical artifact in [release evidence-2026-10-17](https://github.com/qphuc7599/AirProof/releases/tag/evidence-2026-10-17). Documentation on `main` can improve without changing the archived measurements. Run commands from the repository root unless stated otherwise.

## 1. Environment

Use Python 3.11 or newer:

```sh
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell, then install:

```sh
python -m pip install -e ".[dev]"
python -m airproof.cli --help
```

The optional learned-predictor dependencies use `python -m pip install -e ".[modern-predictor]"`. Keep thread/worker settings consistent with the recorded protocol when comparing performance.

## 2. Compact evidence verification

```sh
python scripts/restore_evidence.py --metadata-only
```

This verifies **1,521** original source/configuration/compact-evidence files against recorded byte lengths and SHA-256. Additional documentation and restoration code are versioned by Git. Historical scientific files must retain their recorded bytes, including line endings.

Inspect the primary study without running simulations:

```sh
python -c "import json; from pathlib import Path; x=json.loads(Path('reports/v4_primary/core_final_7600_7629_20260903/analysis.json').read_text()); print(x['worlds'], x['rows'], x['all_primary_hypotheses_rejected'])"
```

Expected output: `30 360 True`. Individual hypotheses, outcomes and scope remain in the underlying files.

## 3. Complete array restoration

```sh
python scripts/restore_evidence.py --download
```

The utility downloads four ZIPs, verifies archive checksums, restores original paths and verifies **1,887** manifest files. It refuses to overwrite different local bytes. Identical objects can map to several original paths.

If the ZIPs are already downloaded:

```sh
python scripts/restore_evidence.py --assets-dir /path/to/downloaded/zips
```

On an already restored tree:

```sh
python scripts/restore_evidence.py --verify-only
```

The ZIPs total 3,115,277,353 bytes. Allow additional space for restored files. Large restored objects remain outside Git; compact arrays already tracked by Git remain tracked.

## 4. Mechanism checks

```sh
python -m pytest tests/test_v6_fairness.py tests/test_v8_release_assimilation.py tests/test_v8_joint_allocation.py -q
```

These **17 targeted tests** cover the indicated mechanisms; they do not rerun the primary campaign. For contract tests, use a Node.js version supported by the pinned Hardhat release and run inside `contracts/`:

```sh
npm ci
npm test
```

## 5. Audit retained numerical evidence

After full restoration:

```sh
python scripts/submission_evidence_audit.py
```

This checks 60 registered lifetime prediction arrays and manifests, recomputes the inexact-solver bound for the 630-step evaluation clock, and rechecks the protected-release/raw-DP paired comparison. It writes `reports/submission_checks_20260927/retained_evidence_recheck.json`; use a working copy if keeping every exported byte untouched is required. This audits stored evidence rather than regenerating historical runs from source-locked inputs.

For the finality study, inspect `reports/submission_checks_20260927/sepolia_finality_20260927/verified_analysis.json` and the original anchor/event journals. The saved audit records canonical receipts, contract readback and finalized-head checks for 30 transactions. Its interpolated quantiles differ from the original summary's floor-index order statistics. Two-confirmation observation and finalized-head observation are distinct clocks.

## 6. Main experiment and analysis entry points

The table identifies the implementations and protocols. Inspect source and registrations before execution; some source-locked campaigns require historical input bundles beyond the released final outputs.

| Study | Entry points | Protocol/evidence |
|---|---|---|
| Primary paired worlds | `scripts/run_v4_core_campaign.py`, `scripts/analyze_v4_core_campaign.py` | `configs/v4/core_campaign.yaml`; primary `protocol.yaml`, `manifest.json`, `source_snapshot/` |
| Privacy grid | `scripts/benchmark_privacy_grid_v4.py`, `scripts/analyze_privacy_grid_v4.py` | `reports/v4_validation/privacy_full_grid_7500_7529/` |
| Deadline DTN | `scripts/confirm_dtn_v4.py`, `scripts/benchmark_empirical_dtn.py` | Synthetic confirmation and empirical-contact manifests |
| Common resources | `scripts/run_v7_shared_resource_confirmation.py` | `configs/v7/`; final shared-core report |
| Component attribution | `scripts/run_v7_shared_core_component_controls.py` | `reports/v7/reviewer_revision/shared_core_component_controls_v1/` |
| Processing sensitivity | `scripts/run_v8_reviewer_fairness_resource_sweep.py` | Final resource-sweep report |
| Protected-release value | `scripts/run_v8_m5_release_confirmation.py` | `configs/v8/m5_release_reference_stress_confirmation_v1.json` |
| External wrapper | `scripts/run_v8_external_wrapper.py` | Final archive `confirmation_v1/` |
| Event intervals | `scripts/analyze_v7_event_interval_confirmation.py` | Final event-interval report |
| Scaling | `scripts/submission_scalability.py` | Retained primary source/configuration; `--agents`, `--side`, `--seed` |
| Audit comparison | `scripts/run_v6_ledger_transparency_same_stream.py` | Final same-stream registration and outcomes |
| Finite checkpoint anchoring | `scripts/submission_anchor_replay.py`, `contracts/scripts/submission-anchor-hourly.mjs` | `reports/submission_checks_20260927/anchoring/` |
| Public finality | `contracts/scripts/sepolia-finality-campaign.mjs` | Saved campaign manifest; local RPC/wallet configuration |

Argument inspection does not launch a campaign:

```sh
python scripts/run_v4_core_campaign.py --help
python scripts/run_v7_shared_resource_confirmation.py --help
python scripts/submission_scalability.py --help
```

Regeneration requires the recorded source, configuration, complete seed set and external inputs. Use a separate output directory. Do not overwrite archived results, rewrite historical source locks, or relabel an observed panel as a new independent confirmation.

## Interpretation notes

- The primary 30-world and common-resource 12-world studies have different contracts and inference; their effect sizes are not interchangeable.
- Nominal-reference utility, reference-calibration stress and archive corruption overlays are separate evaluations. Archived observations contain no observed poisoning labels.
- Conditional traffic replications share measured contact collections and do not represent independent city deployments.
- Local EVM execution, public-testnet finality and finite receipt-return delay measure different stages.
- A checksum establishes file identity. Full regeneration also requires the original inputs and environment. Citation identifiers and licensing from another project do not apply to this artifact.
