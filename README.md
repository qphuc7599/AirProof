# AirProof experimental artifact

This repository contains the implementation and final experimental evidence for AirProof: a citizen air-quality evidence pipeline combining **robust digital twins, fairness, delay-tolerant networking, whole-history differential privacy and blockchain accountability**.

The archived source and evidence are available in the [final experimental release](https://github.com/qphuc7599/AirProof/releases/tag/evidence-2026-10-17). Source, configurations and compact numerical outputs are stored directly in Git. Four release ZIPs hold the larger saved predictions and model objects, totaling **2.90 GiB**. The [manifest](evidence/manifest.json) binds **1,887 original files** to byte lengths and SHA-256 hashes; identical objects are stored once and restored to their original paths.

Only numerical studies reported in the manuscript are included. Paper drafts, figures, intermediate caches, abandoned runs and external raw datasets are omitted. Historical labels in paths identify the original protocols; different studies are not pooled.

## Controlled accuracy, robustness and fairness

The primary study comprises **30 independent paired worlds and 360 runs**, with the registered population, horizon and matched controls. All seven prespecified Holm-adjusted tests pass. Worlds are the independent experimental units; agents, cells and epochs do not increase that sample size.

| Endpoint | Final reported result | Versioned evidence |
|---|---|---|
| Clean accuracy | RMSE within 0.3% of the matched squared-loss twin | [Primary analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_primary/core_final_7600_7629_20260903/analysis.json) |
| Poisoning response | Drift/hotspot growth attenuation: 32.66% / 49.81% | [All run outcomes](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_primary/core_final_7600_7629_20260903/results.jsonl) |
| Fairness | Contributor-gap / worst-group RMSE reductions: 30.65% / 12.91% | [Paired analysis and hypothesis family](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_primary/core_final_7600_7629_20260903/analysis.json) |
| Privacy interactions | Utility, cadence and suppression under a fixed history budget | [Privacy-grid analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_validation/privacy_full_grid_7500_7529/analysis.json) |

The selected protected-release mechanism uses 28 scheduled query vectors, history budget eight and minimum cohort size 20. Its privacy guarantee applies to the declared release view under its stated admission and information assumptions.

## Common-resource execution and component controls

The shared-resource core uses **12 paired worlds** in severe-clean and hotspot conditions, giving 24 main runs. It charges contact, buffer, processing and control resources in the same execution. The four simultaneous efficacy criteria pass. Mean within-world hotspot attenuation is **52.00%**, with simultaneous lower bound **49.74%**.

The matched no-fairness contrast reduces contributor gap by **4.20%** and does not meet the original fairness margins. The primary study's larger equity effects therefore retain their separate 30-world scope. Component controls attribute empirical hotspot robustness mainly to the bounded Huber score; the output cap supplies the uniform displacement envelope.

| Evidence | Versioned files |
|---|---|
| Main outcomes, simultaneous bounds and execution checks | [Shared-core analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v7/reviewer_revision/shared_resource_core_confirmation_v3/analysis_detail.json) |
| Bounded-score, input-clipping and output-cap attribution | [Matched component controls](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v7/reviewer_revision/shared_core_component_controls_v1/analysis.json) |
| Matched no-fairness comparison | [Fairness control](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v7/reviewer_revision/shared_core_fairness_control_v1/analysis.json) |
| Processing-budget sensitivity | [Resource-sweep analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v8/reviewer_revision/fairness_resource_sweep_v1_workers8/analysis.json) |

## Protected-release value, intervals and external validation

Under the prespecified persistent reference-calibration stress, protected-pool RMSE is **2.715**, against **3.389** for public-only: a **19.89%** reduction. In the nominal-reference regime, public-only is better, at **0.865** versus **1.530**. The positive claim is limited to the declared reference-stress regime and publication clock.

Event-conditioned intervals are evaluated over **12 independent worlds** and 1,144,760 event-scoring points. Live event coverage is **95.50% / 98.06%** at nominal 90% / 95%, with whole-world bootstrap intervals. Widths are roughly 2.5 times those of marginal calibration. This is empirical event coverage under the declared synthetic process.

| Evidence | Versioned files |
|---|---|
| Release consumers, point estimates and paired bounds | [Release analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v8/m5_release_reference_stress_confirmation_v1/analysis.json) |
| Coverage, widths and interval scores | [Event-interval analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v7/reviewer_revision/event_interval_confirmation_v1/analysis.json) |
| Equal-information external wrapper | [Archive confirmation](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v8/external_wrapper_gate_c_v1/confirmation_v1/summary.json) |
| Lifetime replacement stability and numerical residuals | [Inexact-solver certificate](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v7/reviewer_revision/lifetime_retained_diagnostic/inexact_solver_certificate.json) |
| Seven-month Beijing station transfer | [Seasonal analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_validation/beijing_seasonal_20260903/analysis.json) |

## Transport, scaling and ledger accountability

Synthetic transport comprises **30 paired traces and 210 policy runs**. Relative to Spray-and-Wait, deadline-aware forwarding improves delivery by **2.32 percentage points** and reduces restricted delay by **22.84%**. Measured-contact replay retains a separate external-validation scope.

The finality campaign comprises **30 anchors in three time blocks**. All succeeded, passed readback and were observed finalized, with zero recorded RPC failures, reverts or right-censored anchors. Median/p95 observed finality is **950.049 / 1,133.123 seconds**. These client/RPC observation latencies describe one short Sepolia campaign.

| Evidence | Versioned files |
|---|---|
| Synthetic transport and paired comparisons | [Transport summary](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_validation/dtn_deadline_confirm_7950_7979/summary.json) |
| Measured-contact replay | [Replay summary](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v4_validation/empirical_contacts_8050_8054/summary.json) |
| Scaling to 10,000 agents and a separate 64 x 64 grid | [Scaling measurements](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/submission_checks_20260927/scalability_summary.json) |
| Same-stream audit/transparency comparison | [Accountability analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/v6/ledger_transparency_same_stream_v2/analysis.json) |
| Finite receipt return and local checkpoint anchoring | [Anchoring outputs](https://github.com/qphuc7599/AirProof/tree/evidence-2026-10-17/reports/submission_checks_20260927/anchoring) |
| Public-network finality, quantiles and chain verification | [Verified finality analysis](https://github.com/qphuc7599/AirProof/blob/evidence-2026-10-17/reports/submission_checks_20260927/sepolia_finality_20260927/verified_analysis.json) |

Cryptographic verification and timely retrieval are separate outcomes. Finite reverse-path delays remain in the evidence. Local-EVM execution, public-testnet finality and mainnet performance are different measurements.

## Repository layout

```text
airproof/                 Simulator, twin, allocation, transport, privacy and audit
configs/                  Mechanisms, registrations and seed configuration
contracts/                Solidity anchor, tests and local/Sepolia scripts
reports/                  Final numerical evidence used in the manuscript
retained_primary/         Historical primary source and base configuration
scripts/                  Experiment runners, analysis and artifact restoration
tests/                    Numerical, resource, privacy and protocol checks
evidence/manifest.json    Original paths, byte lengths, hashes and archive mapping
evidence/SHA256SUMS       Checksums for the four binary ZIP assets
EXPERIMENT_RUNBOOK.md      Setup, evidence checks and experiment entry points
pyproject.toml            Python dependencies and package configuration
```

Historical implementation variants remain as dependencies and controls; their presence does not mean every variant contributes a reported result.

## Requirements

- Python **3.11+**, with dependencies in [pyproject.toml](pyproject.toml).
- Optional PyTorch dependencies for learned-predictor paths.
- A Node.js version supported by the pinned Hardhat release for Solidity tests.
- Disk space for both the ZIPs and restored arrays when using the full binary artifact. Compact numerical summaries can be inspected separately.

The reported workstation used an Intel Core i5-12450HX, 16 GB RAM and Windows. Timings belong to that host and workload.

## Install and inspect the final evidence

```sh
git clone https://github.com/qphuc7599/AirProof.git
cd AirProof
python -m pip install -e ".[dev]"
python scripts/restore_evidence.py --metadata-only
```

Restore the complete saved arrays and verify checksums:

```sh
python scripts/restore_evidence.py --download
```

Run the targeted mechanism checks:

```sh
python -m pytest tests/test_v6_fairness.py tests/test_v8_release_assimilation.py tests/test_v8_joint_allocation.py -q
```

For contracts, run `npm ci` and `npm test` inside `contracts/`. See [EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md) for inspection commands, saved-output audits and experiment/analysis entry points, and the [evidence index](evidence/README.md) for the full path inventory.

## Reproduction scope

Saved-output inspection and regeneration have different prerequisites. External raw datasets and some historical calibration inputs are not bundled. Historical source locks, seed sets and protocol margins must retain their recorded identities; some historical paths may need relocation on another host.

Sepolia campaigns require locally supplied credentials and spend test ETH. Installation, restoration and the targeted checks above do not launch a public campaign. Credentials and `.env` files must remain local.
