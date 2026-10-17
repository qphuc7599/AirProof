# AirProof: source and final experimental artifact

Research implementation of a citizen air-quality evidence pipeline with five pillars: robust digital twins, fairness-constrained allocation, delay-tolerant transport, whole-history protected releases, and blockchain-backed accountability.

This repository now includes the **final numerical evidence used in the manuscript**, alongside source code, experiment configuration, registered seeds, protocol/source snapshots, per-world measurements and analysis scripts. Tuning histories, abandoned runs and intermediate caches are excluded. Results with limitations that are actually discussed in the manuscript remain included with their original scope.

## Download the artifact

- [Final experimental artifact and large arrays](https://github.com/qphuc7599/AirProof/releases/tag/evidence-2026-10-17)
- [Complete file/checksum manifest](evidence/manifest.json)
- [Archive checksums](evidence/SHA256SUMS)
- [Evidence-to-paper map](evidence/README.md)

Source and compact numerical outputs are available in the Git tree. Larger saved prediction arrays are losslessly packaged in release ZIPs. Identical binary objects are stored once and mapped back to their original paths by the manifest. No experiment needs to be rerun to inspect the published measurements.

```sh
git clone https://github.com/qphuc7599/AirProof.git
cd AirProof
python scripts/restore_evidence.py --metadata-only
# Optional: download, restore and verify the complete saved arrays.
python scripts/restore_evidence.py --download
```

Restoration uses Python's standard library, verifies SHA-256 and refuses to overwrite a different local file. The full artifact is several GB; the compact source and numerical summaries can be used without downloading the large arrays. The archive contains no private keys, paper PDFs, figures, or externally downloaded raw datasets.

## Final evidence included

| Study | Numerical evidence |
|---|---|
| Primary accuracy, poisoning and fairness | 30 paired worlds, 360 runs, seven-test family; `reports/v4_primary/core_final_7600_7629_20260903/` |
| Privacy/cadence/suppression and allocation frontier | `reports/v4_validation/privacy_full_grid_7500_7529/`, `fairness_frontier_7400_7407/` |
| Beijing seasonal transfer, contemporary backbone and wrapper | `reports/v4_validation/beijing_seasonal_20260903/`, `esntnn_purpleair_7800/`, `esntnn_wrapper_7850_7861/` |
| Synthetic and measured-contact DTN | `reports/v4_validation/dtn_deadline_confirm_7950_7979/`, `empirical_contacts_8050_8054/` |
| Equal-information external wrapper confirmation | `reports/v8/external_wrapper_gate_c_v1/confirmation_v1/` |
| Lifetime component and inexact-solver certificate | `reports/v6/lifetime_budget7_validation_v2/outcomes/`, `reports/v7/reviewer_revision/lifetime_retained_diagnostic/` |
| Common-resource execution and matched controls | `reports/v7/reviewer_revision/shared_resource_core_confirmation_v3/`, `shared_core_component_controls_v1/`, `shared_core_fairness_control_v1/` |
| Processing-budget sensitivity | `reports/v8/reviewer_revision/fairness_resource_sweep_v1_workers8/` |
| Protected-release incremental value and reference stress | `reports/v8/m5_release_reference_stress_confirmation_v1/` |
| Event-conditioned uncertainty | `reports/v7/reviewer_revision/event_interval_confirmation_v1/` |
| Finite audit return, transparency and anchoring | `reports/v6/ledger_transparency_same_stream_v2/`, `durable_inclusion_async_return_v1/`, `durable_public_checkpoint_v2/`; `reports/submission_checks_20260927/anchoring/` |
| Computational scaling | `reports/submission_checks_20260927/scalability/`, `scalability_summary.json` |
| Public-network execution and finality | `reports/v4_validation/sepolia_30_anchors_20260903/`; `reports/submission_checks_20260927/sepolia_finality_20260927/` |

The primary study's seven Holm-adjusted tests passed. Its drift/hotspot attenuation was 32.66%/49.81%, and contributor-gap/worst-group RMSE reductions were 30.65%/12.91%. These effect sizes belong to that study; they must not be transferred to the common-resource study, whose fairness margins were not met. The finality campaign observed all 30 anchors finalized, with median/p95 observation latency 950.049/1,133.123 seconds and zero recorded reverts or RPC failures. The corrected quantile convention is in `verified_analysis.json`; the raw journal is preserved.

## Source layout and verification

- `airproof/`: generators, twin estimation, allocation, transport, privacy, release assimilation, uncertainty and audit.
- `scripts/`: experiment/analysis entry points and numerical checks. Historical implementation variants remain for dependencies and controls; stored final results are selected separately.
- `configs/`: mechanism, registration and seed configuration.
- `tests/`: unit, numerical and protocol tests.
- `contracts/`: Solidity anchor, Hardhat tests, local anchoring and opt-in Sepolia scripts.
- `retained_primary/` and the primary report's `source_snapshot/`: historical primary implementation, preserved separately from the current tree.

Python 3.11 or newer:

```sh
python -m pip install -e ".[dev]"
python -m airproof.cli --help
python -m pytest tests/test_v6_fairness.py tests/test_v8_release_assimilation.py tests/test_v8_joint_allocation.py -q
```

Optional predictor dependencies: `python -m pip install -e ".[modern-predictor]"`.
For Solidity, use a Node version supported by the pinned Hardhat release, then `cd contracts`, `npm ci`, and `npm test`.

Run scripts expose their configuration in source or `--help`. Analysis of saved outputs and regeneration from external datasets are different operations: external archives, some historical calibration inputs, author-supplied predictors and environments are not bundled. The final artifact supports inspection and checking of the retained results; it does not claim that every historical experiment can be regenerated from a fresh clone without additional inputs. Do not rewrite historical registrations or source hashes to match a newer implementation. Some recorded paths are historical and may need relocation when regenerating runs on another host.

Sepolia campaign scripts are opt-in, require locally supplied environment credentials, and spend test ETH. They are not invoked by installation or the commands above. Never commit keys or `.env` files.
