# Final evidence inventory

This export is selected from the numerical studies reported in the current manuscript. It includes manuscript-reported limitations and unsuccessful contrasts; it does not select individual favorable seeds. Historical version numbers in paths identify protocols and do not make these one pooled experiment.

`manifest.json` maps every exported original file to its exact byte length and SHA-256. Files with `storage: git` are in this repository. Files with `storage: release` are restored from the four release ZIPs using `scripts/restore_evidence.py`. There are 366 binary paths and 306 unique objects; exact duplicate bytes are archived once. ZIP entries contain original bytes, without numerical conversion, truncation or precision changes.

The manifest covers 1,887 source/configuration/evidence files. Additional release documentation and the restore utility are versioned by Git. The four binary ZIPs total 3,115,277,353 bytes (about 2.90 GiB); compact Git-managed files total about 92.65 MiB before Git compression. The full 31 GB local research workspace is not published.

## Included study paths

| Path | Files | Original MiB |
|---|---:|---:|
| `reports/v4_primary/core_final_7600_7629_20260903` | 46 | 8.36 |
| `reports/v4_validation/fairness_frontier_7400_7407` | 7 | 3.67 |
| `reports/v4_validation/privacy_full_grid_7500_7529` | 8 | 38.71 |
| `reports/v4_validation/beijing_seasonal_20260903` | 19 | 4.03 |
| `reports/v4_validation/esntnn_purpleair_7800` | 6 | 154.70 |
| `reports/v4_validation/esntnn_wrapper_7850_7861` | 4 | 0.12 |
| `reports/v4_validation/dtn_deadline_confirm_7950_7979` | 3 | 0.14 |
| `reports/v4_validation/empirical_contacts_8050_8054` | 4 | 0.17 |
| `reports/v4_validation/ingestion_concurrency_20260903` | 1 | 0.00 |
| `reports/v4_validation/sepolia_30_anchors_20260903` | 5 | 0.05 |
| `reports/v5_diagnostics/wrapper_916000_916001` | 3 | 0.13 |
| `reports/v5_diagnostics/original_wrapper_tail_7850_7861.json` | 1 | 0.00 |
| `reports/v5/stress_export` | 9 | 1.09 |
| `reports/v6/lifetime_budget7_validation_v2/outcomes` | 545 | 2214.30 |
| `reports/v6/lifetime_budget7_validation_v2/inputs/global_calibration.json` | 1 | 0.01 |
| `reports/v6/lifetime_budget7_validation_v2/inputs/mechanism` | 1 | 0.01 |
| `reports/v6/causal_lifetime_exposure_validation_v1/outcomes/analysis.json` | 1 | 0.01 |
| `reports/v6/fairness_validation_v1` | 22 | 6.41 |
| `reports/v6/release_incremental_value` | 64 | 5.86 |
| `reports/v6/uncertainty_event_v3_1` | 4 | 0.03 |
| `reports/v6/accountability_integrated_confirmation` | 4 | 0.01 |
| `reports/v6/durable_inclusion_async_return_v1` | 3 | 0.00 |
| `reports/v6/durable_public_checkpoint_v2` | 3 | 0.00 |
| `reports/v6/ledger_transparency_same_stream_v2` | 4 | 5.70 |
| `reports/v7/reviewer_revision/lifetime_retained_diagnostic` | 8 | 1.69 |
| `reports/v7/reviewer_revision/lifetime_uniform_exposure_control_v1` | 51 | 221.41 |
| `reports/v7/reviewer_revision/shared_resource_confirmation_v2/analysis_detail.json` | 1 | 0.01 |
| `reports/v7/reviewer_revision/shared_resource_core_confirmation_v3` | 271 | 892.39 |
| `reports/v7/reviewer_revision/shared_core_fairness_control_v1` | 29 | 0.16 |
| `reports/v7/reviewer_revision/shared_core_component_controls_v1` | 29 | 0.06 |
| `reports/v7/reviewer_revision/event_interval_confirmation_v1` | 4 | 0.06 |
| `reports/v7/reviewer_revision/retained_modern_relay_comparison` | 1 | 0.01 |
| `reports/v8/m5_release_reference_stress_confirmation_v1` | 29 | 1.93 |
| `reports/v8/external_wrapper_gate_c_v1/confirmation_v1` | 3 | 11.78 |
| `reports/v8/development/public_checkpoint_smoke_v1` | 15 | 37.20 |
| `reports/v8/reviewer_revision/fairness_resource_sweep_v1_workers8` | 17 | 0.06 |
| `reports/submission_checks_20260927/scalability` | 12 | 0.33 |
| `reports/submission_checks_20260927/scalability_summary.json` | 1 | 0.00 |
| `reports/submission_checks_20260927/anchoring` | 4 | 2.08 |
| `reports/submission_checks_20260927/sepolia_finality_20260927` | 5 | 0.07 |
| `reports/submission_checks_20260927/numeric_checks.json` | 1 | 0.00 |
| `reports/submission_checks_20260927/retained_evidence_recheck.json` | 1 | 0.00 |
| `reports/raw/anchor_benchmark.csv` | 1 | 0.00 |
| `reports/raw/anchor_benchmark.jsonl` | 1 | 0.00 |
| `reports/raw/anchor_methods_benchmark.csv` | 1 | 0.00 |
| `reports/raw/anchor_methods_benchmark.jsonl` | 1 | 0.01 |
| `reports/raw/e6_receipt_faults.csv` | 1 | 0.04 |
| `reports/raw/e6_receipt_faults.json` | 1 | 0.11 |
| `reports/raw/q1_e6_audit_small.csv` | 1 | 0.00 |
| `reports/raw/q1_e6_audit_small.json` | 1 | 0.01 |
| `reports/raw/q1_e7_availability_small.csv` | 1 | 0.00 |
| `reports/raw/q1_e7_availability_small.json` | 1 | 0.01 |

The exact primary source snapshot is retained in the primary result directory. Other source-locked analyses retain their source-lock records; those historical locks must not be rewritten to claim that the current implementation is identical. External raw datasets, some large historical calibration inputs and learned backbone weights are not included. A saved-output audit and full regeneration are separate capabilities.

For the finality study use `verified_analysis.json` for linearly interpolated quantiles. `summary.json` is the unaltered original journal summary, which used floor-index order statistics. Signing-to-two-confirmation observation and consensus-finalized-head observation are distinct clocks.
