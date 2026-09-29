# AirProof experimental source

Source code for citizen air-quality sensing experiments: robust digital twins, fairness-constrained allocation, delay-tolerant transport, protected releases and blockchain-backed accountability.

This repository contains code, experiment configuration and tests only. Manuscripts, figures, datasets, trained weights, measured outputs and transaction journals are intentionally excluded. No published numerical result is bundled here.

## Components

- `airproof/`: simulation, bounded twin estimation, allocation, transport/resource accounting, privacy, release-only assimilation, uncertainty and audit protocols.
- `scripts/`: experiment runners, controls, validation and statistical analysis. Historical development variants are retained as code dependencies and controls; their presence does not mean every variant supports a manuscript claim.
- `configs/`: experiment settings and protocol configuration.
- `tests/`: numerical, resource, privacy and protocol checks.
- `contracts/`: Solidity anchor contract, Hardhat tests, local anchoring and Sepolia campaign/audit scripts.
- `retained_primary/`: saved primary source and base configuration, without outcomes. `scripts/submission_scalability.py` explicitly uses this snapshot.

## Install and verify

Python 3.11 or newer:

```sh
python -m pip install -e ".[dev]"
python -m airproof.cli --help
python -m pytest tests/test_v6_fairness.py -q
```

Optional learned predictors: `python -m pip install -e ".[modern-predictor]"`.

For contract development, use a Node.js version supported by the pinned Hardhat release:

```sh
cd contracts
npm ci
npm test
```

## Experiment entry points

- Primary paired-world experiments: `scripts/run_v4_core_campaign.py` and `scripts/analyze_v4_core_campaign.py`.
- Shared-resource integration: `scripts/run_v7_shared_resource_confirmation.py`.
- Fairness oracle: `scripts/verify_v6_fairness.py`.
- DTN and empirical contacts: `scripts/confirm_dtn_v4.py`, `scripts/benchmark_empirical_dtn.py`.
- Release assimilation: `scripts/run_v8_m5_release_confirmation.py`.
- Same-stream accountability: `scripts/run_v6_ledger_transparency_same_stream.py`.
- Computational scaling: `scripts/submission_scalability.py`.
- Local checkpoint anchoring: `scripts/submission_anchor_replay.py`, `contracts/scripts/submission-anchor-hourly.mjs`.
- Public-testnet finality: `contracts/scripts/sepolia-finality-campaign.mjs`; read-only verification: `contracts/scripts/audit-sepolia-finality.mjs`.

Inspect each entry point's arguments and configuration before execution. Many analysis and confirmation scripts require locally generated campaign outputs, registered inputs or external datasets, deliberately not shipped here. This source-only export is not a self-contained reproduction of all reported experiments. Source hashes in historical registrations describe the historical runs and must not be rewritten to imply the current tree is identical.

Sepolia scripts are opt-in and can spend test ETH. They read credentials from environment variables; never commit keys or `.env` files. Tests and installation do not launch a public-network campaign.
