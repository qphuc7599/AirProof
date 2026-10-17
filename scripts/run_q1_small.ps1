$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    python -m airproof.cli fixtures
    python -m pytest -q
    python -m ruff check airproof tests

    python -m airproof.cli suite `
        --matrix configs/matrices/q1_pilot_small.yaml `
        --output reports/raw/q1_pilot_small.jsonl
    python -m airproof.cli analyze reports/raw/q1_pilot_small.jsonl `
        --reference airproof `
        --output reports/analysis/q1_pilot_small.json
    python -m airproof.cli power reports/raw/q1_pilot_small.jsonl `
        --metric worst_group_rmse `
        --reference airproof `
        --practical-delta 0.5 `
        --output reports/analysis/q1_power_worst_group_rmse.json
    python -m airproof.cli power reports/raw/q1_pilot_small.jsonl `
        --metric rmse `
        --reference airproof `
        --practical-delta 0.35 `
        --output reports/analysis/q1_power_rmse.json
    python -m airproof.cli power reports/raw/q1_pilot_small.jsonl `
        --metric restricted_mean_aoi `
        --reference airproof `
        --practical-delta 1.0 `
        --output reports/analysis/q1_power_aoi.json

    python -m airproof.cli suite `
        --matrix configs/matrices/q1_e2_private_small.yaml `
        --output reports/raw/q1_e2_private_small.jsonl
    python -m airproof.cli plot-privacy `
        --input reports/raw/q1_e2_private_small.csv `
        --output reports/analysis/q1_e2_private_small.png

    python -m airproof.cli suite `
        --matrix configs/matrices/q1_e2_longitudinal_small.yaml `
        --output reports/raw/q1_e2_longitudinal_small.jsonl

    python -m airproof.cli audit-benchmark `
        --output reports/raw/q1_e6_audit_small.json
    python -m airproof.cli availability-benchmark `
        --output reports/raw/q1_e7_availability_small.json

    Push-Location contracts
    try {
        npm test 2>&1 | Tee-Object -FilePath ../reports/q1_contract_test.log
    } finally {
        Pop-Location
    }
    python -m airproof.cli q1-report --output reports/Q1_SMALL_REPORT
} finally {
    Pop-Location
}
