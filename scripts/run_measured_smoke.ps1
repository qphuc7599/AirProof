$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    python -m airproof.cli fixtures
    python -m pytest
    $matrices = @('measured_pilot_huber', 'measured_frozen', 'measured_e2_privacy', 'measured_frozen_e3_nominal', 'measured_frozen_e3_severe', 'measured_frozen_e4_attack', 'measured_e5_agents', 'measured_e5_grid')
    foreach ($name in $matrices) {
        python -m airproof.cli suite --matrix "configs/matrices/$name.yaml" --output "reports/raw/$name.jsonl"
        $reference = if ($name -like '*e3_*') { 'factorial_111' } else { 'airproof' }
        python -m airproof.cli analyze "reports/raw/$name.jsonl" --reference $reference --output "reports/analysis/$name.json"
        python -m airproof.cli plot --input "reports/raw/$name.csv" --output "reports/analysis/$name.png"
    }
    python -m airproof.cli archived
    python -m airproof.cli plot-privacy --input reports/raw/measured_e2_privacy.csv --output reports/analysis/privacy_curve.png
    python -m airproof.cli plot-attack --input reports/raw/measured_frozen_e4_attack.csv --output reports/analysis/attack_curve.png
    python -m airproof.cli plot-scale --input reports/raw/measured_e5_agents.csv --x agents --output reports/analysis/scale_agents.png
    python -m airproof.cli plot-scale --input reports/raw/measured_e5_grid.csv --x grid_side --output reports/analysis/scale_grid.png
    python -m airproof.cli report
} finally {
    Pop-Location
}
