$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    python -m airproof.cli fixtures
    python -m pytest
    $matrices = @(
        'e1_primary',
        'e2_privacy',
        'e3_nominal',
        'e3_severe',
        'e4_attack',
        'e5_agents',
        'e5_grid'
    )
    foreach ($name in $matrices) {
        python -m airproof.cli suite --matrix "configs/matrices/$name.yaml" --output "reports/raw/$name.jsonl"
        $reference = if ($name -like 'e3_*') { 'factorial_111' } else { 'airproof' }
        python -m airproof.cli analyze "reports/raw/$name.jsonl" --reference $reference --output "reports/analysis/$name.json"
        python -m airproof.cli plot --input "reports/raw/$name.csv" --output "reports/analysis/$name.png"
    }
} finally {
    Pop-Location
}
