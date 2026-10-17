param(
    [string]$OutputRoot = "D:\DoAnChuyenNganh\airprooff\results\campaign_20260902",
    [int]$Workers = 12,
    [string]$Studies = ""
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

$arguments = @(
    "-m", "airproof.campaign",
    "--config", "configs/airproof24h/campaign.yaml",
    "--output", $OutputRoot,
    "--workers", $Workers
)
if ($Studies) {
    $arguments += @("--studies", $Studies)
}
python @arguments
