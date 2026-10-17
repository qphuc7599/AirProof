param(
  [string]$Output = "../reports/submission_checks_20260927/sepolia_finality_20260927",
  [string]$RpcUrl = "https://ethereum-sepolia-rpc.publicnode.com"
)
$ErrorActionPreference = 'Stop'
$secret = $null
$plain = $null
while ([string]::IsNullOrWhiteSpace($plain)) {
  $secret = Read-Host 'Nhap private key cua vi Sepolia 0xc6d80...0023d (input bi an), roi nhan Enter' -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
  try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
  if ([string]::IsNullOrWhiteSpace($plain)) {
    Write-Host 'Khong nhan duoc input; vui long paste lai vao terminal.' -ForegroundColor Yellow
    $secret.Dispose()
  }
}
try {
  $env:SEPOLIA_RPC_URL = $RpcUrl
  $env:SEPOLIA_PRIVATE_KEY = $plain
  $env:SEPOLIA_OUTPUT = $Output
  node scripts/sepolia-finality-campaign.mjs
  if ($LASTEXITCODE -ne 0) { throw "Sepolia campaign failed with exit code $LASTEXITCODE" }
} finally {
  Remove-Item Env:SEPOLIA_RPC_URL -ErrorAction SilentlyContinue
  Remove-Item Env:SEPOLIA_PRIVATE_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:SEPOLIA_OUTPUT -ErrorAction SilentlyContinue
  $plain = $null
  if ($null -ne $secret) { $secret.Dispose() }
}
