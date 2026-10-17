$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repo 'reports/environment.txt'
$lines = @()
$lines += "captured_utc=$([DateTime]::UtcNow.ToString('o'))"
$lines += "python=$(python --version 2>&1)"
$lines += "node=$(node --version 2>&1)"
$lines += "npm=$(npm --version 2>&1)"
$lines += "solc=$(solc --version 2>&1 | Select-Object -Last 1)"
$computer = Get-CimInstance Win32_ComputerSystem
$processor = Get-CimInstance Win32_Processor
$lines += "cpu=$($processor.Name)"
$lines += "cores=$($processor.NumberOfCores)"
$lines += "logical_processors=$($processor.NumberOfLogicalProcessors)"
$lines += "ram_bytes=$($computer.TotalPhysicalMemory)"
$lines += "pip_freeze_begin"
$lines += python -m pip freeze
$lines += "pip_freeze_end"
Set-Content -LiteralPath $output -Value $lines -Encoding utf8
Write-Output $output

