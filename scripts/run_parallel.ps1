$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -Path $repoRoot

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$jobCount = if ($env:JOB_COUNT) { [int]$env:JOB_COUNT } else { 3 }

if ($jobCount -lt 2 -or $jobCount -gt 4) {
    throw "JOB_COUNT must be between 2 and 4, got: $jobCount"
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$jobs = @()

for ($i = 0; $i -lt $jobCount; $i++) {
    $logDir = Join-Path $repoRoot ("log/parallel-gsm8k-{0}-{1}" -f $timestamp, $i)
    $args = @(
        '-m', 'src.run_experiment',
        '--dataset', 'gsm8k',
        '--method', 'cot_selfguide',
        '--num_samples', '1',
        '--mock_llm',
        '--summarize',
        '--log_dir', $logDir
    )

    $jobs += Start-Job -Name ("gsm8k-mock-{0}" -f $i) -WorkingDirectory $repoRoot -ScriptBlock {
        param($pythonExe, $pythonArgs)
        & $pythonExe @pythonArgs 2>&1
    } -ArgumentList $python, $args
}

$results = Wait-Job -Job $jobs | Receive-Job
$combined = ($results | ForEach-Object { $_.ToString() }) -join "`n"

if ($combined -match "No module named 'src'") {
    $jobs | Remove-Job -Force
    throw "Detected import failure: No module named 'src'"
}

$failed = $jobs | Where-Object { $_.State -ne 'Completed' }
if ($failed) {
    $failed | ForEach-Object { Write-Error ("Job failed: {0} ({1})" -f $_.Name, $_.State) }
    $jobs | Remove-Job -Force
    throw 'At least one job failed'
}

$results
$jobs | Remove-Job -Force