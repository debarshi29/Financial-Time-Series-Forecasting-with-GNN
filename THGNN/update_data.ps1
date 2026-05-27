# update_data.ps1 — Incremental data update for THGNN (Windows / PowerShell)
#
# Usage:
#   .\update_data.ps1                              # uses defaults below
#   .\update_data.ps1 -StartDate 2026-04-08        # override start date
#   .\update_data.ps1 -StartDate 2026-04-08 -Processes 8
#
#   Run in background (detached, log still written):
#   Start-Process powershell -ArgumentList "-NoProfile -File `"$PWD\update_data.ps1`"" -WindowStyle Hidden
#
# Progress is written to logs\update_YYYYMMDD_HHMMSS.log via Start-Transcript.
# $ErrorActionPreference = Stop means any failed python call throws and stops the script.

[CmdletBinding()]
param(
    [string] $StartDate = "2026-04-08",
    [int]    $Processes = 4,
    [int]    $Window    = 20,
    [int]    $Horizon   = 3
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Config — edit these if your paths differ
# ---------------------------------------------------------------------------
$ScriptDir   = $PSScriptRoot
$DataDir     = Join-Path $ScriptDir "data"
$TickerFile  = Join-Path $DataDir "valid_nifty500.txt"
$MainPkl     = Join-Path $DataDir "nifty500.pkl"
$NewPkl      = Join-Path $DataDir "nifty500_new.pkl"
$RelationDir = Join-Path $DataDir "relation_nifty500"
$GraphDir    = Join-Path $DataDir "data_train_predict"

# ---------------------------------------------------------------------------
# Logging setup  (Start-Transcript mirrors everything to the log file)
# ---------------------------------------------------------------------------
$LogDir  = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogFile = Join-Path $LogDir "update_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Start-Transcript -Path $LogFile -Append | Out-Null

# Helper functions ── mirrors log() / step_start() / step_done() from bash
function Write-Log {
    param([string]$Message)
    Write-Host "[$( (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') )] $Message"
}

$script:StepT0 = $null

function Step-Start {
    param([string]$Num, [string]$Desc)
    $script:StepT0 = Get-Date
    Write-Log "=== STEP $Num : $Desc ==="
}

function Step-Done {
    param([string]$Num)
    $elapsed = [int]((Get-Date) - $script:StepT0).TotalSeconds
    Write-Log "=== STEP $Num done in ${elapsed}s ==="
    Write-Host ""
}

# Helper — run python and throw a clear error if it exits non-zero
# NOTE: $Args is a reserved PowerShell automatic variable; use $PyArgs instead.
function Invoke-Python {
    param([string[]]$PyArgs)
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE.  Command: python $($PyArgs -join ' ')"
    }
}

# Feature-engineering burn-in: mom20 / rsi14 / vol20 each drop up to 20 rows per
# stock via dropna().  Downloading from exactly $StartDate means the first ~20
# trading days (~28 calendar days) have no valid features, creating a gap that
# breaks the relation-matrix window check (requires EXACTLY 20 aligned days).
# Fix: download from 45 calendar days before $StartDate; the merge step deduplicates
# so historical data is never re-added, but the burn-in rows are present.
$DownloadStart = (Get-Date $StartDate).AddDays(-45).ToString("yyyy-MM-dd")

Write-Log "Starting incremental update | start_date=$StartDate | download_start=$DownloadStart"
Write-Log "Log file : $LogFile"
Write-Log "Working directory : $ScriptDir"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1 — Download new price data
# ---------------------------------------------------------------------------
Step-Start "1" "Download price data from $DownloadStart (burn-in offset from $StartDate)"

$TickerLines  = Get-Content $TickerFile
$TickerCount  = $TickerLines.Count
$Tickers      = $TickerLines -join ','
Write-Log "Tickers: $TickerCount  (from $TickerFile)"

Invoke-Python @(
    "$ScriptDir\utils\download_market_data.py",
    "--tickers", $Tickers,
    "--start",   $DownloadStart,
    "--output",  $NewPkl
)

Write-Log "Downloaded to $NewPkl"
Step-Done "1"

# ---------------------------------------------------------------------------
# Step 1b — Merge new data into existing pickle
# ---------------------------------------------------------------------------
Step-Start "1b" "Merge $NewPkl into $MainPkl"

if (-not (Test-Path $MainPkl)) {
    Write-Log "No existing pickle found at $MainPkl - using new download directly."
    Move-Item -Force $NewPkl $MainPkl
} else {
    # Write the merge logic to a temp file so we can pass Windows paths safely.
    # NOTE: this is a double-quoted here-string — $MainPkl and $NewPkl are
    # expanded by PowerShell before Python sees them.
    $MergeScript = @"
import pickle, pandas as pd

old_path = r'$MainPkl'
new_path = r'$NewPkl'

with open(old_path, 'rb') as f:
    old = pickle.load(f)
with open(new_path, 'rb') as f:
    new = pickle.load(f)

old = pd.DataFrame(old)
new = pd.DataFrame(new)
old['dt'] = pd.to_datetime(old['dt'])
new['dt'] = pd.to_datetime(new['dt'])

before = len(old)
merged = (
    pd.concat([old, new])
    .drop_duplicates(subset=['dt', 'code'])
    .sort_values(['dt', 'code'])
    .reset_index(drop=True)
)
added = len(merged) - before

with open(old_path, 'wb') as f:
    pickle.dump(merged, f)

print(f'  Rows before : {before:,}')
print(f'  Rows after  : {len(merged):,}  (+{added:,} new rows)')
print(f"  Date range  : {merged['dt'].min().date()} -> {merged['dt'].max().date()}")
print(f"  Stocks      : {merged['code'].nunique()}")
"@

    $TmpPy = Join-Path $env:TEMP "thgnn_merge_$(Get-Date -Format 'yyyyMMdd_HHmmssff').py"
    $MergeScript | Out-File -FilePath $TmpPy -Encoding utf8

    try {
        Invoke-Python @($TmpPy)
    } finally {
        Remove-Item -Force $TmpPy -ErrorAction SilentlyContinue
    }

    Remove-Item -Force $NewPkl -ErrorAction SilentlyContinue
    Write-Log "Merged. Removed temp file $NewPkl"
}

Step-Done "1b"

# ---------------------------------------------------------------------------
# Step 2 — Generate relation matrices for new dates only
# ---------------------------------------------------------------------------
Step-Start "2" "Generate relation matrices (start-date=$StartDate, processes=$Processes)"

New-Item -ItemType Directory -Force $RelationDir | Out-Null
$ExistingRelations = (Get-ChildItem "$RelationDir\*.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Log "Existing relation files: $ExistingRelations"

Invoke-Python @(
    "$ScriptDir\utils\generate_relation.py",
    "--data-path",    $MainPkl,
    "--relation-dir", $RelationDir,
    "--window",       $Window,
    "--start-date",   $StartDate,
    "--processes",    $Processes
)

$NewRelations    = (Get-ChildItem "$RelationDir\*.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
$AddedRelations  = $NewRelations - $ExistingRelations
Write-Log "Relation files now: $NewRelations  (added $AddedRelations)"
Step-Done "2"

# ---------------------------------------------------------------------------
# Step 3 — Generate graph samples for new dates only
# ---------------------------------------------------------------------------
Step-Start "3" "Generate graph samples (start-date=$StartDate)"

New-Item -ItemType Directory -Force $GraphDir | Out-Null
$ExistingGraphs = (Get-ChildItem "$GraphDir\*.pkl" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Log "Existing graph samples: $ExistingGraphs"

Invoke-Python @(
    "$ScriptDir\utils\generate_data.py",
    "--data-path",    $MainPkl,
    "--relation-dir", $RelationDir,
    "--output-dir",   $GraphDir,
    "--start-date",   $StartDate,
    "--window",       $Window,
    "--horizon",      $Horizon
)

$NewGraphs   = (Get-ChildItem "$GraphDir\*.pkl" -ErrorAction SilentlyContinue | Measure-Object).Count
$AddedGraphs = $NewGraphs - $ExistingGraphs
Write-Log "Graph samples now: $NewGraphs  (added $AddedGraphs)"
Step-Done "3"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Log "All steps completed successfully."
Write-Log "Main pickle : $MainPkl"
Write-Log "Relations   : $RelationDir  ($NewRelations files)"
Write-Log "Graph data  : $GraphDir  ($NewGraphs files)"
Write-Log "Full log    : $LogFile"

Stop-Transcript | Out-Null
