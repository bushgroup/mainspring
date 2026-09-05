<#
.SYNOPSIS
    Build the mainspring Windows executable and measure its startup time.
.DESCRIPTION
    Drives PyInstaller against packaging/mainspring.spec (dist/ and build/ land at the repo
    root regardless of caller cwd), then launches the built .exe twice -- timing from process
    start to the main window appearing -- so cold and warm startup are both on record. That
    number is what decides onefile vs onedir+Inno Setup (lab record, task 07).
.PARAMETER Mode
    onedir (default): a mainspring/ folder holding mainspring.exe beside its dependencies, no
    extraction, packaged by Inno Setup (packaging/mainspring.iss) -- task 07 measured onefile's
    cold start at 90-190+ s here against onedir's 2.3 s and picked onedir. onefile: a single
    mainspring.exe that extracts to a temp directory on every launch, kept for comparison.
.PARAMETER SkipBuild
    Measure startup against whatever is already in dist/ without rebuilding.
#>
param(
    [ValidateSet("onefile", "onedir")]
    [string]$Mode = "onedir",
    [switch]$SkipBuild,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$distDir = Join-Path $root "dist"
$buildDir = Join-Path $root "build"
$exePath = if ($Mode -eq "onedir") {
    Join-Path $distDir "mainspring\mainspring.exe"
} else {
    Join-Path $distDir "mainspring.exe"
}

if (-not $SkipBuild) {
    Write-Host "Warming the numba kernel cache for packaging/numba_cache_seed..." -ForegroundColor Cyan
    uv run tools/warm_numba_cache.py
    if ($LASTEXITCODE -ne 0) {
        throw "Warming the numba cache failed (exit $LASTEXITCODE)."
    }

    Write-Host "Building mainspring.exe ($Mode) with PyInstaller..." -ForegroundColor Cyan
    $env:MAINSPRING_PACKAGE_MODE = $Mode
    try {
        uv run pyinstaller packaging/mainspring.spec --distpath $distDir --workpath $buildDir --noconfirm --clean
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller build failed (exit $LASTEXITCODE)."
        }
    } finally {
        Remove-Item Env:\MAINSPRING_PACKAGE_MODE -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $exePath)) {
    throw "Expected build output at $exePath, but it does not exist. Run without -SkipBuild first."
}

if ($Mode -eq "onedir") {
    $size = (Get-ChildItem (Split-Path $exePath) -Recurse | Measure-Object -Property Length -Sum).Sum
    Write-Host ("mainspring/ folder size: {0:N1} MB" -f ($size / 1MB))
} else {
    $size = (Get-Item $exePath).Length
    Write-Host ("Executable size: {0:N1} MB" -f ($size / 1MB))
}

function Measure-Startup([string]$label) {
    $proc = Start-Process -FilePath $exePath -PassThru
    $start = Get-Date
    $deadline = $start.AddSeconds($TimeoutSeconds)
    while ((-not $proc.HasExited) -and ($proc.MainWindowHandle -eq [IntPtr]::Zero)) {
        Start-Sleep -Milliseconds 25
        $proc.Refresh()
        if ((Get-Date) -gt $deadline) {
            if (-not $proc.HasExited) { $proc | Stop-Process -Force }
            throw "$label startup: window never appeared within $TimeoutSeconds s."
        }
    }
    if ($proc.HasExited) {
        throw "$label startup: process exited before showing a window (exit $($proc.ExitCode))."
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("{0} startup: {1:N2} s" -f $label, $elapsed.TotalSeconds)
    $proc | Stop-Process -Force
    return $elapsed.TotalSeconds
}

# Onefile re-extracts to a fresh %TEMP%\_MEIxxxxx on every launch (no cross-run cache), so
# "cold" and "warm" differ there only in OS disk-cache state, not in whether extraction runs
# -- that extract-every-start cost is exactly what the onefile-vs-onedir decision is about.
# Onedir has nothing to extract; both numbers below should be close for it.
$cold = Measure-Startup "Cold"
Start-Sleep -Seconds 1
$warm = Measure-Startup "Warm"

Write-Host ""
Write-Host ("[$Mode] Cold {0:N2} s / Warm {1:N2} s" -f $cold, $warm) -ForegroundColor Green
Write-Host "Record these in mainspring-lab/tasks/07-packaging.md's progress log."
