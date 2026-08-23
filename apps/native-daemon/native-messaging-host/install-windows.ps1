# Ssense Native Messaging Host - Windows Installer
# Run this from PowerShell after: cargo build --release
#
# Usage:
#   .\install-windows.ps1 -ExtensionId "abcdefghijklmnopabcdefghijklmnop"

param(
    [Parameter(Mandatory = $true)]
    [string]$ExtensionId
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script's location.
# NOTE: apps/native-daemon is a member of a Cargo *workspace* (see the root
# Cargo.toml), so `cargo build` outputs to the WORKSPACE ROOT's target/ dir,
# not apps/native-daemon/target/ — that's why we go up 3 levels here, not 1.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DaemonDir = Split-Path -Parent $ScriptDir
$WorkspaceRoot = Split-Path -Parent (Split-Path -Parent $DaemonDir)
$ExePath = Join-Path $WorkspaceRoot "target\release\ssense-native-daemon.exe"
$ManifestSrc = Join-Path $ScriptDir "com.ssense.daemon.json"
$ManifestDest = Join-Path $ScriptDir "com.ssense.daemon.installed.json"

if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: Release binary not found at:" -ForegroundColor Red
    Write-Host "  $ExePath" -ForegroundColor Red
    Write-Host "Run 'cargo build --release' in apps\native-daemon first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Daemon binary found: $ExePath" -ForegroundColor Green

# Build the final manifest with real paths substituted in
$manifestContent = Get-Content $ManifestSrc -Raw
$manifestContent = $manifestContent -replace "REPLACE_WITH_ABSOLUTE_EXE_PATH", ($ExePath -replace '\\', '\\')
$manifestContent = $manifestContent -replace "REPLACE_WITH_EXTENSION_ID", $ExtensionId
Set-Content -Path $ManifestDest -Value $manifestContent -Encoding UTF8

Write-Host "Manifest written: $ManifestDest" -ForegroundColor Green

# Register in the current user's registry (no admin rights required)
$RegPath = "HKCU:\Software\Google\Chrome\NativeMessagingHosts\com.ssense.daemon"
New-Item -Path $RegPath -Force | Out-Null
Set-ItemProperty -Path $RegPath -Name "(Default)" -Value $ManifestDest

Write-Host ""
Write-Host "Registered com.ssense.daemon for extension: $ExtensionId" -ForegroundColor Green
Write-Host "Restart Chrome completely (close all windows) for it to take effect." -ForegroundColor Yellow
