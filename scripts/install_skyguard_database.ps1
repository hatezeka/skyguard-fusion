# Installs only additive files. Does NOT edit existing files, commit, push or merge.
[CmdletBinding()]
param(
    [string]$Repository = 'C:\Users\nistr\skyguard-fusion'
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Repository -PathType Container)) {
    throw "SkyGuard repository not found: $Repository"
}
$target = (Resolve-Path -LiteralPath $Repository).Path
if (-not (Test-Path (Join-Path $target '.git'))) { throw "Not a repository: $target" }
$branch = (& git -C $target branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-mvp') {
    throw "Wrong branch: '$branch'. Expected feature/skyguard-mvp"
}
$files = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force | Where-Object {
    $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]' -and
    $_.Extension -notin @('.pyc', '.sqlite3', '.db')
})
$manifest = @()
foreach ($file in $files) {
    $rel = $file.FullName.Substring($source.TrimEnd('\').Length).TrimStart('\', '/')
    $dest = Join-Path $target $rel
    if (Test-Path -LiteralPath $dest) {
        throw "Stopped: existing file will not be overwritten: $dest"
    }
    $manifest += [pscustomobject]@{Source=$file.FullName; Target=$dest; Relative=$rel}
}
Write-Host "Validated $($manifest.Count) new files on $branch; no existing file will be overwritten." -ForegroundColor Cyan
foreach ($entry in $manifest) {
    $parent = Split-Path -Parent $entry.Target
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item -LiteralPath $entry.Source -Destination $entry.Target -ErrorAction Stop
}
Write-Host 'SkyGuard DB add-on files copied. No commit or push performed.' -ForegroundColor Green
& git -C $target status --short
