param(
    [string]$Repository = 'C:\Users\nistr\skyguard-fusion',
    [string]$PackageDirectory = '',
    [switch]$StartAfterInstall
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($Repository)
if (-not $PackageDirectory) { $PackageDirectory = (Split-Path -Parent $PSScriptRoot) }
$source = [IO.Path]::GetFullPath($PackageDirectory)
if (-not (Test-Path -LiteralPath (Join-Path $repo '.git') -PathType Container)) { throw "Not a repository: $repo" }
$branch = (& git -C $repo branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-mvp') { throw "Wrong branch: $branch" }
$manifestPath = Join-Path $source 'SKYGUARD_OPERATOR_MANIFEST_SHA256.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Missing package manifest' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$sourceRoot = $source.TrimEnd('\') + '\'
$destRoot = $repo.TrimEnd('\') + '\'
$copy = @(); $skipped=0
foreach ($property in $manifest.files.PSObject.Properties) {
  $relative = $property.Name.Replace('/', '\')
  $src = [IO.Path]::GetFullPath((Join-Path $source $relative))
  $dst = [IO.Path]::GetFullPath((Join-Path $repo $relative))
  if (-not $src.StartsWith($sourceRoot,[StringComparison]::OrdinalIgnoreCase) -or -not $dst.StartsWith($destRoot,[StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe path: $relative" }
  if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw "Missing package source $relative" }
  $expected = [string]$property.Value
  $actual = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
  if ($actual -ne $expected) { throw "Package tampered or corrupted: $relative" }
  if (Test-Path -LiteralPath $dst) {
    if (-not (Test-Path -LiteralPath $dst -PathType Leaf)) { throw "Directory conflict: $relative" }
    $old = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash
    if ($old -ne $expected) { throw "Existing file conflict; not overwriting: $dst" }
    $skipped++
    continue
  }
  $copy += [pscustomobject]@{Source=$src;Destination=$dst}
}
Write-Host "Validated package; $($copy.Count) new files; $skipped existing identical files." -ForegroundColor Cyan
foreach ($f in $copy) {
  $dir = Split-Path -Parent $f.Destination
  if (-not (Test-Path -LiteralPath $dir)) { [void](New-Item -ItemType Directory -Path $dir -Force) }
  Copy-Item -LiteralPath $f.Source -Destination $f.Destination -ErrorAction Stop
}
Write-Host 'Operator Console files installed. No original App.tsx, package.json or Vite config overwritten.' -ForegroundColor Green
$front=Join-Path $repo 'frontend'
Push-Location $front
try {
  $cmd = Get-Command npm.cmd -ErrorAction Stop
  & $cmd.Source ci --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
  & $cmd.Source exec -- vitest run src/operator/logic.test.ts
  if ($LASTEXITCODE -ne 0) { throw 'Operator logic tests failed' }
  & $cmd.Source run build -- --config operator.vite.config.ts
  if ($LASTEXITCODE -ne 0) { throw 'Frontend TypeScript/Vite build failed' }
} finally { Pop-Location }
Write-Host 'Operator tests and production build passed on this computer.' -ForegroundColor Green
& git -C $repo status --short
if ($StartAfterInstall) { & (Join-Path $repo 'scripts\start_skyguard_operator.ps1') -Repository $repo }
