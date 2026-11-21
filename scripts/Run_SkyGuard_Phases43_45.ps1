# SkyGuard FUSION: evidence-first Phase 43-45 CLI runner.
# Never adds, commits, pushes, or changes production code automatically.
[CmdletBinding()]
param(
    [ValidateSet('tests','inventory','export-smoke','export-full','train','calibrate','benchmark','yolo-map','acceptance','field-template','usb-field')]
    [string]$Action='tests',
    [string]$HalmstadZip='',
    [string]$DutZip='',
    [string]$Dataset='',
    [string]$Weights='',
    [string]$Device='cpu',
    [int]$Epochs=35,
    [int]$CameraIndex=0,
    [int]$MaxFrames=300,
    [switch]$PhysicalConfirm
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$branch=(& git branch --show-current | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-mvp') {throw "Unexpected branch: $branch"}
$py=Join-Path $repo 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {throw "Missing backend Python: $py"}
$backend=Join-Path $repo 'backend'
$reports=Join-Path $backend 'reports\phase43_45'
New-Item -ItemType Directory -Path $reports -Force | Out-Null
$module=@('-m','app.phase43_45.cli')
function Call-Python([string[]]$CliArgs) {
    Push-Location $backend
    try {
        & $py @module @CliArgs
        if ($LASTEXITCODE -ne 0) {throw "SkyGuard phase43_45 CLI failed (exit $LASTEXITCODE)"}
    } finally {Pop-Location}
}
if (-not $HalmstadZip) {$HalmstadZip=Join-Path $env:USERPROFILE 'Downloads\Drone-detection-dataset-master.zip'}
if (-not $DutZip) {$DutZip=Join-Path $env:USERPROFILE 'Downloads\DUT-Anti-UAV-master (1).zip'}
if (-not $Dataset) {$Dataset=Join-Path $reports 'export_full'}
$manifest=Join-Path $reports 'input_sources.json'
switch ($Action) {
    'tests' {
        Push-Location $backend
        try {
            & $py -m pytest -q tests/test_phase43_45_scientific_contract.py tests/test_phase43_45_inventory.py
            if ($LASTEXITCODE -ne 0) {throw 'Phase43_45 tests failed'}
        } finally {Pop-Location}
    }
    'inventory' {
        if (-not (Test-Path -LiteralPath $HalmstadZip)) {throw "ZIP missing: $HalmstadZip"}
        $argsList=@('inventory','--halmstad',$HalmstadZip,'--out',$manifest)
        if (Test-Path -LiteralPath $DutZip) {$argsList+=@('--dut',$DutZip)}
        Call-Python -CliArgs $argsList
    }
    'export-smoke' {
        if (-not (Test-Path -LiteralPath $manifest)) {throw 'Run inventory first'}
        Call-Python -CliArgs @('export','--manifest',$manifest,'--out',(Join-Path $reports 'export_smoke'),
             '--stride','20','--max-sequences','24','--max-frames-per-video','8')
    }
    'export-full' {
        if (-not (Test-Path -LiteralPath $manifest)) {throw 'Run inventory first'}
        Call-Python -CliArgs @('export','--manifest',$manifest,'--out',$Dataset,'--stride','15')
    }
    'train' {
        Call-Python -CliArgs @('train','--data-yaml',(Join-Path $Dataset 'data.yaml'),
             '--epochs',[string]$Epochs,'--device',$Device,'--out',(Join-Path $reports 'training'))
    }
    'calibrate' {
        if (-not $Weights) {throw 'Specify -Weights path to FOUR class checkpoint'}
        Call-Python -CliArgs @('calibrate','--dataset',$Dataset,'--weights',$Weights,
             '--out',(Join-Path $reports 'calibration.json'))
    }
    'benchmark' {
        if (-not $Weights) {throw 'Specify -Weights path to FOUR class checkpoint'}
        $argsList=@('benchmark','--dataset',$Dataset,'--weights',$Weights,
              '--out',(Join-Path $reports '43'))
        $cal=Join-Path $reports 'calibration.json'
        if (Test-Path $cal) {$argsList+=@('--calibration',$cal)}
        Call-Python -CliArgs $argsList
    }
    'yolo-map' {
        if (-not $Weights) {throw 'Specify -Weights path to FOUR class checkpoint'}
        Call-Python -CliArgs @('yolo-map','--dataset',$Dataset,'--weights',$Weights,
            '--device',$Device,'--out',(Join-Path $reports '43\yolo_map.json'))
    }
    'acceptance' { Call-Python -CliArgs @('acceptance','--reports',$reports) }
    'field-template' { Call-Python -CliArgs @('field-template','--out',(Join-Path $reports '45\field_template.json')) }
    'usb-field' {
        if (-not $PhysicalConfirm) {throw 'Only in Phase 45 with physical USB camera and real drone: -PhysicalConfirm'}
        if (-not $Weights) {throw 'Specify -Weights FOUR class checkpoint'}
        $out=Join-Path $reports ('45\usb_trial_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
        Call-Python -CliArgs @('usb-field','--camera-index',[string]$CameraIndex,
             '--max-frames',[string]$MaxFrames,'--weights',$Weights,
             '--out',$out,'--physical-confirm')
    }
}
Write-Host "COMPLETE: $Action | No git commit or push performed" -ForegroundColor Green
