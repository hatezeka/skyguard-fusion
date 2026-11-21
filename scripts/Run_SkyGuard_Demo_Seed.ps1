# Run ONLY against the local SkyGuard PostgreSQL Docker deployment.
# No rebuild, prune, DB wipe, migration, secret output or Git mutation.
[CmdletBinding()]
param(
    [string]$Repository = 'C:\Users\nistr\skyguard-fusion',
    [switch]$Seed,
    [switch]$VerifyGateway
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($Repository).TrimEnd('\')
$source = Join-Path $repo 'backend\scripts\seed_skyguard_demo.py'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw 'Demo seed source not installed into repository.' }
if ((Get-Item -LiteralPath $source).Length -gt 30000) { throw 'Unexpected seed source file size; stop.' }
$branch = (& git -C $repo branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -cne 'feature/skyguard-mvp') { throw 'Wrong Git branch.' }
$container = 'skyguard-fusion-backend-1'
$running = (& docker inspect -f '{{.State.Running}}' $container 2>$null)
if ($LASTEXITCODE -ne 0 -or $running.Trim() -ne 'true') { throw 'Backend Docker container is not running.' }
$ready = Invoke-RestMethod -Uri 'http://127.0.0.1:8088/healthz' -TimeoutSec 6
if ($ready.gateway -ne 'ready') { throw 'Gateway is not ready.' }
Write-Host 'GATEWAY_READY_AND_BACKEND_RUNNING' -ForegroundColor Green
$target = '/tmp/skyguard_seed_demo_v1.py'
& docker cp $source ($container + ':' + $target) | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker copy to temporary container directory failed.' }
$argv = @('exec', '-w', '/opt/skyguard/backend', '-e', 'PYTHONPATH=/opt/skyguard/backend', '-e', 'SKYGUARD_DEMO_SEED_ALLOWED=LOCAL_ONLY_2026', $container, 'python', $target)
if ($Seed) { $argv += '--seed' }
Write-Host '=== LOCAL POSTGRES DEMO: preflight + opt-in seed ===' -ForegroundColor Cyan
$result = @(& docker @argv)
$code = $LASTEXITCODE
$result | ForEach-Object { Write-Host $_ }
if ($code -ne 0) { throw 'Demo PostgreSQL validation/seeding failed. Do not claim end-to-end success.' }
if ($Seed -and -not ($result -contains 'SKYGUARD_DEMO_DB_VERIFIED')) { throw 'Demo database verification marker missing.' }
if (-not $VerifyGateway) {
    Write-Host 'API_BROWSER_CHECK_NOT_PERFORMED: use -VerifyGateway to verify logged-in HTTP API with password prompt.' -ForegroundColor Yellow
    exit 0
}
# Credentials are NEVER stored on disk or copied into the Git tree.
Write-Host 'Operator password will be requested without displaying typed characters.'
$secure = Read-Host 'Current SkyGuard operator password' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
try {
    $loginBody = @{ username = 'operator'; password = $plain } | ConvertTo-Json -Compress
    $web = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $base='http://127.0.0.1:8088'
    $null = Invoke-RestMethod -Uri ($base + '/auth/login') -Method Post -WebSession $web -Body $loginBody -ContentType 'application/json' -Headers @{ Origin=$base } -TimeoutSec 10
    $sessionCheck = Invoke-RestMethod -Uri ($base + '/auth/session') -WebSession $web -TimeoutSec 10
    if (-not $sessionCheck.authenticated) { throw 'Operator HTTP session missing.' }
    $cameras = @(Invoke-RestMethod -Uri ($base + '/api/persistence/cameras') -WebSession $web -TimeoutSec 15)
    $sessions = @(Invoke-RestMethod -Uri ($base + '/api/persistence/sessions') -WebSession $web -TimeoutSec 15)
    $demo = @($sessions | Where-Object { $_.dataset_ref -eq 'SKYGUARD_DEMO_FIXTURE_V1' })
    if ($demo.Count -ne 1) { throw 'Demo session not visible through authenticated gateway.' }
    $sid=[uri]::EscapeDataString([string]$demo[0].id)
    $dets = @(Invoke-RestMethod -Uri ($base + '/api/persistence/detections?session_id=' + $sid) -WebSession $web -TimeoutSec 15)
    $tracks = @(Invoke-RestMethod -Uri ($base + '/api/persistence/tracks?session_id=' + $sid) -WebSession $web -TimeoutSec 15)
    $alerts = @(Invoke-RestMethod -Uri ($base + '/api/persistence/alerts?session_id=' + $sid) -WebSession $web -TimeoutSec 15)
    $summary = Invoke-RestMethod -Uri ($base + '/api/persistence/sessions/' + $sid + '/summary') -WebSession $web -TimeoutSec 15
    if ($dets.Count -ne 15 -or $tracks.Count -ne 2 -or $alerts.Count -ne 2 -or $summary.frames -ne 5) { throw 'Backend/Gateway API counts differ from seeded database.' }
    if ((@($dets | Where-Object { $_.classification -eq 'UNKNOWN' -and $null -ne $_.track_id })).Count -ne 0) { throw 'UNKNOWN unexpectedly has track IDs.' }
    Write-Host 'SKYGUARD_AUTHENTICATED_GATEWAY_BACKEND_POSTGRES_E2E_VERIFIED' -ForegroundColor Green
    Write-Host '5 demo frames / 15 detections / 2 tracks / 2 alerts / 0 fake GPS and FPS' -ForegroundColor Green
    Write-Host 'FRONTEND_BROWSER_RENDERING_STILL_REQUIRES_VISUAL_CHECK' -ForegroundColor Yellow
    Write-Host 'Operator UI: http://127.0.0.1:8088/operator.html'
} finally {
    $plain = $null; $loginBody = $null
}
