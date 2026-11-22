param([string]$Repository = (Split-Path -Parent $PSScriptRoot), [string]$Python, [string]$ModelPath, [string]$DatabasePath, [switch]$Demo, [ValidateRange(1,65535)][int]$BackendPort = 8000, [ValidateRange(1,65535)][int]$FrontendPort = 5173)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath($Repository)
$branch=(& git -C $repo branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-final-engineering') { throw 'Unexpected branch' }
$backend=Join-Path $repo 'backend';$frontend=Join-Path $repo 'frontend'
if ($env:SKYGUARD_DATABASE_URL) { throw 'A custom SKYGUARD_DATABASE_URL is configured; verify it before starting local services.' }
if (-not $Python) { $Python=Join-Path $backend '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw "Backend venv missing: $python" }
if ($ModelPath) {
  $ModelPath=(Resolve-Path -LiteralPath $ModelPath).Path
  if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "Checkpoint file required: $ModelPath" }
}
if ($Demo -and -not $DatabasePath) { throw 'Demo mode needs an isolated database file' }
if ($DatabasePath) {
  $DatabasePath=(Resolve-Path -LiteralPath $DatabasePath).Path
  if (-not (Test-Path -LiteralPath $DatabasePath -PathType Leaf) -or [IO.Path]::GetExtension($DatabasePath) -ne '.sqlite3') { throw 'Expected an existing SQLite database file' }
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Node and npm.cmd required' }
if ($BackendPort -eq $FrontendPort) { throw 'Backend and frontend ports must differ' }
foreach ($port in @($BackendPort,$FrontendPort)) {
  $client=New-Object Net.Sockets.TcpClient
  try {
    $result=$client.BeginConnect('127.0.0.1',$port,$null,$null)
    if ($result.AsyncWaitHandle.WaitOne(300)) {
      try { $client.EndConnect($result); throw "Port $port already in use; do not start duplicate services" }
      catch [System.Net.Sockets.SocketException] { }
    }
  } finally { $client.Close() }
}
# Local-only bearer key, generated with cryptographic randomness; never written to file.
$bytes=New-Object byte[] 32
$rng=[System.Security.Cryptography.RandomNumberGenerator]::Create()
try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
$key=-join ($bytes | ForEach-Object { $_.ToString('x2') })
$logDir=Join-Path $env:TEMP "SkyGuardOperatorLogs-$BackendPort-$FrontendPort"
if (-not (Test-Path $logDir)) { [void](New-Item -ItemType Directory -Path $logDir) }
$previousDbKey=$env:SKYGUARD_DB_API_KEY
$previousDbUrl=$env:SKYGUARD_DATABASE_URL
$previousDemoMode=$env:SKYGUARD_DEMO_MODE
$previousMediaDir=$env:SKYGUARD_MEDIA_REPORT_DIR
$previousApiKey=$env:SKYGUARD_API_KEY
$previousModelPath=$env:SKYGUARD_MODEL_PATH
$previousBackendPort=$env:SKYGUARD_BACKEND_PORT
try {
  $env:SKYGUARD_DB_API_KEY=$key
  $env:SKYGUARD_API_KEY=$key
  if ($DatabasePath) { $env:SKYGUARD_DATABASE_URL = 'sqlite:///' + $DatabasePath.Replace([char]92,[char]47) }
  if ($ModelPath) { $env:SKYGUARD_MODEL_PATH=$ModelPath }
  if ($Demo) { $env:SKYGUARD_MEDIA_REPORT_DIR = Join-Path (Split-Path $DatabasePath) 'demo-media' }
  $api=Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','app.skyguard_persistence.composed:app','--host','127.0.0.1','--port',$BackendPort) -WorkingDirectory $backend -RedirectStandardOutput (Join-Path $logDir 'backend-out.log') -RedirectStandardError (Join-Path $logDir 'backend-err.log') -WindowStyle Hidden -PassThru
  $env:SKYGUARD_DATABASE_URL=$previousDbUrl
  $env:SKYGUARD_MODEL_PATH=$previousModelPath
  if ($Demo) { $env:SKYGUARD_DEMO_MODE = '1' }
  $env:SKYGUARD_BACKEND_PORT=[string]$BackendPort
  $ui=Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d','/s','/c',"npm.cmd run dev -- --config operator.vite.config.ts --host 127.0.0.1 --port $FrontendPort --strictPort") -WorkingDirectory $frontend -RedirectStandardOutput (Join-Path $logDir 'frontend-out.log') -RedirectStandardError (Join-Path $logDir 'frontend-err.log') -WindowStyle Hidden -PassThru
} finally {
  $env:SKYGUARD_DB_API_KEY=$previousDbKey
  $env:SKYGUARD_DATABASE_URL=$previousDbUrl
  $env:SKYGUARD_DEMO_MODE=$previousDemoMode
  $env:SKYGUARD_MEDIA_REPORT_DIR=$previousMediaDir
  $env:SKYGUARD_API_KEY=$previousApiKey
  $env:SKYGUARD_MODEL_PATH=$previousModelPath
  $env:SKYGUARD_BACKEND_PORT=$previousBackendPort
}
Write-Host ''
Write-Host 'SkyGuard local operator services requested.' -ForegroundColor Green
Write-Host "Browser: http://127.0.0.1:${FrontendPort}/operator.html"
Write-Host "Backend PID: $($api.Id); UI PID: $($ui.Id)"
Write-Host "Logs: $logDir"
Write-Host ''
Write-Host 'Persistence reconnects automatically through the local proxy.'
Write-Host 'If a service fails, review log files; do not assume startup succeeded.'
