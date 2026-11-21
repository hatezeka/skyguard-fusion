param([string]$Repository = 'C:\Users\nistr\skyguard-fusion')
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath($Repository)
$branch=(& git -C $repo branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-mvp') { throw 'Unexpected branch' }
$backend=Join-Path $repo 'backend';$frontend=Join-Path $repo 'frontend'
if ($env:SKYGUARD_DATABASE_URL) { throw 'A custom SKYGUARD_DATABASE_URL is configured; verify it before starting local services.' }
$python=Join-Path $backend '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Backend venv missing: $python" }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Node and npm.cmd required' }
foreach ($port in @(8000,5173)) {
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
$env:SKYGUARD_DB_API_KEY=$key
$logDir=Join-Path $env:TEMP 'SkyGuardOperatorLogs'
if (-not (Test-Path $logDir)) { [void](New-Item -ItemType Directory -Path $logDir) }
try {
  $api=Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','app.skyguard_persistence.composed:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $backend -RedirectStandardOutput (Join-Path $logDir 'backend-out.log') -RedirectStandardError (Join-Path $logDir 'backend-err.log') -PassThru
} finally { Remove-Item Env:SKYGUARD_DB_API_KEY -ErrorAction SilentlyContinue }
$ui=Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d','/s','/c','npm.cmd run dev -- --config operator.vite.config.ts --host 127.0.0.1 --strictPort') -WorkingDirectory $frontend -RedirectStandardOutput (Join-Path $logDir 'frontend-out.log') -RedirectStandardError (Join-Path $logDir 'frontend-err.log') -PassThru
Write-Host ''
Write-Host 'SkyGuard local operator services requested.' -ForegroundColor Green
Write-Host 'Browser: http://127.0.0.1:5173/operator.html'
Write-Host "Backend PID: $($api.Id); UI PID: $($ui.Id)"
Write-Host "Logs: $logDir"
Write-Host ''
Write-Host 'LOCAL OPERATOR KEY (paste under Integration settings, never share it):' -ForegroundColor Yellow
Write-Host $key -ForegroundColor Cyan
Write-Host 'This key lives only in backend process environment and console session.'
Write-Host 'If a service fails, review log files; do not assume startup succeeded.'
