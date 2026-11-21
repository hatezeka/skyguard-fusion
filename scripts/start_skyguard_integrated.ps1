param([string]$Repository = 'C:\Users\nistr\skyguard-fusion')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($Repository)
$branch = (& git -C $repo branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/skyguard-mvp') { throw 'Wrong Git branch' }
if ($env:SKYGUARD_DATABASE_URL) { throw 'A non-default DB URL exists: verify before using this local launcher' }
$python = Join-Path $repo 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Backend virtual environment missing' }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'npm.cmd is not installed' }
foreach ($port in @(8000, 5173)) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $ar = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        if ($ar.AsyncWaitHandle.WaitOne(300)) {
            try { $client.EndConnect($ar); throw "Port $port already in use - stop prior SkyGuard instances" }
            catch [Net.Sockets.SocketException] { }
        }
    } finally { $client.Close() }
}
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
$key = -join ($bytes | ForEach-Object { $_.ToString('x2') })
$env:SKYGUARD_DB_API_KEY = $key
$env:SKYGUARD_API_KEY = $key
$logDir = Join-Path $env:TEMP 'SkyGuardLiveIntegratedLogs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
try {
    $backend = Start-Process -FilePath $python `
        -ArgumentList @('-m','uvicorn','app.skyguard_persistence.live_integrated:app','--host','127.0.0.1','--port','8000') `
        -WorkingDirectory (Join-Path $repo 'backend') `
        -RedirectStandardOutput (Join-Path $logDir 'backend-stdout.log') `
        -RedirectStandardError (Join-Path $logDir 'backend-stderr.log') -PassThru
} finally {
    Remove-Item Env:SKYGUARD_DB_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:SKYGUARD_API_KEY -ErrorAction SilentlyContinue
}
$frontend = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/d','/s','/c','npm.cmd run dev -- --config operator.vite.config.ts --host 127.0.0.1 --strictPort') `
    -WorkingDirectory (Join-Path $repo 'frontend') `
    -RedirectStandardOutput (Join-Path $logDir 'frontend-stdout.log') `
    -RedirectStandardError (Join-Path $logDir 'frontend-stderr.log') -PassThru
Start-Sleep -Seconds 2
if ($backend.HasExited) { throw "Backend exited; see $logDir\backend-stderr.log" }
if ($frontend.HasExited) { throw "Frontend exited; see $logDir\frontend-stderr.log" }
Write-Host 'STARTED LOCAL SKYGUARD SERVICES' -ForegroundColor Green
Write-Host 'Open: http://127.0.0.1:5173/operator.html'
Write-Host "Backend PID $($backend.Id); Frontend PID $($frontend.Id); Logs: $logDir"
Write-Host 'LOCAL OPERATOR KEY (do not share; enter in Integration Settings):' -ForegroundColor Yellow
Write-Host $key -ForegroundColor Cyan
Write-Host 'This is a local development credential, NOT production authentication.'
