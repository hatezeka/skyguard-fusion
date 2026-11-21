# Consistent SQLite backup using SQLite's online backup API; do not copy an active DB file.
[CmdletBinding()]
param(
    [string]$Repository = 'C:\Users\nistr\skyguard-fusion',
    [string]$Destination = ''
)
$ErrorActionPreference='Stop'
$backend = Join-Path $Repository 'backend'
$python = Join-Path $backend '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Backend Python venv missing' }
if ($env:SKYGUARD_DATABASE_URL -and -not $env:SKYGUARD_DATABASE_URL.StartsWith('sqlite')) {
    throw 'This script backs up SQLite only; use pg_dump for PostgreSQL'
}
if (-not $Destination) {
    $backupDir = Join-Path $env:USERPROFILE 'Documents\SkyGuardBackups'
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    $Destination = Join-Path $backupDir ('skyguard_backup_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.sqlite3')
}
$sourceFile = Join-Path $backend '.skyguard_data\skyguard.sqlite3'
if (-not (Test-Path $sourceFile)) { throw 'Default SQLite DB not found; check SKYGUARD_DATABASE_URL' }
$src = (Resolve-Path $sourceFile).Path
$dst = [System.IO.Path]::GetFullPath($Destination)
if ($src -eq $dst -or (Test-Path $dst)) { throw 'Backup destination exists or equals source' }
$script = 'import sqlite3,sys; src=sqlite3.connect(sys.argv[1]); dest=sqlite3.connect(sys.argv[2]); src.backup(dest); print("Backup written:",sys.argv[2]); dest.close(); src.close()'
& $python -c $script $src $dst
if ($LASTEXITCODE -ne 0) { throw 'SQLite backup failed' }
