param([string] $Repository = (Split-Path -Parent $PSScriptRoot), [string] $Python, [string] $ModelPath, [ValidateRange(1,65535)][int]$BackendPort = 8000, [ValidateRange(1,65535)][int]$FrontendPort = 5173)
& (Join-Path $PSScriptRoot 'start_skyguard_operator.ps1') -Repository $Repository -Python $Python -ModelPath $ModelPath -BackendPort $BackendPort -FrontendPort $FrontendPort
