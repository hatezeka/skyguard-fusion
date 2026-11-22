param(
    [string]$BaseUrl = 'http://127.0.0.1:5173',
    [Parameter(Mandatory)][string]$StereoProfile,
    [Parameter(Mandatory)][string]$Python,
    [ValidateRange(1,100)][int]$Samples = 5,
    [string]$Output = "backend/reports/stereo-evidence-$(Get-Date -Format yyyyMMdd-HHmmss).json"
)
$ErrorActionPreference = 'Stop'
$origin = [uri]$BaseUrl
if (-not $origin.IsLoopback) { throw 'Use the local operator proxy.' }
if (Test-Path -LiteralPath $Output) { throw 'Report already exists.' }

$health = Invoke-RestMethod "$BaseUrl/api/media/health" -TimeoutSec 5
if (-not $health.model_found) { throw 'The configured model file is unavailable.' }
$profile = Get-Content -LiteralPath $StereoProfile -Raw | ConvertFrom-Json
$profileHash = (Get-FileHash -LiteralPath $StereoProfile -Algorithm SHA256).Hash.ToLowerInvariant()
$modelHash = (Get-FileHash -LiteralPath $health.model_path -Algorithm SHA256).Hash.ToLowerInvariant()
$runtime = & $Python -c "from importlib.metadata import version; print(version('ultralytics'))"
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine the supplied Python environment version.' }

$frames = [Collections.Generic.List[object]]::new()
$session = $null
$seen = [Collections.Generic.HashSet[int]]::new()
$until = (Get-Date).AddSeconds(35)
while ($frames.Count -lt $Samples -and (Get-Date) -lt $until) {
    $frame = Invoke-RestMethod "$BaseUrl/api/persistence/live/frame" -TimeoutSec 5
    if ($frame.state -in @('idle','stopped','error')) { break }
    if ($frame.stereo -and $frame.frame_index -ne $null) {
        if ($session -and $session -ne $frame.session_id) { throw 'LIVE session changed during sampling.' }
        $session = $frame.session_id
        if ($seen.Add([int]$frame.frame_index)) {
            $frames.Add([ordered]@{
                timestamp_utc = $frame.captured_at
                left_frame_index = $frame.frame_index
                right_frame_index = $frame.stereo.right_frame_index
                host_read_skew_ms = $frame.stereo.host_skew_ms
                timestamp_semantics = $frame.stereo.timestamp_semantics
                inference_ms = @($frame.stereo.left_inference_ms, $frame.stereo.right_inference_ms)
                analysis_fps = $frame.fps
                calibration_status = $frame.stereo.calibration_status
                profile_sha256 = $frame.stereo.profile_sha256
                range_status = $frame.stereo.range_status
                cameras = $frame.stereo.cameras
                observations = $frame.stereo.objects
            })
        }
    }
    Start-Sleep -Milliseconds 300
}

$matches = if ($frames.Count) { @($frames | Where-Object {$_.profile_sha256 -ne $profileHash}).Count -eq 0 } else { $null }
$report = [ordered]@{
    collected_utc = [datetime]::UtcNow.ToString('o')
    session_id = $session
    state = $frame.state
    error = $frame.error
    requested_samples = $Samples
    collected_samples = $frames.Count
    calibration_profile = @{
        file = Split-Path -Leaf $StereoProfile
        status = $profile.status
        sha256 = $profileHash
        matches_live_snapshot = $matches
    }
    model = @{
        file = Split-Path -Leaf $health.model_path
        model_directory = Split-Path -Leaf (Split-Path -Parent $health.model_path)
        configured_file_sha256 = $modelHash
        verifier_ultralytics_version = ([string]$runtime).Trim()
    }
    frames = $frames.ToArray()
}
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Output $Output
