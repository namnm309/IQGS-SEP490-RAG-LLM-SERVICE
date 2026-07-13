# Dung RAG API + Cloudflare Tunnel
Set-Location $PSScriptRoot

$ApiPort = 8000
$TunnelName = "iqgs-rag"

Write-Host ""
Write-Host "=== IQGS RAG — Stop All ===" -ForegroundColor White
Write-Host ""

$apiStopped = 0
$connections = Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        Write-Host "Dung API tren port $ApiPort (PID $procId)..." -ForegroundColor Cyan
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        $apiStopped++
    }
} else {
    Write-Host "API: khong co process tren port $ApiPort" -ForegroundColor Gray
}

$tunnelStopped = 0
$procs = Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue
foreach ($proc in $procs) {
    if ($proc.CommandLine -match "tunnel run\s+$([regex]::Escape($TunnelName))") {
        Write-Host "Dung Cloudflare tunnel '$TunnelName' (PID $($proc.ProcessId))..." -ForegroundColor Cyan
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $tunnelStopped++
    }
}

if ($tunnelStopped -eq 0) {
    Write-Host "Tunnel: khong thay process '$TunnelName'" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Da dung $apiStopped API + $tunnelStopped tunnel." -ForegroundColor Green
Write-Host ""
pause
