# Khoi dong day du: Ollama (neu can) + RAG API + Cloudflare Tunnel
Set-Location $PSScriptRoot

$ApiPort = 8000
$TunnelName = "iqgs-rag"
$PublicHost = "iqgsrag.cloud"
$MutexName = "Global\IQGS-RAG-StartAll"

function Write-Step([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Test-OllamaRunning {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Start-OllamaIfNeeded {
    if (Test-OllamaRunning) {
        Write-Host "Ollama: OK" -ForegroundColor Green
        return
    }

    Write-Step "Ollama chua chay - dang thu khoi dong..."
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Host "Loi: Khong tim thay lenh 'ollama'. Hay mo Ollama app roi chay lai script." -ForegroundColor Red
        exit 1
    }

    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 1; $i -le 10; $i++) {
        Start-Sleep -Seconds 2
        if (Test-OllamaRunning) {
            Write-Host "Ollama: da khoi dong" -ForegroundColor Green
            return
        }
    }

    Write-Host "Loi: Ollama khong phan hoi sau 20 giay. Mo Ollama app thu cong." -ForegroundColor Red
    exit 1
}

function Test-ApiHealthy {
    try {
        $url = "http://localhost:$ApiPort/health"
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-TunnelRunning {
    $pattern = "tunnel run\s+" + [regex]::Escape($TunnelName)
    $procs = Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        if ($proc.CommandLine -match $pattern) {
            return $true
        }
    }
    return $false
}

function Stop-StaleProcessOnPort([int]$Port) {
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) { return }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        Write-Step "Port $Port bi chiem nhung API khong healthy - dung PID $procId..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

function Wait-ApiReady([int]$MaxSeconds) {
    $deadline = (Get-Date).AddSeconds($MaxSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-ApiHealthy) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-ApiIfNeeded {
    if (Test-ApiHealthy) {
        $healthUrl = "http://localhost:$ApiPort/api/v1/health"
        Write-Host "API: dang chay san - $healthUrl" -ForegroundColor Green
        return
    }

    Stop-StaleProcessOnPort -Port $ApiPort

    $root = $PSScriptRoot -replace "'", "''"
    $apiScript = @"
`$Host.UI.RawUI.WindowTitle = 'IQGS RAG API'
Set-Location '$root'
Write-Host 'RAG API - http://localhost:$ApiPort' -ForegroundColor Cyan
Write-Host 'Swagger: http://localhost:$ApiPort/api/v1/docs' -ForegroundColor Yellow
Write-Host 'Nhan Ctrl+C de dung API' -ForegroundColor Gray
python -m uvicorn api.app:app --host 0.0.0.0 --port $ApiPort
"@

    Write-Step "Khoi dong RAG API (cua so moi)..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiScript

    Write-Step "Doi API san sang..."
    if (Wait-ApiReady -MaxSeconds 30) {
        Write-Host "API: OK" -ForegroundColor Green
    } else {
        Write-Host "Canh bao: API chua phan hoi - kiem tra cua so 'IQGS RAG API'." -ForegroundColor Yellow
    }
}

function Start-TunnelIfNeeded {
    if (Test-TunnelRunning) {
        Write-Host "Tunnel: dang chay san - https://$PublicHost" -ForegroundColor Green
        return
    }

    $tunnelScript = @"
`$Host.UI.RawUI.WindowTitle = 'IQGS Cloudflare Tunnel'
Write-Host 'Cloudflare Tunnel - $TunnelName' -ForegroundColor Cyan
Write-Host 'Public: https://$PublicHost' -ForegroundColor Yellow
Write-Host 'Nhan Ctrl+C de dung tunnel' -ForegroundColor Gray
cloudflared tunnel run $TunnelName
"@

    Write-Step "Khoi dong Cloudflare Tunnel (cua so moi)..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $tunnelScript
    Start-Sleep -Seconds 2

    if (Test-TunnelRunning) {
        Write-Host "Tunnel: da khoi dong" -ForegroundColor Green
    } else {
        Write-Host "Canh bao: Tunnel chua thay process - kiem tra cua so 'IQGS Cloudflare Tunnel'." -ForegroundColor Yellow
    }
}

$mutex = New-Object System.Threading.Mutex($false, $MutexName)
if (-not $mutex.WaitOne(0, $false)) {
    Write-Host ""
    Write-Host "Start-all dang chay hoac vua chay xong." -ForegroundColor Yellow
    Write-Host "KHONG click nhieu lan - se lam tat API." -ForegroundColor Yellow
    Write-Host "Neu can khoi dong lai: chay stop-all.bat truoc." -ForegroundColor Gray
    Write-Host ""
    pause
    exit 0
}

try {
    Write-Host ""
    Write-Host "=== IQGS RAG - Start All ===" -ForegroundColor White
    Write-Host ""

    Start-OllamaIfNeeded

    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared) {
        Write-Host "Loi: Khong tim thay 'cloudflared'. Cai Cloudflare Tunnel CLI roi chay lai." -ForegroundColor Red
        exit 1
    }

    Start-ApiIfNeeded
    Start-TunnelIfNeeded

    Write-Host ""
    Write-Host "=== San sang ===" -ForegroundColor Green
    Write-Host "  Local:   http://localhost:$ApiPort/api/v1/docs"
    Write-Host "  Public:  https://$PublicHost/api/v1/docs"
    Write-Host ""
    Write-Host "Dung tat ca: chay stop-all.bat" -ForegroundColor Gray
    Write-Host "KHONG click start-all.bat nhieu lan." -ForegroundColor Yellow
    Write-Host ""
} finally {
    $mutex.ReleaseMutex() | Out-Null
    $mutex.Dispose()
}

pause
