# Chạy RAG API local (không Docker)
Set-Location $PSScriptRoot

Write-Host "Kiem tra Ollama..." -ForegroundColor Cyan
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 5
    Write-Host "Ollama: OK" -ForegroundColor Green
} catch {
    Write-Host "Loi: Ollama chua chay. Mo Ollama app hoac chay: ollama serve" -ForegroundColor Red
    exit 1
}

Write-Host "Khoi dong RAG API tai http://localhost:8000 ..." -ForegroundColor Cyan
Write-Host "Health: http://localhost:8000/health" -ForegroundColor Yellow
Write-Host "Internal API: /internal/rag/* (can X-Internal-Api-Key)" -ForegroundColor Yellow
Write-Host "Nhan Ctrl+C de dung" -ForegroundColor Gray

python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
