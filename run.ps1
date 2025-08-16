param(
  [switch]$Rebuild
)

if ($Rebuild) {
  docker compose build --no-cache
}

docker compose up -d
Start-Sleep -Seconds 2
Invoke-WebRequest http://localhost:8000/health -UseBasicParsing
Write-Host "Open http://localhost:8000/ in your browser."
