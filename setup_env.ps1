# =============================================================
# Neuro-Neo4j Environment Bootstrap Script (Windows PowerShell)
# Author: basakanimesh14
# Usage:  .\setup_env.ps1
# =============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Neuro-Neo4j Environment Setup" -ForegroundColor Cyan
Write-Host "  Author: basakanimesh14" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1. Check / Install uv ----
Write-Host "[1/5] Checking for uv package manager..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "      uv not found. Installing via official installer..." -ForegroundColor Yellow
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + $env:PATH
    Write-Host "      [OK] uv installed." -ForegroundColor Green
} else {
    $uvVer = uv --version
    Write-Host "      [OK] uv found: $uvVer" -ForegroundColor Green
}

# ---- 2. Create virtual environment ----
Write-Host ""
Write-Host "[2/5] Creating virtual environment (.venv, Python 3.11)..." -ForegroundColor Yellow
uv venv .venv --python 3.11
Write-Host "      [OK] Virtual environment created." -ForegroundColor Green

# ---- 3. Install dependencies ----
Write-Host ""
Write-Host "[3/5] Installing dependencies from pyproject.toml..." -ForegroundColor Yellow
uv sync
Write-Host "      [OK] Dependencies installed." -ForegroundColor Green

# ---- 4. Copy .env ----
Write-Host ""
Write-Host "[4/5] Configuring environment file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "      [INFO] .env created from .env.example" -ForegroundColor Yellow
    Write-Host "      --> Please edit .env and set your Neo4j password." -ForegroundColor Magenta
} else {
    Write-Host "      [OK] .env already exists (skipped)." -ForegroundColor Green
}

# ---- 5. Create required directories ----
Write-Host ""
Write-Host "[5/5] Creating project directories..." -ForegroundColor Yellow
$dirs = @(
    "data\raw",
    "data\processed",
    "data\synthetic\connectomes",
    "checkpoints",
    "logs",
    "neo4j\schema",
    "neo4j\etl",
    "neo4j\gds"
)
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "      [OK] Directories created." -ForegroundColor Green

# ---- Done ----
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Edit .env with your Neo4j credentials" -ForegroundColor White
Write-Host "  2. Start Neo4j:        docker compose up -d" -ForegroundColor White
Write-Host "  3. Activate venv:      .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  4. Generate data:      uv run python data\synthetic\generate_synthetic.py" -ForegroundColor White
Write-Host "  5. Run dry-run:        uv run python dry_run\dry_run_all.py" -ForegroundColor White
Write-Host "  6. Start API:          uv run uvicorn api.main:app --reload" -ForegroundColor White
Write-Host ""
