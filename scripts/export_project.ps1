# ============================================================
# Export OSINT Project for Transfer
# Only copies code + required model weights (not training junk)
# ============================================================

$SRC = "c:\Users\kundu\Desktop\Projects\ProjectSem8"
$DEST = "c:\Users\kundu\Desktop\OSINT_Export"

Write-Host "=== OSINT Project Exporter ===" -ForegroundColor Cyan
Write-Host "Source: $SRC"
Write-Host "Destination: $DEST"

# Clean previous export
if (Test-Path $DEST) { Remove-Item $DEST -Recurse -Force }

# 1. Copy code (backend, extension, docker, etc.) — skip models, data, __pycache__, venv
Write-Host "`n[1/4] Copying project code..." -ForegroundColor Yellow
$excludeDirs = @("__pycache__", ".git", "venv", ".venv", "node_modules", ".ipynb_checkpoints")
robocopy $SRC $DEST /E /XD $excludeDirs /XD "$SRC\ml\training\models" /XD "$SRC\models" /XF *.pt *.pth *.bin *.onnx *.safetensors /XF osint_logs.db /NFL /NDL /NJH /NJS /NC /NS

# 2. Copy ONLY the required model checkpoints (not all 40 GB)
Write-Host "[2/4] Copying required model weights..." -ForegroundColor Yellow

# Sentiment model: checkpoint-29000 + tokenizer files from parent
$sentDest = "$DEST\ml\training\models\unified_sentiment\trial_0\checkpoint-29000"
New-Item -Path $sentDest -ItemType Directory -Force | Out-Null
Copy-Item "$SRC\ml\training\models\unified_sentiment\trial_0\checkpoint-29000\model.safetensors" $sentDest
Copy-Item "$SRC\ml\training\models\unified_sentiment\trial_0\checkpoint-29000\config.json" $sentDest

# Sentiment tokenizer (in the parent unified_sentiment folder)
$sentTokenizer = "$DEST\ml\training\models\unified_sentiment"
$tokenizerFiles = @("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "config.json")
foreach ($f in $tokenizerFiles) {
    $srcFile = "$SRC\ml\training\models\unified_sentiment\$f"
    if (Test-Path $srcFile) {
        Copy-Item $srcFile $sentTokenizer
    }
}

# Misinfo model: checkpoint-1692 + tokenizer
$misinfoDest = "$DEST\ml\training\models\misinfo_classifier\checkpoint-1692"
New-Item -Path $misinfoDest -ItemType Directory -Force | Out-Null
Copy-Item "$SRC\ml\training\models\misinfo_classifier\checkpoint-1692\model.safetensors" $misinfoDest
Copy-Item "$SRC\ml\training\models\misinfo_classifier\checkpoint-1692\config.json" $misinfoDest

$misinfoTokenizer = "$DEST\ml\training\models\misinfo_classifier"
foreach ($f in $tokenizerFiles) {
    $srcFile = "$SRC\ml\training\models\misinfo_classifier\$f"
    if (Test-Path $srcFile) {
        Copy-Item $srcFile $misinfoTokenizer
    }
}

# 3. Create a setup script for the friend
Write-Host "[3/4] Creating setup script..." -ForegroundColor Yellow

$setupScript = @'
# ============================================================
# OSINT Project Setup Script
# Run this on the target machine
# ============================================================

Write-Host "=== OSINT Project Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment
Write-Host "[1/5] Creating Python virtual environment..."
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
Write-Host "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install PyTorch (CUDA or CPU)
$hasGpu = python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($hasGpu -eq "True") {
    Write-Host "[3/5] Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "[3/5] Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# 4. HuggingFace models will auto-download on first run
Write-Host "[4/5] HuggingFace models (BART, RoBERTa, SBERT) will download on first startup (~3 GB)"

# 5. Start the server
Write-Host "[5/5] Starting server..."
Write-Host ""
Write-Host "Run: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Green
Write-Host ""
Write-Host "Then load the Chrome extension from the 'extension' folder." -ForegroundColor Green
'@

Set-Content -Path "$DEST\SETUP.ps1" -Value $setupScript

# 4. Report
Write-Host "`n[4/4] Calculating export size..." -ForegroundColor Yellow
$exportSize = (Get-ChildItem $DEST -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Host "`n=== Export Complete ===" -ForegroundColor Green
Write-Host "Location: $DEST"
Write-Host "Size: $([math]::Round($exportSize / 1GB, 2)) GB (down from 43.8 GB)"
Write-Host ""
Write-Host "Transfer this folder to your friend via:"
Write-Host "  - USB drive"
Write-Host "  - Google Drive / OneDrive"
Write-Host "  - zip and share"
Write-Host ""
Write-Host "On friend's PC: run SETUP.ps1"
