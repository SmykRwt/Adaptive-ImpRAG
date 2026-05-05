Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Step 1/4: Building demo index..."
python scripts/build_faiss_index.py --config examples/open_baseline_config.json --passages examples/passages.jsonl --index-path outputs/open_demo.index --metadata-path outputs/open_demo_meta.json

Write-Host "Step 2/4: Training baseline model..."
python scripts/train_imprag.py --config examples/open_baseline_config.json --train-jsonl examples/train.jsonl --epochs 1 --warmup-epochs 1 --model-output-dir outputs/open_baseline --config-out outputs/open_baseline_config_out.json

Write-Host "Step 3/4: Training Adaptive ImpRAG model..."
python scripts/train_imprag.py --config examples/open_plus_config.json --train-jsonl examples/train.jsonl --epochs 1 --warmup-epochs 1 --model-output-dir outputs/open_plus --config-out outputs/open_plus_config_out.json

Write-Host "Step 4/4: Running demo inference..."
python scripts/infer_imprag.py --config outputs/open_plus_config_out.json --index-path outputs/open_demo.index --metadata-path outputs/open_demo_meta.json --query "What is the capital of France?" --top-k 2 --max-new-tokens 3
