$ErrorActionPreference = 'Stop'

# 1) Resolve project path automatically (this script's folder)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJ = Resolve-Path $scriptDir
Write-Host "Project: $PROJ"

# 2) Remove any previous container named moodeng (ignore errors)
try { docker rm -f moodeng | Out-Null } catch {}

# 3) Pull CUDA PyTorch image (optional refresh)
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 4) Start container with GPU and mount the project
docker run --gpus all -it --name moodeng --shm-size=16g `
  --mount type=bind,source="$PROJ",target=/workspace `
  -w /workspace pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime bash


