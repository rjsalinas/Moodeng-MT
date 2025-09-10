## GPU Training Quick Re-Run Guide (Windows PowerShell + Docker + CUDA)

This guide restarts training later using your RTX GPU via Docker (CUDA). Each step is one command.

### 0) Prerequisites (done once)
- Install WSL2 + Ubuntu, Docker Desktop, and enable WSL integration. NVIDIA drivers should already be installed on Windows.

### 1) Open Windows PowerShell
Open a new PowerShell window as your user.

### 2) Define your project path
```powershell
$PROJ="C:\Users\Reejay Salinas\OneDrive - Asia Pacific College\Desktop\SalinTala\Moodeng-MT"
```

### 3) (Optional) Clean up any previous container named moodeng
```powershell
docker rm -f moodeng 2>$null
```

### 4) Pull the CUDA PyTorch image (do this occasionally to refresh)
```powershell
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
```

### 5) Start a GPU-enabled container with your repo mounted
```powershell
docker run --gpus all -it --name moodeng --shm-size=16g --mount type=bind,source="$PROJ",target=/workspace -w /workspace pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime bash
```

You should now see a prompt like `root@...:/workspace#` inside the container.

---

## Inside the Container (one command per step)

### 6) Upgrade base Python tooling
```bash
pip install -U pip setuptools wheel
```

### 7) Install project dependencies (plus extras used by the pipeline)
```bash
pip install -r requirements.txt sentencepiece calamancy "spacy[transformers]"
```

### 8) Download spaCy models
```bash
python -m spacy download en_core_web_sm
```

### 9) Download multilingual NER model
```bash
python -m spacy download xx_ent_wiki_sm
```

### 10) Download required NLTK data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 11) (Only if needed) Install latest CUDA nightly PyTorch for SM 12.0 GPUs
```bash
pip uninstall -y torch torchvision torchaudio && pip install --pre --upgrade --extra-index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio
```

### 12) Verify CUDA is available
```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', getattr(torch.version,'cuda',None)); print('Device:', torch.cuda.get_device_name(0))"
```

### 13) Start training (CUDA)
```bash
TOKENIZERS_PARALLELISM=false python model_training_enhanced.py
```

---

## Optional Operations

### A) Monitor GPU usage from Windows (separate PowerShell)
```powershell
nvidia-smi
```

### B) Stop the container training run (from inside container)
```bash
pkill -f model_training_enhanced.py || true
```

### C) Detach from the container without stopping (inside container)
```bash
exit
```

### D) Reattach to the existing container later (PowerShell)
```powershell
docker start -moodeng
```

### E) Copy out results from container (PowerShell; container must be running)
```powershell
docker cp moodeng:/workspace/logs .\
```

### F) Clean up the container when done (PowerShell)
```powershell
docker rm -f moodeng
```

---

## Notes
- Keep one-command-per-step to avoid PowerShell quoting issues.
- If you ever get a CUDA capability warning for RTX 5070, use step 11 to install the nightly cu124 build.
- If you hit out-of-memory, ask to reduce batch size and enable gradient checkpointing in `model_training_enhanced.py`.

---

## Safe Start/Stop (Quick Reference)

### Start (new session)
```powershell
cd "C:\Users\Reejay Salinas\OneDrive - Asia Pacific College\Desktop\SalinTala\Moodeng-MT"
./start_cuda_container.ps1
```

### Reattach (if container already exists)
```powershell
docker start -moodeng
```

### Stop training inside container (no forced kill)
```bash
pkill -f model_training_enhanced.py || true
```

### Exit the container shell
```bash
exit
```

### Stop container from PowerShell
```powershell
docker stop moodeng
```

### Remove container safely (does not delete your project files)
```powershell
docker rm moodeng
```

### Optional: free unused Docker data (images/cache)
```powershell
docker system prune -f
```

Note: Model outputs and logs are written to your mounted Windows folder (`Moodeng-MT`). Stopping/removing the container will not delete those files.


