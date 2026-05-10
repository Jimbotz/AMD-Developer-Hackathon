# ADR Validator Hackathon - Agent Instructions

## Project Overview

This is an AMD ROCm hackathon project that validates Architecture Decision Records (ADRs) using:
- **Qwen3-8B** (fine-tuned) for LLM inference
- **Qdrant** vector database for semantic search of ADRs
- **Obsidian plugin** for real-time validation in note-taking app

## Quick Start Commands

```bash
# 1. Start Qdrant (required for backend)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Index ADRs into Qdrant
cd backend
pip install -r requirements.txt
python qdrant_setup.py              # uses mock vectors (sentence-transformers unavailable on ROCm)
python qdrant_setup.py --embed     # same fallback on ROCm hardware

# 3. Generate training data + Fine-tune model (requires AMD MI300X or GPU 24GB+ VRAM)
cd ../fine-tuning
python generate_dataset.py        # build training-data.jsonl first
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"
python fine_tune_qwen3_8b.py       # outputs to ./models/qwen-adr-lora

# 4. Start backend API
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## ROCm Compatibility Note

`sentence-transformers` AND `transformers` share a PyTorch custom-op type-checking bug on ROCm (fails to register `grouped_mm_fallback`). Both crash at import time with `ValueError: infer_schema(func): Parameter input has unsupported type torch.Tensor`.

All code paths now catch this and fall back gracefully:
- Embeddings: use mock MD5-based vectors with a visible warning
- Fine-tuning: requires Unsloth (`pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"`)

This is a known upstream issue; it does not affect the API's core validation logic. The backend starts and validates ADRs using rule-based detection even without the model.

## Project Structure

```
adr-validator-hackathon/
├── backend/
│   ├── main.py              # FastAPI server (port 8000), graceful embedding fallback
│   └── qdrant_setup.py      # Index ADRs, --embed flag with ROCm-safe import
├── fine-tuning/
│   ├── generate_dataset.py  # build training-data.jsonl before fine-tuning
│   ├── fine_tune_qwen3_8b.py
│   └── training-data.jsonl
├── plugin-obsidian/         # Obsidian plugin (TypeScript)
└── data/                    # Cloned ADR references
    ├── adrs/               # Kubernetes, Django, Rust ADRs
    └── security/           # OWASP, Bandit examples
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check (model loaded, Qdrant connected, embeddings loaded) |
| POST | `/validate-adr` | Validate ADR for contradictions + risks |
| POST | `/index-adr` | Add ADR to Qdrant |

## Fine-tuning Notes

- **Prerequisite**: Run `python generate_dataset.py` first to create `training-data.jsonl`
- **Model**: `Qwen/Qwen3-8B`
- **Method**: LoRA (native BF16 on MI300X, FP16 on CUDA)
- **Hardware**: AMD MI300X (192GB HBM3) ideal; MI210 (64GB) or any GPU 24GB+ works
- **ROCm detection**: script checks `/dev/kfd` and auto-selects BF16/FP16 accordingly
- **Expected time**: ~1.5 hours on MI300X for 3 epochs
- **Output**: `models/qwen-adr-lora/`

## Qdrant Collection Schema

Collection: `adrs`
- Vector size: 1024 (Qwen3-embed-8b; mock vectors on ROCm)
- Distance: COSINE
- Payload indexes: `status`, `category`, `title`

## Obsidian Plugin Development

```bash
cd plugin-obsidian
npm install
npm run build   # Compile TypeScript with esbuild
```

Plugin location for manual install: copy to vault's `.obsidian/plugins/` folder. Configure API endpoint in plugin settings (default: `http://localhost:8000`).

## Important Constraints

- **Qdrant must be running BEFORE starting the backend**
- Plugin requires backend at `http://localhost:8000` (configurable in plugin settings)
- On AMD cloud, open port `8000` in the instance security group for external plugin access
- Fine-tuning on AMD cloud: use the ROCm-indexed PyTorch wheel, not standard PyPI torch
