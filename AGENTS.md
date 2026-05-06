# ADR Validator Hackathon - Agent Instructions

## Project Overview

This is an AMD ROCm hackathon project that validates Architecture Decision Records (ADRs) using:
- **Qwen3-8B** (fine-tuned) for LLM inference
- **Qdrant** vector database for semantic search of ADRs
- **Obsidian plugin** for real-time validation in note-taking app

## Quick Start Commands

```bash
# 1. Start Qdrant (required for backend)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Index ADRs into Qdrant
cd backend
pip install -r requirements.txt
python qdrant_setup.py

# 3. Fine-tune model (requires AMD MI300X or GPU with 24GB+ VRAM)
cd fine-tuning
pip install unsloth transformers peft torch
python fine_tune_qwen3_8b.py

# 4. Start backend API
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
adr-validator-hackathon/
├── backend/           # FastAPI server
│   ├── main.py        # API entrypoint (port 8000)
│   └── qdrant_setup.py # Index ADRs into Qdrant
├── fine-tuning/       # Model training
│   ├── training-data.jsonl
│   └── fine_tune_qwen3_8b.py
├── plugin-obsidian/  # Obsidian plugin (TypeScript)
└── data/              # Cloned reference data
    ├── adrs/          # Kubernetes, Django, Rust ADRs
    └── security/      # OWASP, Bandit examples
```

## Key Dependencies

| Component | Package | Purpose |
|-----------|---------|---------|
| Backend | `qdrant-client` | Vector DB |
| Backend | `sentence-transformers` | Embeddings |
| Backend | `transformers`, `peft`, `unsloth` | LLM fine-tuning |
| Plugin | Obsidian API | Plugin framework |

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/validate-adr` | Validate ADR for contradictions + risks |
| POST | `/index-adr` | Add ADR to Qdrant |

## Fine-tuning Notes

- **Model**: `Qwen/Qwen3-8B`
- **Method**: LoRA/QLoRA (4-bit quantization)
- **Hardware**: AMD MI300X (192GB HBM3) ideal, or any GPU 24GB+
- **Expected time**: ~1.5 hours on MI300X for 3 epochs
- **Output**: `models/qwen-adr-lora/`

## Qdrant Collection Schema

Collection: `adrs`
- Vector size: 1024 (Qwen3-embed-8b)
- Distance: COSINE
- Payload indexes: `status`, `category`, `title`

## Obsidian Plugin Development

```bash
cd plugin-obsidian
npm install
npm run build   # Compile TypeScript
```

Plugin location for manual install: copy to vault's `.obsidian/plugins/` folder.

## Important Constraints

- Qdrant must be running BEFORE starting the backend
- Fine-tuning script requires `rocm-smi` detection or CUDA fallback
- The plugin requires backend running at `http://localhost:8000` (configurable in settings)
