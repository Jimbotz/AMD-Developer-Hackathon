# ADR Security Validator - Project Context

This project provides real-time validation for Architecture Decision Records (ADRs), featuring security analysis, contradiction detection with historical decisions, and secure code recommendations.

## Project Overview

- **Purpose**: Assist architects and developers in writing secure and consistent ADRs.
- **Architecture**:
    - **Backend**: FastAPI-based API that performs semantic search using Qdrant and deep analysis using a fine-tuned Qwen3-8B model.
    - **Vector Database**: Qdrant stores historical ADRs from major projects (Kubernetes, Django, Rust) to detect contradictions.
    - **Inference**: Uses HuggingFace Transformers with 4-bit quantization (BitsAndBytes) and LoRA adapters.
    - **Obsidian Plugin**: Provides a real-time interface within the Obsidian note-taking app.
    - **Fine-tuning**: Custom pipeline to generate training datasets from existing ADRs and security vulnerability patterns (OWASP, Bandit).

## Key Technologies

- **Backend**: Python, FastAPI, Pydantic, Uvicorn.
- **AI/ML**: Qwen3-8B, Sentence-Transformers (Qwen3-embed-8b), Unsloth (for fine-tuning), PEFT/LoRA.
- **Storage**: Qdrant (Vector Search).
- **Frontend**: TypeScript, Obsidian API, Esbuild.

## Building and Running

### 1. Prerequisites
- Docker (for Qdrant)
- Python 3.10+
- Node.js 18+ (for the Obsidian plugin)
- GPU with 24GB+ VRAM (for LLM inference/fine-tuning)

### 2. Backend & Qdrant
```bash
# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Setup Backend
cd backend
pip install -r requirements.txt
python qdrant_setup.py --embed  # Use --embed for semantic search

# Run API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Fine-tuning (Optional)
```bash
cd fine-tuning
python generate_dataset.py
python fine_tune_qwen3_8b.py
```

### 4. Obsidian Plugin
```bash
cd plugin-obsidian
npm install
npm run build
# Copy the folder to your vault's .obsidian/plugins/ directory
```

## Project Structure

- `backend/`: API server, Qdrant integration, and validation logic.
- `fine-tuning/`: Scripts for dataset generation and model training.
- `plugin-obsidian/`: Source code for the Obsidian plugin.
- `data/`: Collected ADRs and security examples used for indexing and training.
- `models/`: Storage for fine-tuned LoRA adapters.

## Development Conventions

- **API Design**: Uses FastAPI with Pydantic for strict typing. Endpoints include `/validate-adr` and `/index-adr`.
- **Validation Logic**: Combines rule-based detection (found in `backend/main.py`) with LLM-based analysis.
- **Embeddings**: Uses 1024-dimensional vectors (compatible with Qwen3-embed-8b).
- **Plugin**: Adheres to Obsidian's plugin architecture; uses `esbuild` for bundling and `eslint` for linting.
- **Hardware Optimization**: Inference uses 4-bit quantization to fit on consumer GPUs; fine-tuning is optimized for AMD ROCm (MI300X).

## TODO / Known Issues
- [ ] Implement unit tests for the backend validation logic.
- [ ] Add support for more ADR formats (currently optimized for Markdown and RST).
- [ ] Improve the "detected technologies" regex mapping in `backend/main.py`.
