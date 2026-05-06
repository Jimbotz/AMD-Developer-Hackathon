# ADR Security Validator

Validates Architecture Decision Records (ADRs) in real-time with security analysis powered by Qwen3-8B and Qdrant vector database.

## Project Structure

```
adr-validator-hackathon/
├── data/
│   ├── adrs/                    # Cloned ADR repositories
│   │   ├── kubernetes/          # Kubernetes sig-architecture ADRs
│   │   ├── django/              # Django Enhancement Proposals
│   │   ├── rust-lang/          # Rust RFCs
│   │   └── plantillas/          # ADR templates
│   └── security/                # Security code examples
│       ├── bandit/              # Bandit security examples
│       └── owasp/               # OWASP examples
├── fine-tuning/                 # Fine-tuning scripts
│   ├── generate_dataset.py      # Generate training data
│   ├── fine_tune_qwen3_8b.py   # Fine-tune Qwen3-8B
│   └── training-data.jsonl      # Training dataset
├── backend/                     # FastAPI backend
│   ├── main.py                  # API server
│   ├── qdrant_setup.py          # Qdrant setup script
│   └── requirements.txt
├── plugin-obsidian/             # Obsidian plugin
│   ├── main.ts
│   ├── manifest.json
│   ├── styles.css
│   ├── package.json
│   └── src/
│       ├── api.ts
│       ├── validatorModal.ts
│       └── settingsTab.ts
└── docs/                        # Documentation
    └── RECURSOS_DESCARGADOS.md
```

## Quick Start

### 1. Setup Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Index ADRs in Qdrant

```bash
cd backend
pip install -r requirements.txt
python qdrant_setup.py
```

### 3. Fine-tune Model (Optional - recommended for better results)

```bash
cd fine-tuning
pip install -r requirements.txt  # or backend/requirements.txt
python fine_tune_qwen3_8b.py
```

### 4. Start Backend API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Install Obsidian Plugin

1. Copy `plugin-obsidian/` to your Obsidian vault's `.obsidian/plugins/` folder
2. Enable the plugin in Obsidian settings
3. Configure the API endpoint (default: `http://localhost:8000`)

## Usage

### Validate an ADR

```bash
curl -X POST http://localhost:8000/validate-adr \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Use PostgreSQL for billing",
    "content": "We will use PostgreSQL for the billing module due to ACID requirements."
  }'
```

### Index an ADR

```bash
curl -X POST "http://localhost:8000/index-adr?title=ADR-042&content=..." \
  -H "Content-Type: application/json"
```

## Features

- **Contradiction Detection**: Detects ADRs that contradict historical decisions
- **Security Analysis**: Identifies security risks in proposed architectures
- **Code Examples**: Generates secure code alternatives
- **Semantic Search**: Uses Qdrant for finding related ADRs
- **Real-time Validation**: Validates as you type in Obsidian

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/validate-adr` | Validate an ADR |
| POST | `/index-adr` | Index a new ADR |

## Fine-tuning Dataset

The training dataset (`fine-tuning/training-data.jsonl`) includes:

- Examples of ADR contradiction detection
- Security vulnerability identification
- Secure code alternatives
- Architecture decision analysis

Generated from:
- Kubernetes sig-architecture ADRs
- Django DEPs
- OWASP security examples
- Bandit vulnerability patterns

## Hardware Requirements

- **AMD MI300X**: 192GB HBM3 (recommended for fine-tuning)
- **Alternative**: Any GPU with 24GB+ VRAM for inference only

## Software Requirements

- Python 3.10+
- Docker (for Qdrant)
- Node.js 18+ (for Obsidian plugin development)