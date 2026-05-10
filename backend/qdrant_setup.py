"""
Qdrant Setup Script for ADR Validator - ROCm Optimized
"""

import json
import os
import uuid
import sys
from pathlib import Path

# --- ROCM COMPATIBILITY PATCHES (Python 3.12 + Torch 2.5) ---
import torch

# 1. Mock missing integer types expected by torchao/transformers
for i in range(1, 9):
    attr = f"int{i}"
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

# 2. Mock pytree register_constant if missing
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: x

# 3. Aggressive Torch schema inference patch
try:
    import torch._library.infer_schema
    _orig = torch._library.infer_schema.infer_schema
    def _patched(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except Exception as e:
            if "unsupported type torch.Tensor" in str(e):
                return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e
    torch._library.infer_schema.infer_schema = _patched
    print("🛠️  ROCm + Python 3.12 compatibility patches applied")
except:
    pass

# Disable torchao to avoid the int1/register_constant errors
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
# -----------------------------------------------------------

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

COLLECTION_NAME = "adrs"
VECTOR_SIZE = 4096  # Qwen3-Embedding-8B output size

def create_collection(client: QdrantClient):
    """Create Qdrant collection for ADRs."""
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Created collection '{COLLECTION_NAME}' with {VECTOR_SIZE}D vectors")
    return True

def extract_adrs_from_files(base_path: Path):
    """Extract ADRs from data folder."""
    adrs = []
    # Search in common locations
    data_path = base_path / "data" / "adrs"
    if not data_path.exists():
        data_path = Path("/root/AMD-Developer-Hackathon/data/adrs")

    for md_file in data_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            adrs.append({
                "title": md_file.stem.replace("-", " ").title(),
                "content": content[:4000],
                "status": "accepted",
                "category": md_file.parent.name,
                "source": str(md_file.relative_to(base_path))
            })
        except:
            continue
    return adrs

def index_adrs(client: QdrantClient, adrs: list, use_embedded_model: bool = False):
    """Index ADRs into Qdrant."""
    embedding_model = None
    if use_embedded_model:
        print("📦 Initializing semantic embedding model (Qwen3-Embedding-8B)...")
        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", trust_remote_code=True)
            model = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", 
                trust_remote_code=True, 
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager" # Avoid flash_attn conflicts
            )
            
            class SimpleEmbedder:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                def encode(self, sentences):
                    if isinstance(sentences, str): sentences = [sentences]
                    inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(self.model.device)
                    with torch.no_grad():
                        out = self.model(**inputs)
                    # Convert to float32 immediately to avoid NumPy/Qdrant compatibility issues
                    return out.last_hidden_state.mean(dim=1).to(torch.float32).cpu().numpy()
            
            embedding_model = SimpleEmbedder(model, tokenizer)
            print("✅ Semantic model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load semantic model: {e}")
            print("📦 Falling back to mock embeddings")

    points = []
    for i, adr in enumerate(adrs):
        text = f"{adr['title']}\n{adr['content']}"
        
        if embedding_model:
            vector = embedding_model.encode(text)[0].tolist()
        else:
            # Deterministic mock vector
            import hashlib, random
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(h); vector = [random.random() for _ in range(VECTOR_SIZE)]

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=adr
        ))

        if len(points) >= 50:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   Indexed {i+1}/{len(adrs)} ADRs...")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"\n✅ Finished indexing {len(adrs)} ADRs")

def setup_qdrant(use_embeddings: bool = False):
    print("🚀 Setting up Qdrant Vector DB\n")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    try:
        client.get_collections()
        create_collection(client)
        adrs = extract_adrs_from_files(Path(__file__).parent.parent)
        print(f"📂 Found {len(adrs)} documents")
        index_adrs(client, adrs, use_embedded_model=use_embeddings)
        return client
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None

if __name__ == "__main__":
    setup_qdrant(use_embeddings="--embed" in sys.argv)
