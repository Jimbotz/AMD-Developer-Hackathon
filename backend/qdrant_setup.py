"""
Qdrant Setup Script for ADR Validator
Creates collection, indexes documents, and provides search functionality.
"""

import json
import os
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

COLLECTION_NAME = "adrs"
VECTOR_SIZE = 1024  # Qwen3-embed-8b embedding size

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

    # Create payload indexes for filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="status",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title",
        field_schema="text"
    )

    print(f"✅ Created collection '{COLLECTION_NAME}' with {VECTOR_SIZE}D vectors")
    return True

def extract_adrs_from_files(base_path: Path):
    """Extract ADRs from downloaded repositories."""
    adrs = []

    # Kubernetes ADRs
    k8s_dir = base_path / "data" / "adrs" / "kubernetes" / "sig-architecture"
    if k8s_dir.exists():
        for md_file in k8s_dir.rglob("*.md"):
            if "recommendations" in str(md_file) or md_file.name in ["production-readiness.md", "api-review-process.md"]:
                content = md_file.read_text(encoding='utf-8')
                adrs.append({
                    "title": md_file.stem.replace("-", " ").title(),
                    "content": content[:3000],
                    "status": "accepted",
                    "category": "kubernetes",
                    "source": "kubernetes/sig-architecture"
                })

    # Django DEPs
    django_dir = base_path / "data" / "adrs" / "django"
    if django_dir.exists():
        for status_dir in django_dir.iterdir():
            if status_dir.is_dir() and status_dir.name in ['final', 'accepted', 'draft', 'rejected']:
                for rst_file in status_dir.rglob("*.rst"):
                    content = rst_file.read_text(encoding='utf-8')[:3000]
                    adrs.append({
                        "title": rst_file.stem.replace("-", " ").title(),
                        "content": content,
                        "status": status_dir.name,
                        "category": "django",
                        "source": f"django/{status_dir.name}"
                    })

    # Plantillas ADR examples
    plantillas_dir = base_path / "data" / "adrs" / "plantillas" / "locales" / "tr" / "examples"
    if plantillas_dir.exists():
        for md_file in plantillas_dir.rglob("index.md"):
            content = md_file.read_text(encoding='utf-8')[:3000]
            adrs.append({
                "title": md_file.parent.name.replace("-", " ").title(),
                "content": content,
                "status": "example",
                "category": "template",
                "source": "plantillas/examples"
            })

    # Cloud Architecture ADRs
    cloud_arch_dir = base_path / "data" / "adrs" / "cloud-architecture"
    if cloud_arch_dir.exists():
        for md_file in cloud_arch_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            status = "accepted"
            if "REJECTED" in content or "rejected" in md_file.name.lower():
                status = "rejected"
            adrs.append({
                "title": md_file.stem.replace("-", " ").title(),
                "content": content[:3000],
                "status": status,
                "category": "cloud-architecture",
                "source": "internal/cloud-architecture"
            })

    # Threat Modeling ADRs
    threat_model_dir = base_path / "data" / "adrs" / "threat-modeling"
    if threat_model_dir.exists():
        for md_file in threat_model_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            adrs.append({
                "title": md_file.stem.replace("-", " ").title(),
                "content": content[:3000],
                "status": "reference",
                "category": "threat-modeling",
                "source": "internal/threat-modeling"
            })

    return adrs

def generate_mock_embedding(text: str) -> list:
    """Generate a mock embedding vector for demo purposes.

    NOTE: In production, use Qwen3-embed-8b or similar model.
    For now, we generate a deterministic pseudo-embedding based on text.
    """
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    import random
    random.seed(hash_val % (2**32))
    return [random.random() for _ in range(VECTOR_SIZE)]

def index_adrs(client: QdrantClient, adrs: list, use_embedded_model: bool = False):
    """Index ADRs into Qdrant collection."""
    embedding_model = None
    if use_embedded_model:
        try:
            # Workaround for Python 3.12 + Transformers 4.45+ type hint bug on ROCm
            from typing import Union, List, Optional, Sequence
            import torch
            from sentence_transformers import SentenceTransformer
            
            embedding_model = SentenceTransformer("Qwen/Qwen3-embed-8b")
            print("📦 Using Qwen3-embed-8b for embeddings")
        except Exception as e:
            print(f"⚠️  Could not load Qwen3-embed-8b: {e}")
            print("📦 Using mock embeddings (not semantic)")

    points = []
    for adr in adrs:
        text_to_embed = f"{adr['title']}\n{adr['content']}"

        if embedding_model:
            vector = embedding_model.encode(text_to_embed).tolist()
        else:
            vector = generate_mock_embedding(text_to_embed)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "title": adr["title"],
                "content": adr["content"],
                "status": adr["status"],
                "category": adr["category"],
                "source": adr.get("source", "")
            }
        )
        points.append(point)

        if len(points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  Indexed {len(points)} ADRs...")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Indexed {len(points)} ADRs...")

    print(f"\n✅ Total ADRs indexed: {len(adrs)}")
    return len(adrs)

def search_adrs(client: QdrantClient, query: str, category: str = None,
                status: str = None, limit: int = 5, use_embedded_model: bool = False):
    """Search ADRs in Qdrant."""
    embedding_model = None
    if use_embedded_model:
        try:
            # Workaround for Python 3.12 + Transformers 4.45+ type hint bug on ROCm
            from typing import Union, List, Optional, Sequence
            import torch
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("Qwen/Qwen3-embed-8b")
        except Exception:
            pass

    if embedding_model:
        vector = embedding_model.encode(query).tolist()
    else:
        vector = generate_mock_embedding(query)

    filters = []
    if category:
        filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if status:
        filters.append(FieldCondition(key="status", match=MatchValue(value=status)))

    search_filter = Filter(must=filters) if filters else None

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )

    return results

def get_related_adrs(client: QdrantClient, adr_content: str, limit: int = 3):
    """Find related ADRs based on content similarity."""
    return search_adrs(client, adr_content, limit=limit)

def setup_qdrant(base_path: Path = None, use_embeddings: bool = False):
    """Main setup function."""
    if base_path is None:
        base_path = Path(__file__).parent.parent

    print("🚀 Setting up Qdrant for ADR Validator\n")

    # Connect to Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

    try:
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)

        # Check connection
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant at {qdrant_url}")
        print(f"📚 Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"❌ Could not connect to Qdrant: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return None

    # Create collection
    create_collection(client)

    # Extract ADRs from files
    print("\n📂 Extracting ADRs from repositories...")
    adrs = extract_adrs_from_files(base_path)
    print(f"   Found {len(adrs)} ADRs to index")

    # Index ADRs
    print("\n💾 Indexing ADRs...")
    index_adrs(client, adrs, use_embedded_model=use_embeddings)

    return client

def validate_setup():
    """Validate Qdrant setup is working."""
    import os
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    try:
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()

        # Check ADR collection
        try:
            count = client.count(collection_name=COLLECTION_NAME)
            print(f"✅ Qdrant validation successful")
            print(f"   Collection '{COLLECTION_NAME}': {count.count} points")
            return True
        except:
            print(f"⚠️  Collection '{COLLECTION_NAME}' not found")
            return False
    except Exception as e:
        print(f"❌ Qdrant validation failed: {e}")
        return False

if __name__ == "__main__":
    import sys

    use_embeddings = "--embed" in sys.argv
    client = setup_qdrant(use_embeddings=use_embeddings)

    if client:
        print("\n" + "="*50)
        print("💡 To start Qdrant with Docker:")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print("\n💡 To use semantic embeddings:")
        print("   python qdrant_setup.py --embed")
        print("="*50)