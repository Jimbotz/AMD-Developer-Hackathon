"""
ADR Security Validator API
FastAPI backend for validating ADRs and detecting security issues.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- GLOBAL ROCM & PY3.12 WORKAROUND ---
# This must run before ANY transformers/torch imports
import torch

# 1. FIX: module 'torch' has no attribute 'int1'
for _int_type in range(1, 9):
    _attr = f"int{_int_type}"
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.int8)

# 2. FIX: torch.utils._pytree has no attribute 'register_constant'
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def _mock_register_constant(cls):
        return cls
    torch.utils._pytree.register_constant = _mock_register_constant

# 3. Disable torchao integration in transformers
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

# 4. Patch the specific bug in Torch 2.4/2.5 + Python 3.12 type hint registration
try:
    import torch._library.infer_schema
    _original_infer_schema = torch._library.infer_schema.infer_schema

    def _patched_infer_schema(*args, **kwargs):
        try:
            return _original_infer_schema(*args, **kwargs)
        except ValueError as e:
            if "unsupported type torch.Tensor" in str(e):
                fn = args[0] if args else kwargs.get('fn')
                if fn and "grouped_mm_fallback" in str(fn):
                    return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e

    torch._library.infer_schema.infer_schema = _patched_infer_schema
except Exception:
    pass

import typing
from typing import Union, List, Optional, Sequence
# ------------------------------

from qdrant_client import QdrantClient
import json
from pathlib import Path

# Check for ROCm + broken transformers custom op (grouped_mm_fallback type error)
def _is_rocm_transformers_broken():
    """Detect ROCm incompatibility with transformers custom op registration."""
    if not os.path.exists("/dev/kfd"):
        return False
    try:
        import torch
        import importlib
        m = importlib.import_module("transformers.modeling_utils")
        return False
    except ValueError as e:
        if "grouped_mm_fallback" in str(e):
            return True
    except Exception:
        pass
    return False

_ROCM_TRANSFORMERS_BROKEN = _is_rocm_transformers_broken()

# Try to import transformers, handle gracefully if not available
try:
    if _ROCM_TRANSFORMERS_BROKEN:
        raise ImportError("ROCm transformers custom op broken on this PyTorch build")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️  Transformers not available: {e}")
    print("   Run: pip install transformers (ROCm-compatible build required)")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not installed. Run: pip install sentence-transformers")

# ============== Pydantic Models ==============

class ADRValidationRequest(BaseModel):
    title: str
    content: str
    context: Optional[str] = None

class SecurityRisk(BaseModel):
    severity: str  # low, medium, high, critical
    type: str
    description: str
    code_example: Optional[str] = None
    secure_alternative: Optional[str] = None

class Contradiction(BaseModel):
    severity: str
    related_adr_title: str
    description: str
    source: Optional[str] = None

class RelatedADR(BaseModel):
    title: str
    similarity: float
    status: str
    category: str

class ADRValidationResponse(BaseModel):
    status: str
    message: str
    contradictions: List[Contradiction] = []
    security_risks: List[SecurityRisk] = []
    recommendations: List[str] = []
    related_adrs: List[RelatedADR] = []
    detected_technologies: List[str] = []

# ============== Global State ==============

model = None
tokenizer = None
qdrant_client = None
embedding_model = None
adapter_path = "./models/qwen-adr-lora"

# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global model, tokenizer, qdrant_client, embedding_model

    print("🔄 Initializing ADR Validator API...")

    # Initialize Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

    try:
        if qdrant_api_key:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            qdrant_client = QdrantClient(url=qdrant_url)

        # Test connection
        qdrant_client.get_collections()
        print(f"✅ Connected to Qdrant at {qdrant_url}")
    except Exception as e:
        print(f"⚠️  Qdrant not connected: {e}")
        qdrant_client = None

    # Initialize embedding model
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
            print("✅ Loaded Qwen3-embed-8b embeddings")
        except Exception as e:
            print(f"⚠️  Could not load embedding model: {e}")
            print("ℹ️  Semantic search will use mock vectors on this hardware")

    # Initialize LLM (if available)
    if TRANSFORMERS_AVAILABLE:
        try:
            print(f"🔄 Loading base model in native BF16...")

            base_model = "Qwen/Qwen3-8B"
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            if os.path.exists(adapter_path):
                print(f"🔄 Loading fine-tuned adapter from {adapter_path}...")
                model = PeftModel.from_pretrained(model, adapter_path)
                print("✅ Adapter loaded successfully")
            else:
                print("ℹ️  No fine-tuned adapter found. Using base model.")

            print("✅ Model ready")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            model = None
            tokenizer = None
    else:
        print("ℹ️  Transformers not available. Using rule-based validation.")

    print("🚀 ADR Validator API ready!")
    yield

# ============== FastAPI App ==============

app = FastAPI(
    title="ADR Security Validator API",
    description="Validates Architecture Decision Records for contradictions and security risks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["app://obsidian.md", "http://localhost", "https://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Detection Logic ==============

def detect_technologies(content: str) -> List[str]:
    """Detect mentioned technologies in the ADR."""
    tech_keywords = {
        "PostgreSQL": ["postgresql", "postgres"],
        "MongoDB": ["mongodb", "mongo"],
        "MySQL": ["mysql"],
        "Redis": ["redis"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Docker": ["docker", "container"],
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "GCP": ["gcp", "google cloud"],
        "Python": ["python"],
        "JavaScript": ["javascript", "node.js", "nodejs"],
        "Go": ["golang", " go "],
        "Rust": ["rust"],
        "GraphQL": ["graphql"],
        "REST": ["rest api", "restful"],
        "gRPC": ["grpc"],
        "Terraform": ["terraform"],
        "Kafka": ["kafka"],
        "RabbitMQ": ["rabbitmq"],
    }

    content_lower = content.lower()
    detected = []

    for tech, keywords in tech_keywords.items():
        if any(kw in content_lower for kw in keywords):
            detected.append(tech)

    return detected

def detect_security_risks(title: str, content: str) -> List[SecurityRisk]:
    """Rule-based security risk detection."""
    risks = []
    combined = f"{title} {content}".lower()

    # NoSQL Injection
    if any(kw in combined for kw in ["mongodb", "mongo"]):
        if "without validation" in combined or "no validation" in combined:
            risks.append(SecurityRisk(
                severity="high",
                type="nosql_injection",
                description="MongoDB proposals should include input validation to prevent NoSQL injection attacks.",
                secure_alternative='''# MongoDB with input validation
from bson.objectid import ObjectId
from pymongo import MongoClient

def get_user(user_id: str):
    if not ObjectId.is_valid(user_id):
        raise ValueError("Invalid ID format")
    return db.users.find_one({"_id": ObjectId(user_id)})'''
            ))

    # SQL Injection patterns
    if "postgresql" in combined or "mysql" in combined or "database" in combined:
        if any(kw in combined for kw in ["string interpolation", "f-string query", "concat sql"]):
            risks.append(SecurityRisk(
                severity="critical",
                type="sql_injection",
                description="SQL queries should use parameterized queries, not string concatenation.",
                secure_alternative='''# Parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'''
            ))

    # Secrets in plaintext
    if any(kw in combined for kw in ["password", "secret", "api key", "credential"]):
        if "plaintext" in combined or "hardcoded" in combined or "in code" in combined:
            risks.append(SecurityRisk(
                severity="high",
                type="hardcoded_credentials",
                description="Credentials should be stored in secure vaults or environment variables, never in code.",
                secure_alternative='''# Use environment variables or secrets manager
import os
from keyring import get_password

api_key = os.environ.get('API_KEY') or get_password('myapp', 'api_key')'''
            ))

    # Encryption
    if "encryption" in combined or "encrypt" in combined:
        if "no encryption" in combined or "without encryption" in combined or "unencrypted" in combined:
            if "at rest" in combined or "storage" in combined:
                risks.append(SecurityRisk(
                    severity="high",
                    type="unencrypted_data",
                    description="Data at rest should be encrypted to protect against data breaches.",
                    secure_alternative='''# Encrypt data at rest
from cryptography.fernet import Fernet
cipher = Fernet(key)
encrypted_data = cipher.encrypt(data)'''
                ))

    # JWT without verification
    if "jwt" in combined:
        if "without verification" in combined or "no verification" in combined:
            risks.append(SecurityRisk(
                severity="medium",
                type="jwt_unverified",
                description="JWT tokens must always be verified for signature, expiration, and claims.",
                secure_alternative='''# Verify JWT
import jwt
decoded = jwt.verify(token, key, algorithms=["HS256"])'''
            ))

    return risks

def detect_contradictions(title: str, content: str, related_adrs: List[RelatedADR]) -> List[Contradiction]:
    """Detect contradictions with related ADRs."""
    contradictions = []
    combined = f"{title} {content}".lower()

    for adr in related_adrs:
        adr_lower = adr.title.lower()

        # PostgreSQL vs MongoDB contradiction
        if ("mongo" in combined or "nosql" in combined) and adr.title.lower().find("postgresql") != -1:
            if adr.status == "accepted":
                contradictions.append(Contradiction(
                    severity="high",
                    related_adr_title=adr.title,
                    description=f"This ADR contradicts {adr.title} which was accepted. PostgreSQL provides ACID guarantees required for financial transactions.",
                    source="django/final"
                ))

        # Docker vs Kubernetes
        if "kubernetes" in combined and ("docker" in adr_lower or "container") in adr_lower:
            contradictions.append(Contradiction(
                severity="low",
                related_adr_title=adr.title,
                description=f"May conflict with container-related decisions in {adr.title}.",
                source=adr.category
            ))

    return contradictions

def generate_recommendations(title: str, content: str, risks: List[SecurityRisk],
                             contradictions: List[Contradiction]) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    if contradictions:
        recommendations.append("Review the related ADRs that this proposal contradicts before proceeding.")
        recommendations.append("If the new use case requires different technology, create a superseding ADR explaining the rationale.")

    if any(r.severity == "critical" for r in risks):
        recommendations.append("Address critical security issues before implementing this architecture.")

    if any(r.type == "nosql_injection" for r in risks):
        recommendations.append("Implement input validation and use MongoDB's built-in security features.")

    if any(r.type == "sql_injection" for r in risks):
        recommendations.append("Use parameterized queries or an ORM to prevent SQL injection attacks.")

    if not risks and not contradictions:
        recommendations.append("No major issues detected. Consider adding more detail about failure scenarios and rollback plans.")

    return recommendations

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {"message": "ADR Security Validator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "qdrant_connected": qdrant_client is not None,
        "embeddings_loaded": embedding_model is not None
    }

@app.post("/validate-adr", response_model=ADRValidationResponse)
async def validate_adr(request: ADRValidationRequest):
    """Validate an ADR for contradictions and security risks."""

    # Search for related ADRs in Qdrant
    related_adrs = []
    if qdrant_client:
        try:
            query_text = f"{request.title} {request.content}"
            if embedding_model:
                vector = embedding_model.encode(query_text).tolist()
            else:
                import hashlib
                import random
                hash_val = int(hashlib.md5(query_text.encode()).hexdigest()[:8], 16)
                random.seed(hash_val % (2**32))
                vector = [random.random() for _ in range(4096)]

            results = qdrant_client.search(
                collection_name="adrs",
                query_vector=vector,
                limit=5,
                with_payload=True
            )

            for r in results:
                related_adrs.append(RelatedADR(
                    title=r.payload.get("title", ""),
                    similarity=float(r.score),
                    status=r.payload.get("status", ""),
                    category=r.payload.get("category", "")
                ))
        except Exception as e:
            print(f"⚠️  Qdrant search error: {e}")

    # Detect technologies
    technologies = detect_technologies(request.content)

    # Detect security risks (rule-based + model if available)
    risks = detect_security_risks(request.title, request.content)

    # If model is available, use it for enhanced analysis
    if model and tokenizer:
        try:
            # Prepare context from related ADRs
            context_parts = [f"Related ADR: {adr.title} ({adr.status})" for adr in related_adrs[:3]]
            context_text = "\n".join(context_parts)

            prompt = f"""<|system|>
You are an expert in Architecture Decision Records (ADRs). Systematically critique the architecture using AWS/Cloud Well-Architected Framework principles and identify security threats using the STRIDE methodology (Spoofing, Tampering, Repudiation, Info Disclosure, DoS, EoP). Provide actionable recommendations.
</|system|>

<|user|>
ADR Title: {request.title}

ADR Content: {request.content}

Related ADRs from history:
{context_text}

Context: {request.context or "No additional context"}
</|user|>

<|assistant|>
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True
                )

            model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"🤖 Model response received (length: {len(model_response)})")

        except Exception as e:
            print(f"⚠️  Model inference error: {e}")

    # Detect contradictions
    contradictions = detect_contradictions(request.title, request.content, related_adrs)

    # Generate recommendations
    recommendations = generate_recommendations(
        request.title, request.content, risks, contradictions
    )

    # Determine status
    if contradictions or any(r.severity in ["critical", "high"] for r in risks):
        status = "needs_review"
        message = "Issues detected that require attention before approval."
    elif risks:
        status = "needs_minor_revision"
        message = "Minor issues found. Consider addressing them."
    else:
        status = "approved"
        message = "ADR appears sound. No major issues detected."

    return ADRValidationResponse(
        status=status,
        message=message,
        contradictions=contradictions,
        security_risks=risks,
        recommendations=recommendations,
        related_adrs=related_adrs,
        detected_technologies=technologies
    )

@app.post("/index-adr")
async def index_adr(title: str, content: str, status: str = "proposed",
                   category: str = "general", source: str = "manual"):
    """Index a new ADR in Qdrant."""

    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not connected")

    try:
        import uuid

        if embedding_model:
            vector = embedding_model.encode(f"{title}\n{content}").tolist()
        else:
            import hashlib, random
            hash_val = int(hashlib.md5(f"{title}{content}".encode()).hexdigest()[:8], 16)
            random.seed(hash_val % (2**32))
            vector = [random.random() for _ in range(4096)]

        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "title": title,
                "content": content,
                "status": status,
                "category": category,
                "source": source
            }
        )

        qdrant_client.upsert(collection_name="adrs", points=[point])

        return {"status": "success", "message": f"ADR '{title}' indexed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)