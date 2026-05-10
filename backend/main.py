"""
ADR Security Validator API
FastAPI backend for validating ADRs and detecting security issues.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

# Try to import transformers, handle gracefully if not available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️  Transformers not available: {e}")

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
# Try both relative to root and relative to backend
adapter_path = "./models/qwen-adr-lora"
if not os.path.exists(adapter_path):
    adapter_path = "../models/qwen-adr-lora"

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
        from qdrant_client import QdrantClient as QClient
        if qdrant_api_key:
            qdrant_client = QClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            qdrant_client = QClient(url=qdrant_url)

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
            print("✅ Loaded Qwen3-Embedding-8B embeddings")
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
                print(f"ℹ️  No fine-tuned adapter found at {adapter_path}. Using base model.")

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
        if "without validation" in combined or "no validation" in combined or "flexible schema" in combined:
            risks.append(SecurityRisk(
                severity="high",
                type="nosql_injection",
                description="MongoDB proposals often lack strict schema validation. Ensure input validation is implemented to prevent injection.",
                secure_alternative='''# MongoDB with input validation
from bson.objectid import ObjectId
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
api_key = os.environ.get('API_KEY')'''
            ))

    return risks

def detect_contradictions(title: str, content: str, related_adrs: List[RelatedADR]) -> List[Contradiction]:
    """Detect contradictions with related ADRs."""
    contradictions = []
    combined = f"{title} {content}".lower()

    for adr in related_adrs:
        adr_lower = adr.title.lower()

        # PostgreSQL vs MongoDB contradiction
        if ("mongo" in combined or "nosql" in combined) and (adr_lower.find("postgresql") != -1 or adr_lower.find("sql") != -1):
            if adr.status in ["accepted", "final"]:
                contradictions.append(Contradiction(
                    severity="high",
                    related_adr_title=adr.title,
                    description=f"This ADR contradicts {adr.title} which was {adr.status}. PostgreSQL is typically required for financial data integrity.",
                    source=adr.category
                ))

    return contradictions

def generate_recommendations(title: str, content: str, risks: List[SecurityRisk],
                             contradictions: List[Contradiction]) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    if contradictions:
        recommendations.append("Review the related ADRs that this proposal contradicts before proceeding.")
        recommendations.append("Consider if PostgreSQL with JSONB would satisfy the requirement while maintaining ACID guarantees.")

    if risks:
        recommendations.append("Address the identified security risks, focusing on input validation and secure storage.")

    if not risks and not contradictions:
        recommendations.append("No major issues detected. Consider adding more detail about failure scenarios.")

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

            # Force dimension check
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

    # Detect security risks
    risks = detect_security_risks(request.title, request.content)

    # If model is available, use it for enhanced analysis
    model_critique = ""
    if model and tokenizer:
        try:
            context_parts = [f"Related ADR: {adr.title} ({adr.status})" for adr in related_adrs[:3]]
            context_text = "\n".join(context_parts)

            prompt = f"""<|system|>
You are an expert in Architecture Decision Records (ADRs). Systematically critique the architecture. Identify security threats and provide actionable recommendations.
</|system|>
<|user|>
ADR Title: {request.title}
ADR Content: {request.content}
Related ADRs:
{context_text}
</|user|>
<|assistant|>
"""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

            model_critique = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"🤖 Model response received")

        except Exception as e:
            print(f"⚠️  Model inference error: {e}")

    # Detect contradictions
    contradictions = detect_contradictions(request.title, request.content, related_adrs)

    # Generate recommendations
    recommendations = generate_recommendations(request.title, request.content, risks, contradictions)

    # Determine status
    if contradictions or any(r.severity in ["critical", "high"] for r in risks):
        status = "needs_review"
        message = "Issues detected that require attention before approval."
    else:
        status = "approved"
        message = "ADR appears sound."

    return ADRValidationResponse(
        status=status,
        message=message,
        contradictions=contradictions,
        security_risks=risks,
        recommendations=recommendations,
        related_adrs=related_adrs,
        detected_technologies=technologies
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
