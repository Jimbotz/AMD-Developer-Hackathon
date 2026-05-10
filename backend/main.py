"""
ADR Security Validator API - STABLE VERSION
Optimized for AMD MI300X with Float32 Inference
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# --- GLOBAL ROCM & PY3.12 WORKAROUND ---
for _int_type in range(1, 9):
    _attr = f"int{_int_type}"
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.int8)

import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def _mock_register_constant(cls): return cls
    torch.utils._pytree.register_constant = _mock_register_constant

os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

try:
    import torch._library.infer_schema
    _original_infer_schema = torch._library.infer_schema.infer_schema
    def _patched_infer_schema(*args, **kwargs):
        try: return _original_infer_schema(*args, **kwargs)
        except ValueError as e:
            if "unsupported type torch.Tensor" in str(e):
                fn = args[0] if args else kwargs.get('fn')
                if fn and "grouped_mm_fallback" in str(fn):
                    return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e
    torch._library.infer_schema.infer_schema = _patched_infer_schema
except Exception: pass

from qdrant_client import QdrantClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ============== Models ==============

class ADRValidationRequest(BaseModel):
    title: str
    content: str
    context: Optional[str] = None

class SecurityRisk(BaseModel):
    severity: str
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
vector_db = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, vector_db, embedding_model
    print("Initializing API in Float32 Stability Mode...")

    # Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        vector_db = QdrantClient(url=qdrant_url)
        vector_db.get_collections()
    except Exception: vector_db = None

    # Embeddings
    try: embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    except Exception: embedding_model = None

    # LLM - FORCE FLOAT32 TO MATCH TRAINING
    try:
        base_model = "Qwen/Qwen3-8B"
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        adapter_path = "./models/qwen-adr-lora"
        if not os.path.exists(adapter_path):
            adapter_path = "../fine-tuning/models/qwen-adr-lora"
            
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Adapter loaded from {adapter_path}")
    except Exception as e: print(f"LLM Error: {e}")

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============== Detection Logic ==============

def detect_security_risks(title: str, content: str) -> List[SecurityRisk]:
    risks = []
    combined = f"{title} {content}".lower()
    if "mongodb" in combined and "validation" not in combined:
        risks.append(SecurityRisk(severity="high", type="nosql_injection", description="MongoDB lacks strict validation in this ADR.", secure_alternative="Use Pydantic or JSON Schema validation."))
    if "config.py" in combined or "password" in combined:
        risks.append(SecurityRisk(severity="critical", type="hardcoded_secrets", description="Potential secrets in code/config detected.", secure_alternative="Use AWS Secrets Manager or Environment Variables."))
    return risks

# ============== Endpoints ==============

@app.post("/validate-adr", response_model=ADRValidationResponse)
async def validate_adr(request: ADRValidationRequest):
    related_adrs = []
    if vector_db and embedding_model:
        try:
            vector = embedding_model.encode(f"{request.title} {request.content}").tolist()
            results = vector_db.search(collection_name="adrs", query_vector=vector, limit=10)
            for r in results:
                title = r.payload.get("title", "")
                # FILTRO DE RUIDO
                if any(kw in title.lower() for kw in ["meeting", "notes", "report", "annual", "agenda"]): continue
                related_adrs.append(RelatedADR(title=title, similarity=float(r.score), status=r.payload.get("status", "accepted"), category=r.payload.get("category", "general")))
        except Exception: pass

    model_critique = ""
    if model and tokenizer:
        try:
            prompt = f"<|system|>\nYou are a senior Architect. Review the ADR for security and consistency.\n<|user|>\nTitle: {request.title}\nContent: {request.content}\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.1, repetition_penalty=1.2)
            
            # Extract only response
            model_critique = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            # Clean Chinese/Tags
            for noise in ["助理", "Assistant", "<|", "|>"]: model_critique = model_critique.replace(noise, "")
        except Exception as e: model_critique = f"Inference error: {e}"

    risks = detect_security_risks(request.title, request.content)
    
    # Status Logic - More sensitive to architectural concerns
    model_lower = model_critique.lower()
    critical_keywords = [
        "risk", "vulnerability", "reject", "critical", "warning", 
        "insecure", "race condition", "overselling", "inconsistency",
        "trade-off", "oversell", "unacceptable", "flaw"
    ]
    
    has_issue = len(risks) > 0 or any(kw in model_lower for kw in critical_keywords)
    status = "needs_review" if has_issue else "approved"
    
    return ADRValidationResponse(
        status=status,
        message="Issues detected that require architectural review" if has_issue else "ADR appears sound",
        security_risks=risks,
        recommendations=[model_critique.strip()] if model_critique else [],
        related_adrs=related_adrs[:5]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
