"""
Generate fine-tuning dataset for ADR Security Validator.
Creates JSONL file with training examples in chat format.
"""

import json
import os
import re
from pathlib import Path

def extract_adr_content(file_path):
    """Extract title and content from ADR files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else Path(file_path).stem

        status_match = re.search(r'\*\*Status\*\*:\s*(\w+)', content)
        status = status_match.group(1) if status_match else "unknown"

        return {
            "title": title,
            "content": content[:2000],
            "status": status,
            "source": str(file_path)
        }
    except Exception as e:
        return None

def process_kubernetes_adrs(base_path):
    """Process Kubernetes ADRs."""
    adrs = []
    arch_dir = base_path / "data" / "adrs" / "kubernetes" / "sig-architecture"
    if arch_dir.exists():
        for md_file in arch_dir.rglob("*.md"):
            adr = extract_adr_content(md_file)
            if adr:
                adr["category"] = "kubernetes"
                adrs.append(adr)
    return adrs

def process_django_deps(base_path):
    """Process Django DEPs."""
    adrs = []
    django_dir = base_path / "data" / "adrs" / "django"
    if django_dir.exists():
        for status_dir in django_dir.iterdir():
            if status_dir.is_dir() and status_dir.name in ['final', 'accepted', 'draft', 'rejected']:
                for rst_file in status_dir.rglob("*.rst"):
                    content = rst_file.read_text(encoding='utf-8')[:2000]
                    adrs.append({
                        "title": rst_file.stem,
                        "content": content,
                        "status": status_dir.name,
                        "source": str(rst_file),
                        "category": "django"
                    })
    return adrs

def create_fine_tuning_dataset(base_path, output_path):
    """Create JSONL dataset for fine-tuning Qwen3-8B."""

    kubernetes_adrs = process_kubernetes_adrs(base_path)
    django_adrs = process_django_deps(base_path)
    all_adrs = kubernetes_adrs + django_adrs

    print(f"Collected {len(kubernetes_adrs)} Kubernetes ADRs")
    print(f"Collected {len(django_adrs)} Django DEPs")

    training_data = []

    # Example 1: ADR contradiction detection
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are an expert in Architecture Decision Records (ADRs). You analyze proposed ADRs, detect contradictions with historical decisions, identify security risks, and provide recommendations with code examples when applicable."},
            {"role": "user", "content": "Analyze this proposed ADR:\n\nTitle: Migrate billing module from PostgreSQL to MongoDB\n\nContent:\nWe propose migrating our billing database from PostgreSQL to MongoDB to improve scalability for high-volume transaction processing. MongoDB's flexible schema will allow faster iteration on billing formats.\n\nHistorical ADRs:\n- ADR-042: PostgreSQL for all financial modules due to ACID requirements and audit compliance\n- ADR-038: NoSQL databases rejected for transaction-heavy workloads\n\nDoes this contradict any previous decisions? What are the security risks?"},
            {"role": "assistant", "content": "## Analysis\n\n### CONTRADICTION DETECTED\n\nADR-042 explicitly states: PostgreSQL for all financial modules due to ACID requirements and audit compliance.\n\nMigrating to MongoDB directly contradicts ADR-042 because:\n1. MongoDB lacks true ACID transactions across multiple documents\n2. Financial module requires audit compliance (PostgreSQL audit features)\n3. ADR-038 specifically rejected NoSQL for transaction-heavy workloads\n\n### SECURITY RISKS\n\n1. NoSQL Injection: MongoDB is vulnerable to injection attacks if user input is not properly validated.\n\n2. Data Exposure: Flexible schema makes it easier to accidentally expose sensitive fields.\n\n3. Audit Trail Gaps: MongoDB's native audit capabilities are limited compared to PostgreSQL.\n\n### SECURE CODE EXAMPLE\n\nSECURE: MongoDB with proper input validation for billing\n\n from pymongo import MongoClient\n from bson.objectid import ObjectId\n from bson.json_util import dumps\n import re\n\n def get_billing_record(billing_id: str, user_id: str):\n     # Validate input formats\n     if not ObjectId.is_valid(billing_id):\n         raise ValueError(\"Invalid billing ID format\")\n     if not re.match(r'^[a-zA-Z0-9-]+$', user_id):\n         raise ValueError(\"Invalid user ID format\")\n     # Use parameterized queries\n     query = {\n         \"_id\": ObjectId(billing_id),\n         \"user_id\": user_id,\n         \"status\": {\"$in\": [\"active\", \"pending\"]}\n     }\n     return collection.find_one(query)\n\nINSECURE - DO NOT USE:\n collection.find_one({\"_id\": billing_id})  # Direct injection"}
        ]
    })

    # Example 2: Security vulnerability detection
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a security expert specializing in code analysis. You identify vulnerabilities and provide secure alternatives."},
            {"role": "user", "content": "Analyze this code snippet for security vulnerabilities:\n\nAPI endpoint to fetch user data:\n\n query = f\"SELECT * FROM users WHERE id = '{user_id}'\"\n cursor.execute(query)\n\nLoad configuration:\n\n import yaml\n config = yaml.load(open('config.yaml'))"},
            {"role": "assistant", "content": "## Security Analysis\n\n### CRITICAL: SQL Injection\nLocation: Line 1\n\nThe code uses string interpolation to build SQL queries, allowing injection attacks.\n\n### CRITICAL: YAML Deserialization\nLocation: Line 5\n\nyaml.load() without a safe Loader can execute arbitrary code.\n\n### WARNING: Credential Exposure\nLocation: Line 5\n\nLoading config file directly may expose sensitive credentials.\n\n### SECURE ALTERNATIVES\n\nSECURE: Parameterized queries\n\n def get_user_data(user_id):\n     query = \"SELECT * FROM users WHERE id = %s\"\n     cursor.execute(query, (user_id,))  # Parameterized\n     return cursor.fetchone()\n\nSECURE: Safe YAML loading\n\n import yaml\n with open('config.yaml') as f:\n     config = yaml.safe_load(f)  # Uses SafeLoader"}
        ]
    })

    # Example 3: ADR with security concern
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are an expert in Architecture Decision Records. Validate ADRs, detect contradictions, and identify security risks."},
            {"role": "user", "content": "Validate this ADR proposal:\n\nTitle: Use Redis for session storage without encryption\n\nContent:\nWe will store user sessions in Redis for performance. Sessions will contain user_id, role, and preferences. No encryption at rest since Redis has built-in security.\n\nShould we proceed with this decision?"},
            {"role": "assistant", "content": "## ADR Validation\n\n### SECURITY RISK IDENTIFIED\n\nUnencrypted session data poses significant risks:\n\n1. Data Breach: If Redis is compromised, attacker gains access to all user sessions\n2. Session Hijacking: Unencrypted session IDs can be stolen and reused\n3. Compliance Violation: GDPR, CCPA may require encryption of PII\n4. Memory Dump Attacks: Sessions in RAM without encryption are vulnerable\n\n### RECOMMENDATIONS\n\nIf proceeding with Redis, implement:\n\nSECURE: Encrypted session storage\n\n from cryptography.fernet import Fernet\n import json\n import redis\n\n r = redis.Redis(host='localhost', port=6379)\n fernet = Fernet(fernet_key_from_kms)  # Key from KMS\n\n def store_session(session_id: str, user_data: dict):\n     encrypted = fernet.encrypt(json.dumps(user_data).encode())\n     r.setex(f\"session:{session_id}\", 3600, encrypted)\n\n def get_session(session_id: str) -> dict:\n     encrypted = r.get(f\"session:{session_id}\")\n     if encrypted:\n         return json.loads(fernet.decrypt(encrypted).decode())\n     return None\n\nAlso implement:\n- TLS for transit\n- Redis ACLs\n- Session rotation\n- Rate limiting\n\nTRADE-OFF ANALYSIS:\n\n| Approach | Security | Performance | Complexity |\n|----------|----------|-------------|------------|\n| Unencrypted | LOW | HIGH | LOW |\n| Encrypted (Fernet) | MEDIUM | MEDIUM | MEDIUM |\n| HSM-backed | HIGH | LOW | HIGH |\n\nRecommendation: Use encrypted Redis with Fernet. Performance impact is minimal (~1ms overhead) and security is significantly improved."}
        ]
    })

    # Add more training examples from collected ADRs
    for i, adr in enumerate(all_adrs[:20]):
        training_data.append({
            "messages": [
                {"role": "system", "content": "You are an expert in Architecture Decision Records. Analyze and provide insights."},
                {"role": "user", "content": f"Provide a brief analysis of this ADR:\n\nTitle: {adr['title']}\n\nContent excerpt: {adr['content'][:500]}\n\nStatus: {adr['status']}\nCategory: {adr['category']}"},
                {"role": "assistant", "content": f"## ADR Analysis\n\nTitle: {adr['title']}\nStatus: {adr['status']}\nSource: {adr['category']}\n\n### Key Points\n- This ADR is in **{adr['status']}** status\n- Category: {adr['category']} architecture decisions\n\n### Context\nThis decision follows standard ADR practices as seen in {adr['category']} projects.\n\nNote: For comprehensive analysis, more context about the surrounding technical landscape would be needed."}
            ]
        })

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nDataset created: {len(training_data)} training examples")
    print(f"Output: {output_path}")

    return training_data

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    output = base / "fine-tuning" / "training-data.jsonl"
    create_fine_tuning_dataset(base, output)