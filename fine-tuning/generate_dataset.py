"""
Enhanced Fine-tuning Dataset Generator for ADR Security Validator.
Sweeps all available data sources and generates a high-volume, diverse training dataset.
"""

import json
import os
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_markdown_content(file_path):
    """Extract title and content from Markdown files."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Try to find a H1 title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        # Basic cleanup: remove very short files
        if len(content.strip()) < 100:
            return None

        return {
            "title": title,
            "content": content[:4000], # Allow more context for large GPUs
            "source_path": str(file_path)
        }
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return None

def extract_rst_content(file_path):
    """Extract title and content from RST files (common in Django)."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # In RST, the title is usually underlined with === or ---
        lines = content.splitlines()
        title = file_path.stem
        for i in range(len(lines) - 1):
            if i < len(lines) - 1 and len(lines[i+1]) > 3 and all(c == '=' for c in lines[i+1]):
                title = lines[i].strip()
                break
        
        if len(content.strip()) < 100:
            return None

        return {
            "title": title,
            "content": content[:4000],
            "source_path": str(file_path)
        }
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return None

def discover_all_adrs(base_path):
    """Scan all data directories for ADR-like content."""
    all_data = []
    data_root = base_path / "data" / "adrs"
    
    if not data_root.exists():
        logging.error(f"Data root {data_root} not found!")
        return []

    # 1. Kubernetes (Markdown) - Scan ALL sub-sigs, not just architecture
    k8s_dir = data_root / "kubernetes"
    if k8s_dir.exists():
        logging.info("Scanning Kubernetes ADRs...")
        count = 0
        # Search in all sig-* directories
        for md_file in k8s_dir.rglob("*.md"):
            # Skip common non-ADR files
            if md_file.name.lower() in ['readme.md', 'contributing.md', 'license', 'code-of-conduct.md']:
                continue
            
            item = extract_markdown_content(md_file)
            if item:
                item["category"] = "kubernetes"
                # Determine sub-category from folder name (e.g., sig-auth)
                parts = md_file.parts
                for p in parts:
                    if p.startswith("sig-") or p.startswith("wg-"):
                        item["sub_category"] = p
                        break
                else:
                    item["sub_category"] = "general"
                
                all_data.append(item)
                count += 1
        logging.info(f"Found {count} Kubernetes documents.")

    # 2. Django (RST)
    django_dir = data_root / "django"
    if django_dir.exists():
        logging.info("Scanning Django DEPs...")
        count = 0
        for rst_file in django_dir.rglob("*.rst"):
            if rst_file.name.lower() == 'readme.rst': continue
            item = extract_rst_content(rst_file)
            if item:
                item["category"] = "django"
                item["sub_category"] = "web-framework"
                all_data.append(item)
                count += 1
        logging.info(f"Found {count} Django DEPs.")

    # 3. Rust (Markdown)
    rust_dir = data_root / "rust-lang" / "text"
    if rust_dir.exists():
        logging.info("Scanning Rust RFCs...")
        count = 0
        for md_file in rust_dir.glob("*.md"):
            if md_file.name == '0000-template.md': continue
            item = extract_markdown_content(md_file)
            if item:
                item["category"] = "rust"
                item["sub_category"] = "language-design"
                all_data.append(item)
                count += 1
        logging.info(f"Found {count} Rust RFCs.")

    # 4. Cloud Architecture & Threat Modeling
    others = ["cloud-architecture", "threat-modeling"]
    for other in others:
        other_dir = data_root / other
        if other_dir.exists():
            logging.info(f"Scanning {other}...")
            count = 0
            for md_file in other_dir.rglob("*.md"):
                item = extract_markdown_content(md_file)
                if item:
                    item["category"] = other
                    item["sub_category"] = "reference"
                    all_data.append(item)
                    count += 1
            logging.info(f"Found {count} {other} documents.")

    return all_data

def generate_diverse_samples(item):
    """Generate multiple training pairs from a single document to augment the dataset."""
    samples = []
    
    # Template A: Brief Analysis
    samples.append({
        "messages": [
            {"role": "system", "content": "You are an expert in Architecture Decision Records. Analyze and provide technical insights."},
            {"role": "user", "content": f"Provide a technical analysis of this architecture document:\n\nTitle: {item['title']}\n\nContent:\n{item['content'][:2000]}"},
            {"role": "assistant", "content": f"## Analysis: {item['title']}\n\n**Category:** {item['category']} ({item['sub_category']})\n\n### Summary\nThis document outlines a significant technical decision within the {item['category']} ecosystem. It addresses design patterns and implementation details relevant to {item['sub_category']}.\n\n### Architectural Significance\nFollowing established patterns in {item['category']}, this decision emphasizes consistency and long-term maintainability. Projects adopting similar approaches should evaluate the trade-offs regarding complexity and performance specific to this domain."}
        ]
    })
    
    # Template B: Security Perspective (If applicable)
    samples.append({
        "messages": [
            {"role": "system", "content": "You are a security-focused Software Architect. Evaluate documents for security implications and consistency."},
            {"role": "user", "content": f"Review this {item['category']} document for any security considerations or architectural risks:\n\nTitle: {item['title']}\n\nContent:\n{item['content'][:2000]}"},
            {"role": "assistant", "content": f"## Security Review: {item['title']}\n\nFrom a security perspective, this {item['category']} decision should be implemented following the principle of least privilege and secure defaults. \n\n**Key Considerations:**\n1. **Integrity:** Ensure the proposed changes do not introduce unauthorized state transitions.\n2. **Confidentiality:** If this involves data handling, verify encryption standards are met.\n3. **Availability:** Assess if the architecture introduces new single points of failure.\n\n**Recommendation:** Align this implementation with the STRIDE threat model and ensure all interfaces are properly validated."}
        ]
    })
    
    return samples

def create_dataset(base_path, output_path):
    """Main execution to create the large-scale dataset."""
    raw_data = discover_all_adrs(base_path)
    
    if not raw_data:
        logging.error("No data found to process!")
        return

    training_data = []

    # 1. Add the "Gold Standard" manual examples (keeping your original high-quality logic)
    # [Note: These are hardcoded in the script to ensure they are always present]
    gold_samples = [
        # PostgreSQL to MongoDB contradiction example
        {
            "messages": [
                {"role": "system", "content": "You are an expert in Architecture Decision Records (ADRs). You analyze proposed ADRs, detect contradictions with historical decisions, identify security risks, and provide recommendations with code examples when applicable."},
                {"role": "user", "content": "Analyze this proposed ADR:\n\nTitle: Migrate billing module from PostgreSQL to MongoDB\n\nContent:\nWe propose migrating our billing database from PostgreSQL to MongoDB to improve scalability for high-volume transaction processing. MongoDB's flexible schema will allow faster iteration on billing formats.\n\nHistorical ADRs:\n- ADR-042: PostgreSQL for all financial modules due to ACID requirements and audit compliance\n- ADR-038: NoSQL databases rejected for transaction-heavy workloads\n\nDoes this contradict any previous decisions? What are the security risks?"},
                {"role": "assistant", "content": "## Analysis\n\n### CONTRADICTION DETECTED\n\nADR-042 explicitly states: PostgreSQL for all financial modules due to ACID requirements and audit compliance.\n\nMigrating to MongoDB directly contradicts ADR-042 because:\n1. MongoDB lacks true ACID transactions across multiple documents\n2. Financial module requires audit compliance (PostgreSQL audit features)\n3. ADR-038 specifically rejected NoSQL for transaction-heavy workloads\n\n### SECURITY RISKS\n\n1. NoSQL Injection: MongoDB is vulnerable to injection attacks if user input is not properly validated.\n\n2. Data Exposure: Flexible schema makes it easier to accidentally expose sensitive fields.\n\n3. Audit Trail Gaps: MongoDB's native audit capabilities are limited compared to PostgreSQL."}
            ]
        },
        # SQL Injection example
        {
            "messages": [
                {"role": "system", "content": "You are a security expert specializing in code analysis. You identify vulnerabilities and provide secure alternatives."},
                {"role": "user", "content": "Analyze this code snippet for security vulnerabilities:\n\nAPI endpoint to fetch user data:\n\n query = f\"SELECT * FROM users WHERE id = '{user_id}'\"\n cursor.execute(query)"},
                {"role": "assistant", "content": "## Security Analysis\n\n### CRITICAL: SQL Injection\nThe code uses string interpolation to build SQL queries, allowing injection attacks.\n\n### SECURE ALTERNATIVE\n\n```python\ndef get_user_data(user_id):\n    query = \"SELECT * FROM users WHERE id = %s\"\n    cursor.execute(query, (user_id,))  # Parameterized\n    return cursor.fetchone()\n```"}
            ]
        }
    ]
    training_data.extend(gold_samples)

    # 2. Process all discovered documents
    logging.info(f"Generating training samples from {len(raw_data)} documents...")
    for item in raw_data:
        samples = generate_diverse_samples(item)
        training_data.extend(samples)

    # 3. Write JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info("="*40)
    logging.info(f"FINAL DATASET SIZE: {len(training_data)} samples")
    logging.info(f"OUTPUT: {output_path}")
    logging.info("="*40)

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    out = base / "fine-tuning" / "training-data.jsonl"
    create_dataset(base, out)
