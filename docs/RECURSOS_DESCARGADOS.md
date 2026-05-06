# ADR Security Validator - Recursos Descargados

## Estructura de Datos

```
adr-validator-hackathon/
├── data/
│   ├── adrs/
│   │   ├── kubernetes/        # ADRs oficiales de Kubernetes (sig-architecture)
│   │   ├── rust-lang/         # RFCs de Rust (text/ folder)
│   │   ├── django/            # Django Enhancement Proposals (DEPs)
│   │   └── plantillas/        # Plantillas ADR de joelparkerhenderson + adr-tools
│   └── security/
│       ├── owasp/             # OWASP WrongSecrets, Swiss-Cheese, NodeGoat
│       ├── bandit/            # Ejemplos de código seguro/inseguro (90+ archivos)
│       └── gosec/             # Go security examples
```

## ADRs Disponibles para Qdrant

### Kubernetes (~15 ADRs)
- `sig-architecture/naming/recommendations/*.md`
- `sig-architecture/production-readiness.md`
- `sig-architecture/backlog.md`
- `sig-architecture/api-review-process.md`

### Django DEPs (~30 ADRs)
- `final/` - Decisiones aceptadas y finalizadas
- `accepted/` - Aceptadas pero no implementadas
- `draft/` - En desarrollo
- `rejected/` - Rechazadas

### Rust RFCs (~3 en shallow clone)
- text/ folder con archivos .md

### Plantillas ADR
- Múltiples formatos: MADR, Alexandrian, Nygard, etc.
- Ejemplos completos en `locales/tr/examples/`

## Código Seguro/Inseguro para Fine-tuning

### Bandit Examples (90+ archivos)
Cada archivo muestra un patrón vulnerable y su versión segura:

| Archivo | Vulnerabilidad |
|---------|----------------|
| `sql_statements.py` | SQL Injection |
| `yaml_load.py` | YAML Deserialization |
| `pickle_deserialize.py` | Pickle Deserialization |
| `eval.py` | Code Injection via eval() |
| `exec.py` | Code Injection via exec() |
| `hardcoded-passwords.py` | Hardcoded credentials |
| `ssl-insecure-version.py` | Weak SSL/TLS |
| `subprocess_shell.py` | Shell injection |

### OWASP WrongSecrets
- Ejemplos de secretos mal manejados en diferentes lenguajes
- Terraform, Kubernetes, Node.js, Java, Go, Python

### OWASP Swiss-Cheese
- Implementaciones de OWASP Top 10 en Python
- Cada vulnerabilidad con exploit y mitigación

## Próximos Pasos

1. **Extraer ADRs** - Parsear todos los archivos .md y convertirlos a formato JSONL
2. **Crear dataset seguridad** - Generar pares seguro/inseguro con explicaciones
3. **Preparar fine-tuning** - Formato chatML para Qwen3-8B

## Comandos de Inspección Rápida

```bash
# Contar ADRs de Kubernetes
(Get-ChildItem -Path "data/adrs/kubernetes/sig-architecture" -Filter "*.md" -Recurse).Count

# Contar DEPs de Django
Get-ChildItem -Path "data/adrs/django" -Directory | ForEach-Object {
    $count = (Get-ChildItem -Path $_.FullName -Filter "*.rst" -Recurse).Count
    Write-Output "$($_.Name): $count DEPs"
}

# Listar ejemplos de Bandit
Get-ChildItem -Path "data/security/bandit/examples" -Filter "*.py" | Select-Object Name
```