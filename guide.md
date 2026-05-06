# Guía de Despliegue en AMD Developer Cloud - ADR Security Validator

Esta guía detalla los pasos necesarios para desplegar y ejecutar el sistema de validación de ADRs en la infraestructura de AMD, aprovechando los aceleradores Instinct (MI300X, MI210, etc.) y el ecosistema ROCm.

## 1. Selección de Hardware y Gráfica

Para este proyecto (Qwen3-8B + Qdrant + Fine-tuning), recomendamos la siguiente selección en el panel de AMD Developer Cloud:

### Opción Recomendada (Fine-tuning + Inferencia)
- **Instancia**: Instancia con **1x AMD Instinct™ MI300X** (o MI250).
- **VRAM**: 192 GB HBM3.
- **Razón**: El entrenamiento (Fine-tuning) con LoRA/QLoRA es extremadamente rápido en esta tarjeta y permite manejar el contexto completo de 8B parámetros sin degradación de velocidad.

### Opción Económica (Solo Inferencia)
- **Instancia**: Instancia con **1x AMD Instinct™ MI210**.
- **VRAM**: 64 GB HBM2e.
- **Razón**: Suficiente para ejecutar el modelo base o el modelo con adaptadores cargados en 4-bit/8-bit para validación en tiempo real.

## 2. Configuración de la Instancia

Una vez lanzada la instancia:
1. Conéctate vía SSH.
2. Asegúrate de tener **ROCm** instalado (normalmente pre-configurado en las imágenes de AMD). Verifica con:
   ```bash
   rocm-smi
   ```

## 3. Instalación del Entorno (Paso a Paso)

### A. Clonar y preparar entorno virtual
```bash
git clone <tu-repositorio>
cd adr-validator-hackathon
python3 -m venv venv
source venv/bin/activate
```

### B. Instalar PyTorch optimizado para ROCm
No uses la instalación estándar. Instala la versión compilada para los aceleradores AMD:
```bash
# Para ROCm 6.0 (Ajustar según la versión reportada por rocm-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### C. Instalar dependencias del proyecto
```bash
pip install -r backend/requirements.txt
pip install transformers peft sentence-transformers accelerate bitsandbytes
```

## 4. Despliegue de Servicios

### Paso 1: Levantar base de datos vectorial (Qdrant)
Ejecuta Qdrant usando Docker para el almacenamiento de los ADRs históricos:
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Paso 2: Indexación de datos
Antes de validar, debemos procesar los ADRs de Kubernetes, Django y Rust:
```bash
cd backend
python qdrant_setup.py --embed
```
*Nota: Esto descargará `Qwen3-embed-8b` para generar los vectores de alta fidelidad.*

### Paso 3: Fine-tuning (Opcional pero Recomendado)
Si deseas que el modelo aprenda específicamente tus reglas de seguridad y contradicciones:
```bash
cd ../fine-tuning
python fine_tune_qwen3_8b.py
```
*El resultado se guardará en `../models/qwen-adr-lora`.*

### Paso 4: Iniciar la API Backend
Lanza el servidor FastAPI para recibir peticiones desde Obsidian o cURL:
```bash
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 5. Acceso Externo (Obsidian)

Para conectar tu plugin de Obsidian local con la nube de AMD:
1. **IP Pública**: Obtén la IP de tu instancia en el panel de AMD.
2. **Puerto**: Asegúrate de que el puerto `8000` esté abierto en las reglas de red (Security Groups) de la consola de AMD.
3. **Configuración**: En el plugin de Obsidian, cambia `localhost:8000` por `http://<IP_DE_AMD>:8000`.

## 6. Monitoreo de Recursos
Mientras el sistema esté corriendo, puedes monitorear el uso de la GPU (VRAM y Compute) con:
```bash
watch -n 1 rocm-smi
```

---
**Desarrollado para el AMD ROCm Hackathon 2026**
