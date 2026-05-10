# Guía Avanzada de Despliegue en AMD Developer Cloud - ADR Security Validator

Esta guía contiene los pasos actualizados y las lecciones aprendidas para desplegar con éxito el proyecto en instancias **AMD Instinct™ (MI300X/MI210)** con **ROCm 6.2+** y **Python 3.12**.

---

## 1. Configuración de Jupyter Server
Si deseas usar el notebook interactivo (`amd_deployment_guide.ipynb`), sigue estos pasos para iniciar el servidor de forma segura:

1. **Iniciar Jupyter**:
   ```bash
   jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
   ```
2. **Acceso**: Copia el token que aparece en la terminal y accede desde tu navegador local usando la IP de tu instancia: `http://<IP_INSTANCIA>:8888/?token=<tu_token>`.

---

## 2. Gestión de Contenedores (Qdrant)
Para evitar conflictos de puertos y asegurar la persistencia de los datos, usamos **Docker Compose**.

### Limpieza de contenedores antiguos:
Si tienes un contenedor de Qdrant corriendo fuera de compose, bórralo:
```bash
# Detener y borrar por ID o nombre (ej: goofy_keller)
docker ps -q --filter ancestor=qdrant/qdrant | xargs -r docker stop
docker ps -a -q --filter ancestor=qdrant/qdrant | xargs -r docker rm
```

### Iniciar con Docker Compose:
```bash
docker compose up -d
```
*Los datos se guardarán en `./qdrant_storage` para persistir entre reinicios.*

---

## 3. Preparación del Entorno (Fix para ROCm + Python 3.12)
Existen incompatibilidades conocidas entre las versiones más recientes de `transformers` y los drivers de ROCm en Python 3.12. Sigue este orden exacto:

### A. Limpieza de librerías conflictivas:
```bash
source /root/hackathon_env/bin/activate
pip uninstall -y transformers torchao unsloth
```

### B. Instalación de Versiones Estables:
Instalamos la "versión dorada" (4.51.0) que soporta Qwen3 sin los bugs de tipos de tensores:
```bash
pip install "transformers==4.51.0" accelerate einops "sentence-transformers>=2.7.0" trl gunicorn
```

---

## 4. Iniciar la API Backend (Gunicorn + Stable Settings)
Para un despliegue estable en AMD Cloud, se recomienda usar **Gunicorn** con un tiempo de espera (timeout) extendido, ya que el modelo Qwen3-8B toma tiempo en cargarse e inferir sobre la GPU.

```bash
cd backend
# Iniciar servidor con 5 min de timeout y un solo trabajador para maximizar potencia GPU
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300
```

---

## 5. Autenticación en Hugging Face
Para evitar límites de descarga y mensajes de advertencia, usa la nueva herramienta `hf`:

```bash
# Iniciar sesión con tu token (Gratis en huggingface.co/settings/tokens)
hf auth login --token "tu_token_aqui"
```

---

## 6. Solución de Errores Comunes (Workarounds)

### Error: `AttributeError: module 'torch' has no attribute 'int1'`
Este error ocurre porque `torchao` busca funciones que no están en el PyTorch de ROCm. El proyecto ya incluye un parche automático en `backend/main.py`.

### Error: `TypeError: infer_schema() takes 1 positional argument but 3 were given`
Corregido mediante un parche de bajo nivel en el código que intercepta las llamadas de `transformers`.

---

## 7. Verificación del Sistema
Una vez que el servidor esté corriendo, puedes verificar que todo funciona con este script de prueba. El tiempo estimado de respuesta es de **30 a 60 segundos** debido a la profundidad del análisis de la IA.

```python
import requests, json

url = "http://localhost:8000/validate-adr"
payload = {
    "title": "Prueba de Almacenes WMS",
    "content": "Usaremos MongoDB para el inventario y guardaremos el password en config.py.",
}
response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

### ¿Qué buscar en la respuesta?
1. **`AI Analysis`**: Debe aparecer en las recomendaciones (confirmación de que el Fine-tuning funciona).
2. **`related_adrs`**: Debe mostrar ADRs reales (confirmación de que Qdrant está conectado).
3. **`status`**: Debe ser `needs_review`.

---
**Desarrollado para el AMD ROCm Hackathon 2026**
