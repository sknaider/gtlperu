# 🚀 AI ML Studio

**Plataforma Integrada Todo-en-Uno para Entrenamiento y Experimentación con Modelos de IA**

Una solución completa y modular que combina 10 herramientas esenciales para ML/AI en un solo ecosistema.

---

## 📋 Características Principales

### 🔧 1. Pipeline de Entrenamiento Automatizado
- Soporte para PyTorch y TensorFlow
- Hyperparameter tuning con Optuna y Ray Tune
- Configuración mediante YAML
- Checkpointing automático
- Multi-GPU training

### 🤖 2. Sistema de Fine-tuning para Modelos Open Source
- Fine-tuning de LLaMA, Mistral, Phi, y más
- Implementaciones LoRA y QLoRA
- PEFT (Parameter-Efficient Fine-Tuning)
- Scripts de evaluación integrados
- Cuantización INT4/INT8

### 📊 3. Plataforma de Experimentación ML
- Integración con MLflow
- Tracking con Weights & Biases
- Comparación visual de modelos
- Métricas en tiempo real
- Gestión de versiones de modelos

### 🏷️ 4. Data Labeling Assistant
- Interfaz web intuitiva
- Pre-etiquetado inteligente con Claude API
- Soporte multi-formato (texto, imagen, audio)
- Exportación a formatos estándar (COCO, YOLO, JSON)
- Colaboración en equipo

### 🎲 5. Generador de Datasets Sintéticos
- Generación con Claude API
- Aumentación automática de datos
- Validación de calidad
- Balanceo de clases
- Templates personalizables

### 🔍 6. Sistema RAG (Retrieval Augmented Generation)
- Embeddings con Sentence-Transformers
- Vector DB con ChromaDB/Pinecone
- Búsqueda semántica
- Fine-tuning de retrieval
- Multi-documento

### 👁️ 7. Pipeline de Computer Vision
- Clasificación de imágenes
- Object Detection (YOLO, Faster R-CNN)
- Segmentación semántica
- Transfer learning
- Exportación ONNX

### 🖼️ 8. Data Augmentation Automático
- Augmentation para imágenes (albumentations)
- Augmentation para texto (NLP)
- Políticas de augmentation automáticas
- Generación sintética
- Preview en tiempo real

### 💬 9. Sistema de Prompt Engineering
- Editor de prompts interactivo
- Few-shot learning optimizado
- Evaluación automática de prompts
- Dataset de ejemplos
- Versionado de prompts

### 📝 10. Fine-tuning Dataset Creator
- Generación con Claude API
- Formateo para OpenAI/Anthropic/HuggingFace
- Validación y limpieza automática
- Conversión entre formatos
- Análisis de calidad

---

## 🛠️ Stack Tecnológico

### Backend
- **Python 3.10+**
- **PyTorch** & **TensorFlow**
- **FastAPI** - API REST
- **Celery** - Task queue
- **Redis** - Caché y queue
- **PostgreSQL** - Base de datos principal
- **ChromaDB** - Vector database

### Frontend
- **React 18** con TypeScript
- **Next.js 14** - Framework
- **TailwindCSS** - Estilos
- **Shadcn/ui** - Componentes
- **Recharts** - Visualizaciones
- **React Query** - Data fetching

### ML/AI Libraries
- **Transformers** (HuggingFace)
- **PEFT** (LoRA/QLoRA)
- **Optuna** - Hyperparameter tuning
- **Ray Tune** - Distributed tuning
- **MLflow** - Experiment tracking
- **Weights & Biases** - Experiment tracking
- **Albumentations** - Image augmentation
- **Sentence-Transformers** - Embeddings
- **LangChain** - LLM orchestration

### DevOps
- **Docker** & **Docker Compose**
- **Nginx** - Reverse proxy
- **Prometheus** & **Grafana** - Monitoring
- **GitHub Actions** - CI/CD

---

## 📦 Instalación

### Requisitos Previos
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- CUDA 11.8+ (para GPU)
- 16GB RAM mínimo (32GB recomendado)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/ai-ml-studio.git
cd ai-ml-studio

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Opción 1: Con Docker (Recomendado)
docker-compose up -d

# Opción 2: Instalación Local
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
npm run dev

# Iniciar servicios
python backend/main.py
```

### Configuración con GPU

```bash
# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 1. Entrenar tu Primer Modelo

```bash
# Usando el CLI
python cli.py train \
  --config configs/train_config.yaml \
  --model resnet50 \
  --dataset cifar10

# Usando la API
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet50", "dataset": "cifar10"}'
```

### 2. Fine-tuning de LLaMA

```bash
# Con LoRA
python cli.py finetune \
  --model meta-llama/Llama-2-7b-hf \
  --dataset my_dataset.json \
  --method lora \
  --rank 8
```

### 3. Generar Dataset Sintético

```bash
# Generar datos con Claude
python cli.py generate-dataset \
  --task "customer support conversations" \
  --samples 1000 \
  --output data/synthetic/
```

### 4. Etiquetar Datos

```bash
# Iniciar labeling assistant
python cli.py label \
  --data data/raw/images/ \
  --output data/labeled/ \
  --auto-prelabel
```

### 5. Crear Sistema RAG

```bash
# Indexar documentos
python cli.py rag index \
  --documents docs/ \
  --collection my_knowledge_base

# Query
python cli.py rag query \
  --question "How to train a model?" \
  --collection my_knowledge_base
```

---

## 📚 Estructura del Proyecto

```
ai-ml-studio/
├── backend/
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── database.py            # Database connections
│   │   ├── cache.py               # Redis cache
│   │   └── claude_client.py       # Claude API client
│   ├── modules/
│   │   ├── training/              # Training pipeline
│   │   │   ├── trainer.py
│   │   │   ├── optimizers.py
│   │   │   └── schedulers.py
│   │   ├── finetuning/            # Fine-tuning system
│   │   │   ├── lora.py
│   │   │   ├── qlora.py
│   │   │   └── evaluator.py
│   │   ├── experimentation/       # MLflow/W&B integration
│   │   │   ├── tracker.py
│   │   │   └── comparator.py
│   │   ├── labeling/              # Data labeling
│   │   │   ├── app.py
│   │   │   └── prelabeler.py
│   │   ├── datasets/              # Synthetic data generation
│   │   │   ├── generator.py
│   │   │   └── validator.py
│   │   ├── rag/                   # RAG system
│   │   │   ├── embeddings.py
│   │   │   ├── retriever.py
│   │   │   └── vectordb.py
│   │   ├── vision/                # Computer vision
│   │   │   ├── classifier.py
│   │   │   ├── detector.py
│   │   │   └── segmentation.py
│   │   ├── augmentation/          # Data augmentation
│   │   │   ├── image_aug.py
│   │   │   └── text_aug.py
│   │   ├── prompts/               # Prompt engineering
│   │   │   ├── editor.py
│   │   │   └── evaluator.py
│   │   └── export/                # Dataset export
│   │       └── formatter.py
│   ├── api/                       # FastAPI routes
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   └── utils/                     # Utilities
├── frontend/
│   ├── dashboard/                 # Main dashboard
│   ├── components/                # React components
│   ├── pages/                     # Next.js pages
│   └── styles/                    # CSS/Tailwind
├── notebooks/                     # Jupyter notebooks
├── configs/                       # Configuration files
├── scripts/                       # Utility scripts
├── data/                          # Data directory
├── models/                        # Model storage
├── docs/                          # Documentation
├── tests/                         # Tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── package.json
└── README.md
```

---

## 🔧 Configuración

### Variables de Entorno (.env)

```env
# API Keys
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
WANDB_API_KEY=your_wandb_api_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aimlstudio
REDIS_URL=redis://localhost:6379

# Vector DB
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Storage
MODEL_STORAGE_PATH=/models
DATA_STORAGE_PATH=/data

# Training
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=10
MIXED_PRECISION=true

# GPU
CUDA_VISIBLE_DEVICES=0,1
```

---

## 📖 Documentación Completa

- [Guía de Usuario](docs/user-guide.md)
- [API Reference](docs/api-reference.md)
- [Tutoriales](docs/tutorials/)
- [Ejemplos](docs/examples/)
- [FAQ](docs/faq.md)

---

## 💰 Estimación de Costos

### Claude API (para 1000 operaciones)

| Operación | Tokens | Costo |
|-----------|--------|-------|
| Pre-etiquetado | ~500k | ~$4 |
| Generación dataset | ~1M | ~$8 |
| Prompt evaluation | ~200k | ~$1.60 |
| **Total mensual** | | **~$50-150** |

### Infraestructura Cloud (opcional)

| Servicio | Costo/mes |
|----------|-----------|
| GPU Cloud (Lambda) | $100-400 |
| Storage (S3) | $10-30 |
| Database (RDS) | $20-50 |
| **Total** | **$130-480** |

### Con $500/mes puedes:
- ✅ 10,000+ operaciones de Claude
- ✅ 100-400 horas GPU
- ✅ Almacenamiento ilimitado
- ✅ Múltiples experimentos en paralelo

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Roadmap

- [x] Estructura base del proyecto
- [ ] Pipeline de entrenamiento PyTorch/TensorFlow
- [ ] Fine-tuning LoRA/QLoRA
- [ ] Integración MLflow/W&B
- [ ] Data labeling assistant
- [ ] Generador de datasets
- [ ] Sistema RAG
- [ ] Computer vision pipeline
- [ ] Data augmentation
- [ ] Prompt engineering
- [ ] Frontend dashboard
- [ ] Docker deployment
- [ ] Documentation completa
- [ ] Tests (>80% coverage)

---

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles

---

## 🙏 Agradecimientos

- HuggingFace Transformers
- PyTorch & TensorFlow teams
- MLflow & Weights & Biases
- Anthropic Claude API
- Open source community

---

## 📧 Contacto

- **Email**: contact@aimlstudio.com
- **Discord**: [Join our community](https://discord.gg/aimlstudio)
- **Twitter**: [@aimlstudio](https://twitter.com/aimlstudio)

---

**¡Construido con ❤️ para la comunidad de ML/AI!**
