# 🚀 Quick Start Guide - AI ML Studio

Get started with AI ML Studio in minutes!

---

## 📦 Installation

### Option 1: Local Installation (Recommended for Development)

```bash
# Clone the repository
cd ai-ml-studio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize workspace
python cli.py init

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Option 2: Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up -d

# Check services status
docker-compose ps

# View logs
docker-compose logs -f backend
```

---

## ⚡ Quick Examples

### 1. Generate Synthetic Dataset

Generate training data using Claude AI:

```bash
# Generate customer support conversations
python cli.py dataset generate \
  --task "customer support conversations for e-commerce" \
  --samples 500 \
  --output ./data/synthetic/customer_support.jsonl \
  --type conversation

# Generate text classification data
python cli.py dataset generate \
  --task "sentiment analysis of product reviews" \
  --samples 1000 \
  --output ./data/synthetic/sentiment.jsonl \
  --type classification

# Generate Q&A pairs
python cli.py dataset generate \
  --task "Python programming" \
  --samples 200 \
  --output ./data/synthetic/python_qa.jsonl \
  --type qa
```

### 2. Fine-tune a Model with LoRA

```bash
# Fine-tune LLaMA 2 with LoRA
python cli.py finetune start \
  --model meta-llama/Llama-2-7b-hf \
  --dataset ./data/synthetic/customer_support.jsonl \
  --method lora \
  --rank 8 \
  --output ./models/lora-llama2-support

# Fine-tune with QLoRA (4-bit, lower memory)
python cli.py finetune start \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset ./data/synthetic/python_qa.jsonl \
  --method qlora \
  --rank 64 \
  --output ./models/qlora-mistral-python
```

### 3. Train a Computer Vision Model

```bash
# Train image classifier
python cli.py train start \
  --model resnet50 \
  --dataset cifar10 \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001

# Resume from checkpoint
python cli.py train resume <run_id>
```

### 4. Build a RAG System

```bash
# Index your documents
python cli.py rag index \
  --documents ./docs \
  --collection my_knowledge_base

# Query the system
python cli.py rag query \
  --question "How do I deploy models?" \
  --collection my_knowledge_base \
  --top-k 5
```

### 5. Data Labeling

```bash
# Start labeling assistant with auto pre-labeling
python cli.py label start \
  --data ./data/raw/images \
  --output ./data/labeled \
  --auto-prelabel
```

---

## 🐍 Python API Examples

### Generate Dataset Programmatically

```python
from backend.modules.datasets.generator import DatasetGenerator

# Initialize generator
generator = DatasetGenerator()

# Generate classification dataset
dataset = generator.generate_text_classification_dataset(
    task_description="Classify customer feedback sentiment",
    categories=["positive", "negative", "neutral"],
    num_samples=1000
)

# Save to file
from pathlib import Path
generator.save_dataset(
    dataset,
    Path("./data/synthetic/feedback.jsonl"),
    format="jsonl"
)
```

### Fine-tune with LoRA

```python
from backend.modules.finetuning.lora import LoRAFineTuner, LoRAConfig

# Configure
config = LoRAConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    r=8,
    lora_alpha=16,
    learning_rate=2e-4,
    num_epochs=3,
    output_dir="./models/lora-custom"
)

# Initialize and train
finetuner = LoRAFineTuner(config)
finetuner.load_model()

# Load your dataset
from datasets import load_dataset
train_data = load_dataset("json", data_files="./data/train.jsonl")["train"]

# Train
finetuner.train(train_data)

# Generate with fine-tuned model
response = finetuner.generate(
    prompt="What is machine learning?",
    max_length=200
)
print(response)
```

### Fine-tune with QLoRA (Memory Efficient)

```python
from backend.modules.finetuning.qlora import QLoRAFineTuner, QLoRAConfig

# Configure for 4-bit training
config = QLoRAConfig(
    model_name="meta-llama/Llama-2-13b-hf",  # Larger model!
    r=64,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    num_epochs=3
)

# Train (uses much less VRAM)
finetuner = QLoRAFineTuner(config)
finetuner.load_model()
finetuner.train(train_dataset)
```

### Training Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.modules.training import Trainer, TrainingConfig

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Create model
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes

# Configure training
config = TrainingConfig(
    model_name="resnet50",
    epochs=10,
    learning_rate=0.001,
    optimizer="adam",
    scheduler="cosine",
    mixed_precision=True,
    save_dir="./models/checkpoints"
)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader
)

# Train
history = trainer.train()
print(f"Best metric: {history['best_metric']}")
```

---

## 🌐 API Server

### Start the FastAPI Server

```bash
# Development mode (with auto-reload)
python cli.py serve --reload

# Production mode
python cli.py serve --host 0.0.0.0 --port 8000

# Using uvicorn directly
uvicorn backend.api.main:app --reload
```

### API Endpoints (Coming Soon)

```bash
# Generate dataset
curl -X POST http://localhost:8000/api/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "customer support",
    "type": "conversation",
    "num_samples": 100
  }'

# Start training
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "resnet50",
    "dataset": "cifar10",
    "epochs": 10
  }'

# Check training status
curl http://localhost:8000/api/train/status/<job_id>
```

---

## 📊 Monitoring with MLflow

```bash
# MLflow is automatically started with docker-compose
# Access at: http://localhost:5000

# Or start manually
mlflow ui --backend-store-uri postgresql://... --port 5000
```

---

## 💡 System Information

```bash
# Check system info
python cli.py info

# Output:
# === AI ML Studio Info ===
# Version: 0.1.0
# Python: 3.10.12
# PyTorch: 2.1.0
# CUDA Available: True
# CUDA Version: 11.8
# GPU Count: 1
#   GPU 0: NVIDIA RTX 4090
```

---

## 📁 Project Structure After Init

```
ai-ml-studio/
├── data/
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   ├── synthetic/        # Generated datasets
│   └── labeled/          # Labeled data
├── models/
│   ├── pretrained/       # Downloaded models
│   ├── finetuned/        # Fine-tuned models
│   └── checkpoints/      # Training checkpoints
├── configs/              # Configuration files
├── logs/                 # Log files
└── notebooks/            # Jupyter notebooks
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
WANDB_API_KEY=...
HUGGINGFACE_TOKEN=hf_...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aimlstudio

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Training Config (YAML)

```yaml
# configs/my_training.yaml
model:
  name: "resnet50"
  num_classes: 10

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
```

---

## 🎓 Next Steps

1. **Generate your first dataset**: Start with a small classification task
2. **Try fine-tuning**: Use LoRA on a small model like GPT-2
3. **Train a vision model**: Use the training pipeline with CIFAR10
4. **Build a RAG system**: Index your documentation
5. **Explore notebooks**: Check out example notebooks in `notebooks/`

---

## 🆘 Troubleshooting

### CUDA Out of Memory

```python
# Use QLoRA instead of LoRA
# Reduce batch size
# Enable gradient checkpointing
# Use smaller model
```

### API Key Errors

```bash
# Make sure .env file exists
cp .env.example .env

# Set your keys
export ANTHROPIC_API_KEY=sk-ant-...
```

### Docker Issues

```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## 📚 Documentation

- [Full Documentation](docs/README.md)
- [API Reference](docs/api-reference.md)
- [Tutorials](docs/tutorials/)
- [Examples](docs/examples/)

---

## 🤝 Community

- GitHub: [Issues & Discussions](https://github.com/your-repo/ai-ml-studio)
- Discord: [Join our community](#)
- Twitter: [@aimlstudio](#)

---

**Ready to build amazing ML/AI projects? Let's go! 🚀**
