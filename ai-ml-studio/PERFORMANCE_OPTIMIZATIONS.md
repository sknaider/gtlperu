# ⚡ Performance Optimizations for High-End Hardware

**Optimized for:**
- **GPU:** NVIDIA RTX 5090 (24GB VRAM)
- **CPU:** AMD Ryzen 9 9950X (16 cores / 32 threads)
- **RAM:** 128GB DDR5
- **OS:** Linux

---

## 🚀 Hardware-Specific Optimizations

### GPU Optimizations (RTX 5090)

#### 1. **TF32 Precision**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
- 8x faster than FP32 with minimal accuracy loss
- Native support on Ada Lovelace architecture

#### 2. **Flash Attention 2**
- 2-4x faster attention computation
- Lower memory usage
- Automatically enabled in supported models

#### 3. **Torch Compile**
```python
model = torch.compile(model, mode="max-autotune")
```
- Up to 2x inference speedup
- Optimized for RTX 5090's architecture

#### 4. **Large Batch Sizes**
With 24GB VRAM, you can use:
- **Training:** Batch size 128-256
- **Inference:** Batch size 512+
- **Embeddings:** Batch size 1024+

### CPU Optimizations (Ryzen 9 9950X)

#### 1. **Multi-threading**
```python
torch.set_num_threads(32)  # Use all 32 threads
```

#### 2. **DataLoader Settings**
```python
DataLoader(
    dataset,
    batch_size=128,
    num_workers=16,  # All physical cores
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

#### 3. **Parallel Hyperparameter Tuning**
```python
# Optuna with parallel trials
study.optimize(objective, n_trials=100, n_jobs=16)
```

### Memory Optimizations (128GB RAM)

#### 1. **No Memory Constraints**
- Load entire datasets into RAM
- Cache embeddings and features
- No need for 8-bit optimizers

#### 2. **High-Quality LoRA**
```python
LoRAConfig(
    r=64,  # High rank (vs typical r=8)
    lora_alpha=128,
    use_bf16=True,  # Better precision
)
```

---

## 📊 Performance Benchmarks

### Expected Performance on Your Hardware

| Task | Batch Size | Throughput | Notes |
|------|------------|-----------|--------|
| **Training ResNet50** | 256 | ~5000 img/s | Mixed precision |
| **Fine-tuning LLaMA-7B (LoRA)** | 8 | ~12 tokens/s | No quantization needed |
| **Fine-tuning LLaMA-13B (QLoRA)** | 4 | ~8 tokens/s | 4-bit quantization |
| **Embedding Generation** | 1024 | ~50k texts/s | Sentence-Transformers |
| **Image Classification Inference** | 512 | ~20k img/s | EfficientNet-B0 |

### Memory Usage Estimates

| Model | Precision | VRAM | RAM | Fits? |
|-------|-----------|------|-----|-------|
| **LLaMA-7B** | FP16 | 14GB | 16GB | ✅ Yes |
| **LLaMA-13B** | FP16 | 26GB | 28GB | ❌ Use QLoRA |
| **LLaMA-13B** | QLoRA (4-bit) | 8GB | 16GB | ✅ Yes |
| **LLaMA-70B** | QLoRA (4-bit) | 35GB | 40GB | ❌ Too large |
| **Mistral-7B** | FP16 | 14GB | 16GB | ✅ Yes |
| **Vision Transformer** | FP16 | 2GB | 4GB | ✅ Yes |

---

## 🔧 Auto-Optimization

The system automatically optimizes for your hardware:

```python
from backend.core.hardware_optimizer import enable_optimizations

enable_optimizations()
```

This configures:
- ✅ TF32 precision
- ✅ cuDNN benchmarking
- ✅ Optimal thread count
- ✅ JIT fusion
- ✅ Optimal batch sizes

---

## 💡 Recommended Configurations

### For Maximum Speed

```python
config = {
    "mixed_precision": "bf16",
    "torch_compile": True,
    "compile_mode": "max-autotune",
    "batch_size": 256,
    "num_workers": 16,
    "gradient_checkpointing": False,  # Plenty of VRAM
}
```

### For Maximum Model Size

```python
# Fine-tune larger models with QLoRA
qlora_config = QLoRAConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    r=64,
    load_in_4bit=True,
    batch_size=4,
    gradient_accumulation_steps=4,
)
```

### For Maximum Throughput

```python
# Inference with large batches
classifier = ImageClassifier(
    model_name="efficientnet_b0",
    device="cuda"
)

# Compile for speed
classifier.model = torch.compile(classifier.model)

# Process large batches
results = classifier.predict_batch(images, batch_size=512)
```

---

## 🎯 Best Practices

### 1. Always Use Mixed Precision
```python
# BF16 is better than FP16 on RTX 5090
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

### 2. Maximize Batch Size
- Start with large batches (128-256)
- Reduce if you get OOM errors
- Your 24GB VRAM can handle a lot!

### 3. Use All CPU Cores
- Set `num_workers=16` in DataLoader
- Use `n_jobs=16` in Optuna
- Parallel processing wherever possible

### 4. Cache Aggressively
- 128GB RAM means you can cache everything
- Cache embeddings, features, preprocessed data

### 5. Compile Your Models
```python
model = torch.compile(model, mode="max-autotune")
```

---

## 🔬 Advanced Optimizations

### 1. Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

Then in your model:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

### 2. Channels Last Memory Format
```python
# Better memory layout for CNNs
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

### 3. CUDA Graphs (for fixed-size inputs)
```python
# Capture CUDA graph for faster repeated inference
static_input = torch.randn(batch_size, 3, 224, 224, device="cuda")

# Warmup
for _ in range(3):
    model(static_input)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (super fast!)
static_input.copy_(actual_input)
g.replay()
```

---

## 📈 Monitoring Performance

### GPU Utilization
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### Detailed Profiling
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Benchmark Script
```bash
python start.py  # Choose option 4: Run Benchmark
```

---

## 🎓 Optimization Checklist

Before training/inference:

- [ ] Enable hardware optimizations (`enable_optimizations()`)
- [ ] Use BF16 mixed precision
- [ ] Set large batch size (start with 128)
- [ ] Set `num_workers=16` in DataLoader
- [ ] Enable `torch.compile()` for models
- [ ] Use Flash Attention 2 for transformers
- [ ] Monitor GPU utilization (should be 95%+)
- [ ] Check memory usage (optimize if needed)

---

## 💰 Cost Efficiency

With $500 for Claude API:

### Recommended Allocation
- **$100** - Dataset generation (10k+ samples)
- **$150** - Data pre-labeling and validation
- **$150** - RAG system queries and embeddings
- **$100** - Prompt engineering and experimentation

### Cost-Saving Tips
1. **Cache everything** - With 128GB RAM, cache all Claude responses
2. **Batch API calls** - Generate multiple samples per request
3. **Use RAG** - Reduce Claude calls by retrieving from knowledge base
4. **Pre-compute embeddings** - Cache sentence embeddings

---

**Your hardware is BEAST MODE. Use it! 🚀**
