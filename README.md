# LLMs from Scratch - Research-Oriented Learning Path

There is a significant gap between most online courses and the depth required for industry-level research engineering in LLMs.
This repository was created to bridge that gap through hands-on implementation.
The notebooks were originally built for my own learning, and are shared here in case they are useful to others pursuing a deeper understanding.

This repository is a structured, end-to-end curriculum for learning Large Language Models (LLMs) from first principles to a level suitable for **research engineering roles**.

The goal is deep understanding, not library magic.
Every lesson is a self-contained Jupyter notebook that builds and trains a real model, explains *why* it works, and tests it with inference.

The models are intentionally small and readable, but each notebook includes scaling notes describing how the same ideas apply to frontier-scale systems.

---

## Philosophy

- **From scratch first** - implement core ideas manually before relying on frameworks
- **One model per lesson** - clear mental boundaries
- **Readable PyTorch** - clarity over cleverness
- **Train + infer in every notebook**
- **Production awareness** - each lesson explains how it scales in real systems

The curriculum mirrors how LLM knowledge actually accumulates in practice:
from language modeling basics -> transformers -> modern architecture -> training tricks -> fine-tuning -> alignment -> research experimentation.

---

## Environment

- **Python**: 3.11
- **Framework**: PyTorch
- **GPU**: I tested these on an NVIDIA RTX 3070 used for local training.
- **Optional**: Google Colab (A100) for larger experiments.
- **Environment management**: `uv sync`

If you use CUDA, install a build of PyTorch compatible with your NVIDIA drivers.

---

## Curriculum Overview

### Lesson 01 - Bigram Character Language Model
**Concepts**
- Language modeling objective
- Token probabilities, logits, softmax
- Cross-entropy loss
- Autoregressive generation

**Model**
- Bigram table over characters

> Establishes the core LM objective that every modern LLM still uses.

---

### Lesson 02 - Tokenization & N-gram Neural Language Model
**Concepts**
- Why tokenization exists
- Byte Pair Encoding (BPE)
- Context windows
- Embeddings + MLP language models

**Model**
- N-gram neural LM over BPE tokens

> Shows how raw text becomes model-friendly discrete tokens.

---

### Lesson 03 - Recurrent Language Models (GRU)
**Concepts**
- Sequential modeling
- Hidden states
- Teacher forcing
- Gradient clipping

**Model**
- GRU-based autoregressive language model

> Historical context: why RNNs were replaced by Transformers.

---

### Lesson 04 - Transformer Decoder (GPT-style) from Scratch
**Concepts**
- Self-attention
- Causal masking
- Multi-head attention
- Residual connections
- LayerNorm

**Model**
- Minimal GPT-style Transformer decoder

> The foundational architecture behind GPT, LLaMA, Claude, etc.

---

### Lesson 05 - Modern GPT Blocks (RMSNorm, SwiGLU, RoPE)
**Concepts**
- RMSNorm vs LayerNorm
- SwiGLU feed-forward networks
- Rotary positional embeddings

**Model**
- Modernized Transformer decoder (LLaMA-style)

> Bridges toy GPTs to modern production LLMs.

---

### Lesson 06 - KV Cache for Fast Autoregressive Inference
**Concepts**
- Why naive generation is slow
- Key-Value caching
- Inference-time optimization

**Model**
- Transformer decoder with KV cache support

> Critical for real-world LLM serving and latency reduction.

---

### Lesson 07 - Training Engineering & Optimization Tricks
**Concepts**
- Learning-rate warmup
- Cosine decay
- Gradient accumulation
- Mixed precision (AMP)

**Model**
- Transformer decoder with advanced training loop

> Focuses on *how models are actually trained* in practice.

---

### Lesson 08 - Supervised Fine-Tuning with LoRA
**Concepts**
- Pretrained LLMs
- Supervised fine-tuning (SFT)
- Parameter-efficient training (LoRA)

**Model**
- GPT-style model fine-tuned with LoRA adapters

> Reflects how most modern LLM work starts: from a pretrained base.

---

### Lesson 09 - Quantization for Inference
**Concepts**
- Model size vs accuracy
- INT8 / lower-precision inference
- Latency & memory tradeoffs

**Model**
- Quantized causal language model

> Practical deployment considerations for real systems.

---

### Lesson 10 - Mixture of Experts (MoE)
**Concepts**
- Conditional computation
- Expert routing
- Load balancing intuition

**Model**
- Transformer with MoE feed-forward blocks

> Introduces scaling strategies used in very large models.

---

### Lesson 11 - Preference Optimization (DPO)
**Concepts**
- Alignment vs pretraining
- Preference datasets
- Direct Preference Optimization (DPO)

**Model**
- Pretrained LM fine-tuned on preference pairs

> Modern alignment technique used instead of RLHF in many pipelines.

---

### Lesson 12 - Research Capstone: Ablations & Experiments
**Concepts**
- Reproducibility
- Controlled experiments
- Architectural ablations
- Result comparison

**Model**
- Configurable GPT with multiple architectural toggles

> Mimics the workflow of a research engineer running model experiments.

---

## How to Use This Repository

### Set up the Python environment

To set up the environment, you can use either a standard Python setup (venv or Conda) or `uv`.

You can use either a standard Python environment (venv or Conda) or `uv`.

```bash
# Option A — Standard Python Environment (pip / venv / Conda)
pip install -r requirements.txt

# Option B — Using uv
uv sync
```

**Open a lesson notebook** and run the cells top-to-bottom.

First run will download `wikitext-2-raw-v1` from Hugging Face.
