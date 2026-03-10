# bareboneGPT

**bareboneGPT** is a minimal GPT-style (decoder-only) language model that can be trained from scratch using PyTorch.

The goal of this project is educational clarity rather than raw performance. It demonstrates how modern language models work internally while keeping the implementation small enough to read end-to-end.

The repository includes:

- a minimal GPT architecture implementation
- a configurable training pipeline
- support for large streaming datasets
- text generation utilities
- a lightweight API server for inference
- CI for automated smoke tests

Repository:

https://github.com/saumajitdas/bareboneGPT

---

## Why this project exists

Modern LLMs like GPT-3 and GPT-4 can feel like black boxes. However, the core architecture behind them is surprisingly understandable.

This project was built to answer a simple question:

> How little code do you actually need to train a GPT-style model?

bareboneGPT shows that the essential components of a transformer language model can be implemented in a relatively small codebase while still supporting:

- real datasets
- streaming training
- checkpointing
- text generation
- API serving

---

## Architecture Overview

The system follows the standard autoregressive GPT pipeline:

```text
Dataset
   ↓
Tokenizer
   ↓
Training Dataset
   ↓
Transformer (GPT)
   ↓
Checkpoint
   ↓
Text Generation
   ↓
API Server
```

The model itself is a decoder-only transformer, consisting of:
	•	token embeddings
	•	positional embeddings
	•	stacked transformer blocks
	•	causal self-attention
	•	feed-forward networks
	•	residual connections
	•	output projection to vocabulary

Each training step teaches the model to predict the next token given previous tokens.

---

## Project Structure

```text
bareboneGPT
│
├─ src/barebonegpt
│   ├─ model.py              # GPT architecture
│   ├─ train.py              # training pipeline
│   ├─ generate.py           # text generation
│   ├─ tokenizer.py          # byte-level tokenizer
│   ├─ streaming_dataset.py  # large dataset streaming loader
│   └─ server.py             # FastAPI inference server
│
├─ configs
│   ├─ tiny.json             # small local training config
│   ├─ stream.json           # large dataset training config
│   └─ ci.json               # CI smoke test config
│
├─ scripts
│   └─ download_sample_data.py
│
└─ checkpoints
```

---

## Quickstart

Clone the repository and create a Python environment:

```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a sample dataset:

```text
python scripts/download_sample_data.py
```

Train the model:

```text
PYTHONPATH=src python -m barebonegpt.train --config configs/tiny.json
or,
PYTHONPATH=src python -m barebonegpt.train --config configs/stream.json
```

Generate text from trained checkpoint:

```text
PYTHONPATH=src python -m barebonegpt.generate \
  --checkpoint checkpoints/model.pt \
  --prompt "Hello"
```

Start the API server:

```text
PYTHONPATH=src python -m barebonegpt.server \
  --checkpoint checkpoints/model.pt
```
