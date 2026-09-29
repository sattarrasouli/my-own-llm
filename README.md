# My Own LLM

A small GPT-style language model built from scratch with **Python and PyTorch**.

The goal of this project is to understand how modern language models work internally by implementing the main components step by step, rather than relying on high-level LLM frameworks.

## Architecture

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Multi-Head Self-Attention
 ↓
Transformer Blocks
 ↓
Language Model Head
 ↓
Logits
 ↓
Cross-Entropy Loss
 ↓
Backpropagation + AdamW
```

## Current Features

- Character-level tokenizer
- Token and positional embeddings
- Causal self-attention
- Multi-head self-attention
- Transformer blocks
- Feed-forward networks with GELU
- Layer normalization
- Residual connections
- GPT-style language model
- Next-token prediction
- Cross-entropy loss
- AdamW training loop

## Project Structure

```text
my-own-llm/
├── attention.py
├── dataset.py
├── embedding.py
├── loss.py
├── model.py
├── tokenizer.py
├── transformer.py
├── trainer.py
├── train.py
└── tests/
```

## Goals

The project will progressively explore:

- BPE tokenization
- Better training and evaluation pipelines
- Text generation
- Model checkpoints
- Learning-rate scheduling
- Larger Transformer models
- Instruction tuning
- LoRA / parameter-efficient fine-tuning
- Quantization
- RAG integration
- Model serving

## Tech Stack

- Python
- PyTorch
- NumPy
- pytest

## Status

🚧 **Work in progress**

The model is intentionally built incrementally, with each component implemented and tested independently.
