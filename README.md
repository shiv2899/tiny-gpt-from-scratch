# Tiny GPT from Scratch

A Transformer-based Language Model built from scratch in PyTorch, trained on Tiny Shakespeare.

## Results
| Model | Train Loss | Val Loss |
|---|---|---|
| Bigram (Baseline) | ~2.47 | - |
| Transformer GPT (1000 steps) | 1.60 | 1.21 |

- Total Params: 0.816M
- Vocab Size: 65 characters
- Dataset: 1,115,394 characters - Tiny Shakespeare

## Architecture
- 4 Transformer Blocks, 4 Attention Heads
- Embedding Dim: 128, Block Size: 64, Batch Size: 32
- Self-Attention + FeedForward + LayerNorm + Residual Connections

## Sample Output
```
Ne? Otor seer hear.
MENEVOR:
I no ewfll I pecoetser.
Noserds wisgmy, charde wevan me tly wontrsp...
```

## How to Run
```bash
pip install -r requirements.txt
# Download data
wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cd src
python train.py
python generate.py
```

## What I Learned
- Character-level Tokenization
- Bigram vs Transformer
- Self-Attention, Positional Embeddings
- Training Loop, Loss, Sampling with Temperature

Inspired by Andrej Karpathy - Let's build GPT / nanoGPT
