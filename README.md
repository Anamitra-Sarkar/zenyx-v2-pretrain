# Zenyx-v2 Pretraining — TPU v5e-8

> **Nano-Titan** | ~280M unique params | 32 effective layers | 16k context | Pure BF16

## Architecture

| Component | Design |
|---|---|
| **Attention** | MLA (Multi-head Latent Attention) — 4.5× KV memory saving at 16k ctx |
| **FFN** | ConvSwiGLU — depthwise k=3 conv + SwiGLU gate |
| **Depth** | 8 unique blocks × 4 recurrences = 32 effective layers |
| **Position** | YaRN-scaled RoPE (native 16k from 512 base) |
| **Prediction** | MTP: t+1 (weight-tied), t+2, t+3 simultaneously |
| **Precision** | Pure BF16 weights + activations, FP32 optimizer state |
| **Tokenizer** | [Arko007/zenyx-v2-tokenizer](https://huggingface.co/Arko007/zenyx-v2-tokenizer) — 32k BPE vocab |

## Model Config

```
d_model         = 576
n_heads         = 9   (head_dim = 64)
n_kv_heads      = 3   (GQA 3:1)
mlp_hidden      = 1536
n_unique_blocks = 8
n_recurrences   = 4   → effective_depth = 32
max_seq_len     = 16,384
kv_latent       = 128
q_latent        = 384
mtp_heads       = 3
vocab_size      = 32,768
```

## Training Data  

| Source | Ratio | Config |
|---|---|---|
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 40% | `finemath-3plus` |
| [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) | 40% | 24 languages (scala excluded) |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 20% | Default full split, score≥3.0 |

**Total target:** 150B tokens

## Training Config

```
Global batch    = 256 seqs × 16,384 tokens = ~4.2M tokens/step
Grad accum      = 16 steps
LR schedule     = WSD: warmup(2k) → stable(80k) → cosine decay(18k)
Peak LR         = 3e-4 → min 3e-5
Optimizer       = AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=0.1)
Max steps       = 100,000
Grad clip       = 1.0
```

## Setup

```bash
# On Kaggle TPU v5e-8 notebook
bash install.sh
python train.py
```

## Checkpointing

Checkpoints are saved every 500 steps to `Arko007/zenyx-v2-base` on HuggingFace Hub:
- `checkpoints/state_stepN.msgpack` — full TrainState (resume training)
- `params/params_stepN.msgpack` — params only (~440MB bf16, for inference)
- `metadata.json` — loss history, model config
- `val_set.npy` — fixed validation set (saved once, reused across sessions)

## Why Recurrent Blocks?

The 8 unique blocks are passed 4 times sequentially. This means:
- **Parameter count stays ~280M** (only 8 unique blocks stored)
- **Compute is 32 layers deep** (each pass builds on previous)
- **Gradients flow through the full unrolled graph** — no stop_gradient tricks
- Reasoning depth emerges from iterative refinement, similar to how humans re-read complex sentences

## BF16 Size Note

The `params_only` checkpoint will be ~440MB (correct for 220M params × 2 bytes/bf16).
The full `TrainState` checkpoint is larger (~1.1GB) because it includes FP32 AdamW optimizer moments (2× params in fp32).
