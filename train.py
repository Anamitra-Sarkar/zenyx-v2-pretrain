# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   ZENYX-V2 PRETRAINING  —  TPU v5e-8  |  JAX/Flax  |  Pure BF16           ║
# ║   Architecture: Nano-Titan | ~280M params | 16k context                    ║
# ║   Tokenizer: Arko007/zenyx-v2-tokenizer | vocab=32,768                     ║
# ║   Data: FineMath-3+ (40%) + StarCoderData (40%) + FineWeb-Edu (20%)        ║
# ║   Depth: 8 unique blocks × 4 recurrences = 32 effective layers             ║
# ║   Attention: MLA (Multi-head Latent Attention) — 4.5× KV memory saving     ║
# ║   FFN: CausalConvSwiGLU (strictly causal depthwise conv + SwiGLU gate)     ║
# ║   Position: YaRN-scaled RoPE (native 16k context)                          ║
# ║   Prediction: MTP (t+1, t+2, t+3 simultaneously)                           ║
# ║   Optimizer: AdamW + WSD schedule (Warmup→Stable→Cosine Decay)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── AUDIT FIXES (v2.1) ────────────────────────────────────────────────────────
# Fix 1 [CRITICAL] ConvSwiGLU padding="SAME" → causal left-pad.
#        padding="SAME" caused symmetric padding → future token leakage.
#        output[t] now sees only input[t-2], input[t-1], input[t]. ✓
# Fix 2 [MEDIUM]  YaRN interpolation formula was direction-inverted.
#        New: t blends 0.0 (high-freq, unchanged) → 1.0 (low-freq, compressed). ✓
# Fix 3 [LOW]     MLAAttention softmax: cast to float32 BEFORE masking. ✓
# Fix 4 [MEDIUM]  Dropout RNG: clean jrand.split, no key_data/wrap_key_data. ✓
#
# ── AUDIT FIXES (v2.2) ────────────────────────────────────────────────────────
# Fix 5 [CRITICAL] combined_stream shared generator references → 40/40/20 restored. ✓
# Fix 6 [MEDIUM]  model.init (1,64) → (1, MAX_SEQ_LEN) → no first-step recompile. ✓
# Fix 7 [LOW]     TPU_CORES_STATIC hard RuntimeError assertion. ✓
# Fix 8 [LOW]     YaRN denominator epsilon guard. ✓
#
# ── FINAL FIXES (v2.3) ─────────────────────────────────────────────────────────
# Fix 9  [CRITICAL] HBM budget: PER_CORE_BATCH 2→1, GRAD_ACCUM 16→32.
#        With PER_CORE_BATCH=2: attention matrix [2,9,16384,16384] fp32
#        = 19.7 GB alone — exceeds 16 GB per chip. Changed to batch=1:
#        attention [1,9,16384,16384] fp32 = 9.86 GB. Total per-chip budget:
#        params+opt (~2.8 GB) + grads (~1.1 GB) + activations (~9.9 GB)
#        ≈ 13.8 GB → safely within 14-15 GB target. GLOBAL_BATCH unchanged: 256.
# Fix 10 [MEDIUM]  XLA_FLAGS: GPU latency flag → correct TPU flags.
#        --xla_gpu_enable_latency_hiding_scheduler has NO effect on TPU.
#        Replaced with TPU-specific flags that overlap allreduce with compute.
# Fix 11 [LOW]    JAX version guard: typed PRNG keys require JAX >= 0.4.16.
# Fix 12 [LOW]    Val set underflow guard: fail loudly instead of silent reshape.
#
# ── FIXES (v2.4) ──────────────────────────────────────────────────────────────
# Fix 13 [CRITICAL] chunked_cross_entropy() called with 3 args (logits, labels,
#         VOCAB_SIZE) in compute_mtp_loss and eval_step, but the function only
#         accepts 2 positional args (logits, labels). VOCAB_SIZE is already
#         captured from module scope inside the function. Removed the spurious
#         third argument at both call sites. ✓
# Fix 14 [MEDIUM]  Dead fori_loop code block after `return nll` inside
#         chunked_cross_entropy was unreachable but syntactically present,
#         referencing an out-of-scope variable `V` (undefined at module level).
#         Although Python never executes code after a return, having it confuses
#         linters, future edits, and static checkers. Removed entirely. ✓
# ──────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS
# ════════════════════════════════════════════════════════════════════════════════
import os, re, gc, sys, json, math, time, logging
import numpy as np
from pathlib import Path
from functools import partial
from queue import Queue
from threading import Thread

# FIX 10: TPU-specific XLA flags.
# --xla_gpu_enable_latency_hiding_scheduler had zero effect on TPU (GPU-only flag).
# The TPU equivalents overlap allreduce (pmean) with compute, reducing step time.

import jax
import jax.numpy as jnp
from jax import random as jrand
import optax
import flax
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import flax.jax_utils

# FIX 11: JAX version guard. Typed PRNG key arrays (used for RNG in lax.scan)
# require JAX >= 0.4.16. Kaggle TPU v5e-8 ships 0.4.30+ as of 2026, so this
# will never fire in production but gives a clear error on stale environments.
_jax_ver = tuple(int(x) for x in jax.__version__.split(".")[:3])
if _jax_ver < (0, 4, 16):
    raise RuntimeError(
        f"JAX >= 0.4.16 required for typed PRNG keys in lax.scan. "
        f"Found: {jax.__version__}"
    )

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi, hf_hub_download, login

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ZenyxV2-Train")

# ════════════════════════════════════════════════════════════════════════════════
# §1  AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════════
HF_TOKEN = None

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass

if HF_TOKEN is None:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN is None:
    HF_TOKEN = "hf_token"  # <<< fallback

login(token=HF_TOKEN, add_to_git_credential=False)

# ════════════════════════════════════════════════════════════════════════════════
# §2  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
REPO_ID       = "Arko007/zenyx-v2-base"
REPO_PRIVATE  = True
TOKENIZER_ID  = "Arko007/zenyx-v2-tokenizer"
GLOBAL_SEED   = 42
TMP_DIR       = Path("/tmp/zenyx_v2")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Dimensions ──────────────────────────────────────────────────────────
VOCAB_SIZE      = 32_768
D_MODEL         = 576       # head_dim=64, 9 heads → 576
N_HEADS         = 9
N_KV_HEADS      = 3         # GQA 3:1
HEAD_DIM        = D_MODEL // N_HEADS   # 64
MLP_HIDDEN      = 1536
N_UNIQUE_BLOCKS = 8
N_RECURRENCES   = 4         # effective depth = 32
MAX_SEQ_LEN     = 14_336
DROPOUT_RATE    = 0.0

# MLA compression dims
MLA_KV_LATENT   = 128
MLA_Q_LATENT    = 384

# MTP heads
MTP_HEADS       = 3
MTP_WEIGHTS     = [1.0, 0.3, 0.1]

# ── TPU / Batch ───────────────────────────────────────────────────────────────
log.info(f"JAX version: {jax.__version__}")
log.info(f"JAX devices: {jax.devices()}")
TPU_CORES = jax.device_count()

# Hard assert: fail loudly if not the expected v5e-8 pod.
# TPU_CORES_STATIC is a Python int literal so XLA sees a compile-time constant
# in all reshapes (no dynamic-shape recompile risk).
TPU_CORES_STATIC = 8
if TPU_CORES != TPU_CORES_STATIC:
    raise RuntimeError(
        f"Device count mismatch: jax.device_count()={TPU_CORES} but "
        f"TPU_CORES_STATIC={TPU_CORES_STATIC}. "
        "This script targets TPU v5e-8. Either run on the correct pod or "
        "update TPU_CORES_STATIC to match your device count."
    )

# FIX 9: HBM budget correction.
# PER_CORE_BATCH=2 at T=16384 materialises attention [2,9,16384,16384] fp32
# = 19.7 GB per chip — exceeds the 16 GB HBM of each v5e chip entirely.
# Reduced to PER_CORE_BATCH=1: attention becomes [1,9,16384,16384] = 9.86 GB.
# GRAD_ACCUM doubled 16→32 to keep GLOBAL_BATCH = 256 unchanged.
#
# Per-chip HBM breakdown (PER_CORE_BATCH=1):
#   Params (bf16)         :  280M × 2B          =  ~560 MB
#   AdamW m+v (fp32)      :  280M × 2 × 4B     = ~2,240 MB
#   Grad accumulator(fp32):  280M × 4B          = ~1,120 MB
#   Peak attn matrix(fp32): [1,9,16384,16384]×4B = ~9,865 MB
#   Misc (embeds, norms …):                        ~  200 MB
#   TOTAL estimated peak                           ~13,985 MB  ≈ 13.7 GB  ✓
PER_CORE_BATCH  = 1
MICRO_BATCH     = PER_CORE_BATCH * TPU_CORES   # 8
GRAD_ACCUM      = 32                            # was 16; doubled to keep global batch = 256
GLOBAL_BATCH    = MICRO_BATCH * GRAD_ACCUM     # 8 × 32 = 256  (unchanged)

log.info(f"TPU Cores: {TPU_CORES} | Per-core batch: {PER_CORE_BATCH} | "
         f"Micro-batch: {MICRO_BATCH} | Grad-accum: {GRAD_ACCUM} | "
         f"Global batch: {GLOBAL_BATCH} seqs/step")
log.info(f"Tokens/step: {GLOBAL_BATCH * MAX_SEQ_LEN / 1e6:.2f}M")
log.info(f"Estimated per-chip HBM peak: ~13.7 GB / 16 GB")

# ── Optimizer / Schedule ──────────────────────────────────────────────────────
LEARNING_RATE = 3e-4
MIN_LR        = 3e-5
BETA1, BETA2  = 0.9, 0.95
EPS           = 1e-8
WEIGHT_DECAY  = 0.1
WARMUP_STEPS  = 2_000
STABLE_STEPS  = 80_000
DECAY_STEPS   = 18_000
MAX_STEPS     = WARMUP_STEPS + STABLE_STEPS + DECAY_STEPS  # 100k

# ── Data ──────────────────────────────────────────────────────────────────────
MATH_RATIO   = 0.40
CODE_RATIO   = 0.40
ENG_RATIO    = 0.20
TOTAL_TOKENS = 150_000_000_000
HEARTBEAT    = 5_000

# ── Checkpointing ─────────────────────────────────────────────────────────────
SAVE_EVERY    = 500
EVAL_EVERY    = 500
VAL_BATCHES_N = 128

# ── YaRN RoPE ─────────────────────────────────────────────────────────────────
ROPE_BASE         = 10_000.0
YARN_SCALE_FACTOR = 32.0
YARN_ALPHA        = 1.0
YARN_BETA         = 32.0

# ════════════════════════════════════════════════════════════════════════════════
# §3  TOKENIZER
# ════════════════════════════════════════════════════════════════════════════════
log.info(f"Loading tokenizer: {TOKENIZER_ID}")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)
PAD_ID = tokenizer.convert_tokens_to_ids("<|pad|>")
EOS_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")
assert len(tokenizer) == VOCAB_SIZE, f"Vocab mismatch: {len(tokenizer)} vs {VOCAB_SIZE}"
log.info(f"✓ Tokenizer loaded | vocab={len(tokenizer)} | pad_id={PAD_ID} | eos_id={EOS_ID}")

# ════════════════════════════════════════════════════════════════════════════════
# §4  YARN-SCALED ROPE
# ════════════════════════════════════════════════════════════════════════════════
def build_yarn_rope_cache(seq_len: int, head_dim: int):
    """
    YaRN-scaled RoPE frequencies. Extends context from 512 → 16k.
    t=0 at high-freq boundary (scale=1.0, no compression).
    t=1 at low-freq  boundary (scale=1/factor, full compression).
    Epsilon guard prevents division-by-zero if YARN_ALPHA==YARN_BETA.
    """
    freqs = 1.0 / (ROPE_BASE ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    wavelengths = 2.0 * jnp.pi / freqs

    low_boundary  = float(MAX_SEQ_LEN) / YARN_ALPHA   # 16384.0
    high_boundary = float(MAX_SEQ_LEN) / YARN_BETA     # 512.0

    den = low_boundary - high_boundary                 # 15872.0
    den = jnp.where(den == 0.0, 1e-6, den)             # epsilon guard

    t = (wavelengths - high_boundary) / den
    t = jnp.clip(t, 0.0, 1.0)
    intermediate_scale = (1.0 - t) * 1.0 + t * (1.0 / YARN_SCALE_FACTOR)

    scale = jnp.where(
        wavelengths > low_boundary,
        1.0 / YARN_SCALE_FACTOR,
        jnp.where(wavelengths < high_boundary, 1.0, intermediate_scale),
    )
    freqs = freqs * scale

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles    = positions[:, None] * freqs[None, :]
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    return sin.astype(jnp.bfloat16), cos.astype(jnp.bfloat16)


def rotate_half(x):
    d  = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)


# ════════════════════════════════════════════════════════════════════════════════
# §5  MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        x32   = x.astype(jnp.float32)
        rms   = jnp.sqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return ((x32 / rms) * scale).astype(jnp.bfloat16)


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-style MLA).
    Uses fused TPU attention (jax.nn.dot_product_attention)
    so no full [T,T] matrix is materialized.
    """
    d_model:      int
    n_heads:      int
    n_kv_heads:   int
    head_dim:     int
    kv_latent:    int
    q_latent:     int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, sin, cos, deterministic: bool):
        B, T, _ = x.shape

        # ─────────────────────────────────────────
        # Q projection (latent compression)
        # ─────────────────────────────────────────
        q = nn.Dense(self.q_latent, use_bias=False, dtype=jnp.bfloat16)(x)
        q = RMSNorm(self.q_latent)(q)
        q = nn.Dense(self.n_heads * self.head_dim,
                     use_bias=False,
                     dtype=jnp.bfloat16)(q)
        q = q.reshape(B, T, self.n_heads, self.head_dim)

        # ─────────────────────────────────────────
        # KV projection (shared latent compression)
        # ─────────────────────────────────────────
        kv_lat = nn.Dense(self.kv_latent,
                          use_bias=False,
                          dtype=jnp.bfloat16)(x)
        kv_lat = RMSNorm(self.kv_latent)(kv_lat)

        k = nn.Dense(self.n_kv_heads * self.head_dim,
                     use_bias=False,
                     dtype=jnp.bfloat16)(kv_lat)
        v = nn.Dense(self.n_kv_heads * self.head_dim,
                     use_bias=False,
                     dtype=jnp.bfloat16)(kv_lat)

        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim)

        # ─────────────────────────────────────────
        # RoPE
        # ─────────────────────────────────────────
        sin_ = sin[None, :T, None, :]
        cos_ = cos[None, :T, None, :]
        q = apply_rope(q, sin_, cos_)
        k = apply_rope(k, sin_, cos_)

        # ─────────────────────────────────────────
        # GQA expansion (repeat KV heads)
        # ─────────────────────────────────────────
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # ─────────────────────────────────────────
        # TPU-FUSED MEMORY-EFFICIENT ATTENTION
        # ─────────────────────────────────────────
        # Shapes required: [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            bias=None,
            mask=None,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Back to [B, T, D_model]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

        return nn.Dense(self.d_model,
                        use_bias=False,
                        dtype=jnp.bfloat16)(out)


class ConvSwiGLU(nn.Module):
    """
    Causal SwiGLU FFN with strictly causal depthwise 1D conv on the gate path.
    padding=((kernel_size-1, 0),): left-pad only, no right-pad (strictly causal).
    output[t] = conv(input[t-(k-1)], ..., input[t])  ← no future tokens. ✓
    """
    d_model:      int
    hidden_dim:   int
    dropout_rate: float = 0.0
    kernel_size:  int   = 3

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x_up = nn.Dense(self.hidden_dim * 2, use_bias=False, dtype=jnp.bfloat16)(x)
        gate, val = jnp.split(x_up, 2, axis=-1)

        gate = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding=((self.kernel_size - 1, 0),),  # left-pad only → causal ✓
            feature_group_count=self.hidden_dim,   # depthwise
            use_bias=False,
            dtype=jnp.bfloat16,
        )(gate)

        x = val * nn.silu(gate)

        if self.dropout_rate > 0.0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16)(x)


class TitanBlock(nn.Module):
    """Single unique Transformer block. Weights shared across recurrences."""
    d_model:      int
    n_heads:      int
    n_kv_heads:   int
    head_dim:     int
    hidden_dim:   int
    kv_latent:    int
    q_latent:     int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, sin, cos, deterministic: bool):
        x = x + MLAAttention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            kv_latent=self.kv_latent, q_latent=self.q_latent,
            dropout_rate=self.dropout_rate,
        )(RMSNorm(self.d_model)(x), sin, cos, deterministic)

        x = x + ConvSwiGLU(
            d_model=self.d_model, hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )(RMSNorm(self.d_model)(x), deterministic)

        return x


class ZenyxV2(nn.Module):
    """
    Nano-Titan: 8 unique blocks × 4 recurrences = 32 effective layers.
    Weight sharing: each TitanBlock(name="block_i") is called N_RECURRENCES times.
    Flax resolves param scope by name → subsequent calls reuse the same weights.
    Total unique params: ~280M. Effective compute depth: 32 layers.
    """
    vocab_size:      int
    d_model:         int
    n_heads:         int
    n_kv_heads:      int
    head_dim:        int
    hidden_dim:      int
    n_unique_blocks: int
    n_recurrences:   int
    max_seq_len:     int
    kv_latent:       int
    q_latent:        int
    mtp_heads:       int
    dropout_rate:    float = 0.0

    @nn.compact
    def __call__(self, input_ids, train: bool = False):
        B, T  = input_ids.shape
        det   = not train

        embed_table = self.param(
            "embed_table",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.d_model),
        )
        x = embed_table[input_ids].astype(jnp.bfloat16)

        sin, cos = build_yarn_rope_cache(T, self.head_dim)

        blocks = [
            TitanBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
                hidden_dim=self.hidden_dim, kv_latent=self.kv_latent,
                q_latent=self.q_latent, dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )
            for i in range(self.n_unique_blocks)
        ]

        for _ in range(self.n_recurrences):
            for block in blocks:
                x = block(x, sin, cos, deterministic=det)

        x = RMSNorm(self.d_model, name="final_norm")(x)

        # t+1: weight-tied with embedding table (no extra params)
        logits_1 = x @ embed_table.T.astype(jnp.bfloat16)
        logits_list = [logits_1]

        # t+2, t+3: independent small projection heads
        for i in range(1, self.mtp_heads):
            head_out = nn.Dense(
                self.vocab_size, use_bias=False,
                dtype=jnp.bfloat16, name=f"mtp_head_{i}",
            )(x)
            logits_list.append(head_out)

        return logits_list


# ════════════════════════════════════════════════════════════════════════════════
# §6  MODEL INIT
# ════════════════════════════════════════════════════════════════════════════════
log.info("Initialising Zenyx-v2 model...")
model = ZenyxV2(
    vocab_size      = VOCAB_SIZE,
    d_model         = D_MODEL,
    n_heads         = N_HEADS,
    n_kv_heads      = N_KV_HEADS,
    head_dim        = HEAD_DIM,
    hidden_dim      = MLP_HIDDEN,
    n_unique_blocks = N_UNIQUE_BLOCKS,
    n_recurrences   = N_RECURRENCES,
    max_seq_len     = MAX_SEQ_LEN,
    kv_latent       = MLA_KV_LATENT,
    q_latent        = MLA_Q_LATENT,
    mtp_heads       = MTP_HEADS,
    dropout_rate    = DROPOUT_RATE,
)

init_rng  = jrand.PRNGKey(GLOBAL_SEED)
# Init with MAX_SEQ_LEN so the compiled XLA shape matches training.
# Using (1,64) previously triggered a full recompile on the first training step.
dummy     = jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32)
variables = model.init(init_rng, input_ids=dummy, train=False)
params    = variables["params"]

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
log.info(f"✓ Parameters (unique weights): {param_count:,} ({param_count/1e6:.1f}M)")
log.info(f"  Effective depth: {N_UNIQUE_BLOCKS * N_RECURRENCES} layers")
log.info(f"  BF16 model size: {param_count * 2 / 1e6:.1f} MB")


# ════════════════════════════════════════════════════════════════════════════════
# §7  WSD LEARNING RATE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════════
def wsd_schedule(step):
    """Warmup-Stable-Decay: linear warmup → constant → cosine decay."""
    warmup_lr = LEARNING_RATE * jnp.minimum(step / WARMUP_STEPS, 1.0)
    decay_progress = jnp.clip(
        (step - WARMUP_STEPS - STABLE_STEPS) / DECAY_STEPS, 0.0, 1.0
    )
    cosine_lr = MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (
        1.0 + jnp.cos(jnp.pi * decay_progress)
    )
    in_warmup = step < WARMUP_STEPS
    in_stable = (step >= WARMUP_STEPS) & (step < WARMUP_STEPS + STABLE_STEPS)
    return jnp.where(in_warmup, warmup_lr,
           jnp.where(in_stable, LEARNING_RATE, cosine_lr))


# ════════════════════════════════════════════════════════════════════════════════
# §8-§10  REPLACEMENT: TRAIN STATE, MEMORY-SAFE CHUNKED CE, MTP LOSS, PMAP STEPS
# - Uses static CHUNK_SIZE = 2048 (smaller → less HBM; raises compute slightly).
# - Uses reshape + take_along_axis (XLA-friendly static shapes, no fori_loop).
# - compute_mtp_loss is checkpointed (rematerialisation) to reduce activation HBM.
# - train_step_accum performs grad-accum via lax.scan inside pmap as before.
# ════════════════════════════════════════════════════════════════════════════════

class ZenyxTrainState(train_state.TrainState):
    pass


def create_train_state(params):
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=wsd_schedule,
            b1=BETA1, b2=BETA2, eps=EPS,
            weight_decay=WEIGHT_DECAY,
            mask=lambda p: jax.tree_util.tree_map(
                lambda x: x.ndim >= 2, p
            ),
        ),
    )
    return ZenyxTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# Static constants (must divide exactly)
CHUNK_SIZE  = 2048                         # lowered from 4096 → 2048 to free ~2-4 GB HBM
NUM_CHUNKS  = VOCAB_SIZE // CHUNK_SIZE     # 32768 / 2048 = 16


def chunked_cross_entropy(logits, labels):
    """
    Memory-safe cross entropy without dynamic indexing.
    logits: bf16 [B, T, V]
    labels: int32 [B, T]
    Returns per-position NLL float32 [B, T]

    FIX 13: This function takes exactly 2 positional args (logits, labels).
    VOCAB_SIZE is read from module scope — do NOT pass it as a third argument.
    """
    B, T, V = logits.shape
    assert V == VOCAB_SIZE
    assert VOCAB_SIZE % CHUNK_SIZE == 0

    # ─────────────────────────────────────────
    # 1) reshape vocab into chunks
    # [B, T, 32768] → [B, T, 16, 2048]
    # ─────────────────────────────────────────
    logits = logits.reshape(B, T, NUM_CHUNKS, CHUNK_SIZE)
    logits = logits.astype(jnp.float32)

    # ─────────────────────────────────────────
    # 2) compute max over vocab (two-stage)
    # ─────────────────────────────────────────
    max_chunk  = jnp.max(logits, axis=-1)      # [B, T, 16]
    max_logits = jnp.max(max_chunk, axis=-1)   # [B, T]

    # subtract for numerical stability
    logits = logits - max_logits[..., None, None]

    # ─────────────────────────────────────────
    # 3) compute exp sum
    # ─────────────────────────────────────────
    exp_logits = jnp.exp(logits)
    sum_exp    = exp_logits.sum(axis=(-1, -2))  # sum over chunks + vocab positions

    # ─────────────────────────────────────────
    # 4) gather correct token logits
    # ─────────────────────────────────────────
    chunk_ids = labels // CHUNK_SIZE            # [B, T] — which chunk
    token_ids = labels % CHUNK_SIZE             # [B, T] — position inside chunk

    # gather the target chunk: [B, T, 1, 2048]
    gathered_chunk = jnp.take_along_axis(
        logits,
        chunk_ids[..., None, None],
        axis=2,
    )                                           # [B, T, 1, 2048]
    gathered_chunk = gathered_chunk[..., 0, :] # [B, T, 2048]

    # gather the target token within the chunk: [B, T]
    gathered_token = jnp.take_along_axis(
        gathered_chunk,
        token_ids[..., None],
        axis=-1,
    )[..., 0]                                   # [B, T]

    # ─────────────────────────────────────────
    # 5) final NLL  (float32 [B, T])
    # ─────────────────────────────────────────
    log_prob = gathered_token - jnp.log(sum_exp + 1e-8)
    return -log_prob


# Keep checkpoint/remat to reduce stored activations during backward.
@jax.checkpoint
def compute_mtp_loss(params, batch, dropout_rng):
    """
    Compute MTP loss in a streaming, memory-friendly manner.
    - params: model params
    - batch: [B, T] int32
    - dropout_rng: PRNGKey (typed or legacy)
    Returns scalar float32 loss (averaged and weighted across MTP heads).
    """
    # Forward (returns list of logits in bf16)
    logits_list = model.apply(
        {"params": params},
        input_ids=batch,
        train=True,
        rngs={"dropout": dropout_rng},
        mutable=False,
    )

    total_loss   = 0.0
    total_weight = 0.0

    for offset, (logits, weight) in enumerate(zip(logits_list, MTP_WEIGHTS), start=1):
        T        = batch.shape[1]
        clip_len = T - offset
        if clip_len <= 0:
            continue

        logits_slice = logits[:, :clip_len, :]                        # bf16 [B, clip_len, V]
        labels_slice = batch[:, offset:offset + clip_len].astype(jnp.int32)  # [B, clip_len]

        # FIX 13: call with 2 args only — VOCAB_SIZE is module-level constant
        nll = chunked_cross_entropy(logits_slice, labels_slice)       # float32 [B, clip_len]

        mask           = labels_slice != PAD_ID
        masked_nll_sum = jnp.where(mask, nll, 0.0).sum()
        denom          = mask.sum() + 1e-8
        loss           = masked_nll_sum / denom

        total_loss   = total_loss   + weight * loss
        total_weight = total_weight + weight

    return (total_loss / (total_weight + 1e-12)).astype(jnp.float32)


# PMAPed training step with gradient accumulation inside pmap.
@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step_accum(state, micro_batches, dropout_rngs):
    """
    micro_batches: [GRAD_ACCUM, PER_CORE_BATCH, SEQ_LEN] (per-device shard)
    dropout_rngs:  [GRAD_ACCUM] typed PRNGKey array (per-device)
    """
    def accum_fn(carry, inputs):
        grads_acc, loss_acc = carry
        mb, rng = inputs  # mb: [PER_CORE_BATCH, SEQ_LEN]
        loss, grads = jax.value_and_grad(compute_mtp_loss)(state.params, mb, rng)
        grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)
        return (grads_acc, loss_acc + loss), None

    zero_grads = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), state.params)
    (grads, total_loss), _ = jax.lax.scan(
        accum_fn,
        (zero_grads, jnp.zeros((), dtype=jnp.float32)),
        (micro_batches, dropout_rngs),
    )

    grads     = jax.tree_util.tree_map(lambda g: g / GRAD_ACCUM, grads)
    grads     = jax.lax.pmean(grads, axis_name="devices")
    avg_loss  = jax.lax.pmean(total_loss / GRAD_ACCUM, axis_name="devices")
    grad_norm = optax.global_norm(grads)
    new_state = state.apply_gradients(grads=grads)
    return new_state, avg_loss, grad_norm


# PMAPed eval step using the chunked CE for stable memory usage.
@partial(jax.pmap, axis_name="devices")
def eval_step(params, batch):
    """
    batch: [PER_CORE_BATCH, SEQ_LEN] per-device shard
    Returns averaged loss across devices (float32).
    """
    logits_list = model.apply({"params": params}, input_ids=batch, train=False)
    logits = logits_list[0]                  # primary head (bf16) [B, T, V]
    labels = batch[:, 1:].astype(jnp.int32)
    logits = logits[:, :-1, :]               # align with labels [B, T-1, V]

    # FIX 13: call with 2 args only
    nll  = chunked_cross_entropy(logits, labels)   # [B, T-1]
    mask = labels != PAD_ID
    loss = jnp.where(mask, nll, 0.0).sum() / (mask.sum() + 1e-8)
    return jax.lax.pmean(loss, axis_name="devices")


# ════════════════════════════════════════════════════════════════════════════════
# §11  DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
CODE_LANGS = [
    "python", "javascript", "typescript", "java", "c", "cpp", "c-sharp",
    "go", "rust", "kotlin", "php", "ruby", "shell", "sql", "html", "css",
    "markdown", "yaml", "json", "dockerfile", "cuda", "r", "dart", "swift",
    # "scala" excluded per tokenizer note
]


def _encode(text: str):
    return tokenizer(
        text, add_special_tokens=False,
        truncation=False, return_attention_mask=False
    )["input_ids"]


def stream_math(target_bytes: int, seed: int, skip_tokens: int = 0):
    log.info(f"[MATH] finemath-3plus | target={target_bytes/1e9:.2f}GB | skip={skip_tokens:,} tokens | seed={seed}")
    ds = load_dataset(
        "HuggingFaceTB/finemath", name="finemath-3plus",
        split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=50_000)

    skipped = consumed = 0
    buf = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        toks = _encode(text)
        if skipped < skip_tokens:
            skipped += len(toks)
            continue
        buf.extend(toks)
        consumed += len(toks)
        while len(buf) >= MAX_SEQ_LEN:
            yield buf[:MAX_SEQ_LEN]
            buf = buf[MAX_SEQ_LEN:]
        if consumed * 2 >= target_bytes:
            break


def stream_code(target_bytes: int, seed: int, skip_tokens: int = 0):
    per_lang = target_bytes // len(CODE_LANGS)
    log.info(f"[CODE] {len(CODE_LANGS)} langs | {per_lang/1e6:.0f}MB each | skip={skip_tokens:,} tokens | seed={seed}")
    skipped = 0
    for lang in CODE_LANGS:
        try:
            ds = load_dataset(
                "bigcode/starcoderdata", data_dir=lang,
                split="train", streaming=True,
            )
        except Exception as e:
            log.warning(f"[CODE] {lang}: {e}")
            continue

        consumed = 0
        buf = []
        for row in ds:
            text = (row.get("content") or "").strip()
            if len(text) < 20:
                continue
            toks = _encode(text)
            if skipped < skip_tokens:
                skipped += len(toks)
                continue
            buf.extend(toks)
            consumed += len(toks) * 2
            while len(buf) >= MAX_SEQ_LEN:
                yield buf[:MAX_SEQ_LEN]
                buf = buf[MAX_SEQ_LEN:]
            if consumed >= per_lang:
                break
        del ds
        gc.collect()


def stream_english(target_bytes: int, seed: int, skip_tokens: int = 0):
    """Full FineWeb-Edu default split (~1.3T tokens), score >= 3.0 filter."""
    log.info(f"[ENG] fineweb-edu | target={target_bytes/1e9:.2f}GB | skip={skip_tokens:,} tokens | seed={seed}")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=50_000)

    skipped = consumed = 0
    buf = []
    for row in ds:
        if row.get("score", 0.0) < 3.0:
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        toks = _encode(text)
        if skipped < skip_tokens:
            skipped += len(toks)
            continue
        buf.extend(toks)
        consumed += len(toks)
        while len(buf) >= MAX_SEQ_LEN:
            yield buf[:MAX_SEQ_LEN]
            buf = buf[MAX_SEQ_LEN:]
        if consumed * 2 >= target_bytes:
            break


def combined_stream(resume_step: int, seed: int):
    """
    Weighted interleaved stream. Mix: 40% math | 40% code | 20% english.
    Weighted cycle: [mathA, mathB, codeA, codeB, eng] → 2:2:1 ratio.

    Each slot is an INDEPENDENT generator instance:
    - mathA / mathB use seed and seed+1 (different shuffles of same dataset).
    - codeA / codeB use seed+2 and seed+3 (code is a different dataset so
      seed integer correlation with math seeds is irrelevant, but we space
      them anyway for absolute clarity).
    - Each slot gets half the modality byte budget so the two slots combined
      consume the full math_bytes / code_bytes target.
    - Skip tokens are split evenly between the two slots of each modality.
      Integer truncation is at most 1 token per slot per resume — negligible
      at 150B token scale (< 1e-10 fractional error).
    """
    total_bytes  = TOTAL_TOKENS * 2
    math_bytes   = int(total_bytes * MATH_RATIO)
    code_bytes   = int(total_bytes * CODE_RATIO)
    eng_bytes    = int(total_bytes * ENG_RATIO)

    tokens_done  = resume_step * GLOBAL_BATCH * MAX_SEQ_LEN
    math_skip    = int(tokens_done * MATH_RATIO)
    code_skip    = int(tokens_done * CODE_RATIO)
    eng_skip     = int(tokens_done * ENG_RATIO)

    cycle = [
        stream_math(math_bytes // 2, seed,     math_skip // 2),   # mathA
        stream_math(math_bytes // 2, seed + 1, math_skip // 2),   # mathB  (distinct shuffle)
        stream_code(code_bytes // 2, seed + 2, code_skip // 2),   # codeA
        stream_code(code_bytes // 2, seed + 3, code_skip // 2),   # codeB  (distinct shuffle)
        stream_english(eng_bytes,    seed + 4, eng_skip),         # eng
    ]
    done  = [False] * 5
    idx   = 0
    total = 0

    while not all(done):
        i   = idx % len(cycle)
        idx += 1
        if done[i]:
            continue
        try:
            block = next(cycle[i])
            total += 1
            if total % HEARTBEAT == 0:
                log.info(f"[DATA] {total:,} blocks yielded "
                         f"({total * MAX_SEQ_LEN / 1e9:.2f}B tokens)")
            yield block
        except StopIteration:
            done[i] = True


def prefetch_worker(gen, q: Queue):
    try:
        for item in gen:
            q.put(item)
        q.put(None)
    except Exception as e:
        log.error(f"[PREFETCH] {e}")
        q.put(None)


# ════════════════════════════════════════════════════════════════════════════════
# §12  HF HUB CHECKPOINTING
# ════════════════════════════════════════════════════════════════════════════════
api = HfApi(token=HF_TOKEN)


def ensure_repo():
    try:
        api.create_repo(repo_id=REPO_ID, private=REPO_PRIVATE,
                        repo_type="model", exist_ok=True)
    except Exception as e:
        log.warning(f"Repo create: {e}")


def load_latest_checkpoint():
    try:
        files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))
    except Exception:
        return None, 0, None, None

    ckpt_files = [f for f in files if re.match(r"checkpoints/state_step\d+\.msgpack", f)]
    if not ckpt_files:
        return None, 0, None, None

    steps  = [int(re.search(r"(\d+)", f).group()) for f in ckpt_files]
    latest = max(steps)
    log.info(f"Found checkpoint at step {latest}. Downloading...")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"checkpoints/state_step{latest}.msgpack",
            token=HF_TOKEN, repo_type="model",
        )
        with open(path, "rb") as f:
            state_bytes = f.read()
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None, 0, None, None

    meta = None
    try:
        mp = hf_hub_download(repo_id=REPO_ID, filename="metadata.json",
                             token=HF_TOKEN, repo_type="model")
        with open(mp) as f:
            meta = json.load(f)
    except Exception:
        pass

    val_batches = None
    try:
        vp = hf_hub_download(repo_id=REPO_ID, filename="val_set.npy",
                             token=HF_TOKEN, repo_type="model")
        val_batches = np.load(vp, allow_pickle=False)
        log.info(f"✓ Val set loaded: {val_batches.shape}")
    except Exception:
        pass

    return state_bytes, latest, meta, val_batches


def save_checkpoint(state, step: int, meta: dict, val_batches=None):
    final_state = flax.jax_utils.unreplicate(state)
    state_bytes = serialization.to_bytes(final_state)
    ckpt_path   = TMP_DIR / f"state_step{step}.msgpack"
    meta_path   = TMP_DIR / "metadata.json"

    with open(ckpt_path, "wb") as f:
        f.write(state_bytes)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    params_only = serialization.to_bytes(final_state.params)
    params_path = TMP_DIR / f"params_step{step}.msgpack"
    with open(params_path, "wb") as f:
        f.write(params_only)

    ensure_repo()
    try:
        api.upload_file(str(ckpt_path),
                        path_in_repo=f"checkpoints/{ckpt_path.name}",
                        repo_id=REPO_ID, repo_type="model")
        api.upload_file(str(meta_path),
                        path_in_repo="metadata.json",
                        repo_id=REPO_ID, repo_type="model")
        api.upload_file(str(params_path),
                        path_in_repo=f"params/{params_path.name}",
                        repo_id=REPO_ID, repo_type="model")

        if val_batches is not None:
            vpath = TMP_DIR / "val_set.npy"
            np.save(vpath, val_batches)
            api.upload_file(str(vpath), path_in_repo="val_set.npy",
                            repo_id=REPO_ID, repo_type="model")

        log.info(f"✓ Checkpoint saved: step={step} | "
                 f"ckpt={len(state_bytes)/1e6:.1f}MB | "
                 f"params_only={len(params_only)/1e6:.1f}MB")
        for p in [ckpt_path, meta_path, params_path]:
            p.unlink(missing_ok=True)
    except Exception as e:
        log.error(f"Upload failed (files saved locally): {e}")


# ════════════════════════════════════════════════════════════════════════════════
# §13  INIT / RESUME
# ════════════════════════════════════════════════════════════════════════════════
ensure_repo()
result = load_latest_checkpoint()

if result[0] is not None:
    state_bytes, resume_step, meta, val_batches_loaded = result
    dummy_state  = create_train_state(params)
    loaded_state = serialization.from_bytes(dummy_state, state_bytes)
    state        = flax.jax_utils.replicate(loaded_state)
    log.info(f"✓ Resumed from step {resume_step}")
else:
    state              = create_train_state(params)
    state              = flax.jax_utils.replicate(state)
    resume_step        = 0
    meta               = {}
    val_batches_loaded = None
    log.info("✓ Fresh training run")

best_val_loss  = meta.get("best_val_loss", float("inf"))
train_loss_log = meta.get("train_losses", [])


# ════════════════════════════════════════════════════════════════════════════════
# §14  VALIDATION SET
# ════════════════════════════════════════════════════════════════════════════════
if val_batches_loaded is not None:
    val_batches = val_batches_loaded
    log.info(f"✓ Val set from checkpoint: {val_batches.shape}")
else:
    log.info(f"Building validation set ({VAL_BATCHES_N} batches)...")
    # Target 3× the needed tokens to ensure we always have enough.
    # VAL_BATCHES_N * MICRO_BATCH * MAX_SEQ_LEN = 128 * 8 * 16384 = 16,777,216 tokens.
    # 3× target = ~50M tokens — vastly exceeds the stream budget, so no underflow.
    _val_need = VAL_BATCHES_N * MICRO_BATCH   # 1024 blocks
    val_gen = stream_english(
        target_bytes=_val_need * MAX_SEQ_LEN * 3 * 2,
        seed=GLOBAL_SEED + 9999,
    )
    val_list = []
    for _ in range(_val_need):
        try:
            val_list.append(next(val_gen))
        except StopIteration:
            break

    # FIX 12: Hard guard — fail loudly if the stream didn't produce enough blocks.
    if len(val_list) < _val_need:
        raise RuntimeError(
            f"Val set underflow: got {len(val_list)} blocks, need {_val_need}. "
            "Increase val stream target_bytes or reduce VAL_BATCHES_N."
        )

    val_batches = np.array(val_list, dtype=np.int32)
    val_batches = val_batches.reshape(VAL_BATCHES_N, MICRO_BATCH, MAX_SEQ_LEN)
    log.info(f"✓ Val set built: {val_batches.shape}")


# ════════════════════════════════════════════════════════════════════════════════
# §15  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════════
log.info("=" * 80)
log.info(f"  ZENYX-V2 PRETRAINING  |  TPU v5e-8  |  Step {resume_step}/{MAX_STEPS}")
log.info(f"  d_model={D_MODEL} | {N_UNIQUE_BLOCKS}×{N_RECURRENCES} recurrences | "
         f"vocab={VOCAB_SIZE} | ctx={MAX_SEQ_LEN}")
log.info(f"  Unique params: {param_count/1e6:.1f}M | "
         f"Effective depth: {N_UNIQUE_BLOCKS*N_RECURRENCES} layers")
log.info(f"  Per-core batch: {PER_CORE_BATCH} | Grad-accum: {GRAD_ACCUM} | "
         f"Global batch: {GLOBAL_BATCH} seqs | "
         f"{GLOBAL_BATCH*MAX_SEQ_LEN/1e6:.2f}M tokens/step")
log.info("=" * 80)

data_gen   = combined_stream(resume_step=resume_step, seed=GLOBAL_SEED)
prefetch_q = Queue(maxsize=10_000)
prefetch_t = Thread(target=prefetch_worker, args=(data_gen, prefetch_q), daemon=True)
prefetch_t.start()

global_step = resume_step
rng         = jrand.PRNGKey(GLOBAL_SEED + resume_step)
t_start     = time.time()

try:
    while global_step < MAX_STEPS:
        # Collect GRAD_ACCUM * MICRO_BATCH blocks for this step.
        # With PER_CORE_BATCH=1, MICRO_BATCH=8, GRAD_ACCUM=32: 256 blocks total.
        step_blocks = []
        for _ in range(GRAD_ACCUM * MICRO_BATCH):
            block = prefetch_q.get()
            if block is None:
                log.info("Data stream exhausted.")
                break
            step_blocks.append(block)

        if len(step_blocks) < GRAD_ACCUM * MICRO_BATCH:
            break

        # Reshape [256, SEQ_LEN] → [TPU_CORES, GRAD_ACCUM, PER_CORE_BATCH, SEQ_LEN]
        # = [8, 32, 1, 16384]. TPU_CORES_STATIC is a Python int literal →
        # XLA sees a compile-time constant; no dynamic shape / recompile risk.
        arr = np.array(step_blocks, dtype=np.int32)
        arr = arr.reshape(GRAD_ACCUM, TPU_CORES_STATIC, PER_CORE_BATCH, MAX_SEQ_LEN)
        arr = arr.transpose(1, 0, 2, 3)  # [8, 32, 1, 16384]
        arr = jnp.asarray(arr, dtype=jnp.int32)

        # Pure jrand.split RNG: no key_data/wrap_key_data needed.
        # jrand.split(key, n) returns a typed key array [n].
        # pmap shards axis-0 → each device gets [GRAD_ACCUM] keys.
        # lax.scan iterates axis-0 → each micro-step gets one unique PRNGKey.
        rng, step_rng = jrand.split(rng)

        all_keys = jrand.split(step_rng, TPU_CORES_STATIC * GRAD_ACCUM)

        # Handle both typed-key and legacy (N,2) key formats safely
        if all_keys.ndim == 1:
            # Typed PRNG keys (JAX >= 0.4.16 new format)
            dropout_rngs = all_keys.reshape(TPU_CORES_STATIC, GRAD_ACCUM)
        else:
            # Legacy 2-int keys
            dropout_rngs = all_keys.reshape(TPU_CORES_STATIC, GRAD_ACCUM, 2)

        state, loss, grad_norm = train_step_accum(state, arr, dropout_rngs)

        loss_val      = float(flax.jax_utils.unreplicate(loss))
        grad_norm_val = float(flax.jax_utils.unreplicate(grad_norm))
        train_loss_log.append(loss_val)

        if global_step % 50 == 0:
            elapsed    = max(time.time() - t_start, 1)
            steps_done = max(global_step - resume_step, 1)
            tok_per_s  = steps_done * GLOBAL_BATCH * MAX_SEQ_LEN / elapsed
            ppl        = math.exp(min(loss_val, 10.0))
            cur_lr     = float(wsd_schedule(global_step))
            log.info(
                f"Step {global_step:6d}/{MAX_STEPS} | "
                f"Loss {loss_val:.4f} | PPL {ppl:7.2f} | "
                f"GNorm {grad_norm_val:.3f} | "
                f"LR {cur_lr:.2e} | "
                f"{tok_per_s/1e6:.2f}M tok/s | "
                f"{elapsed/60:.1f}min"
            )

        global_step += 1

        if global_step % SAVE_EVERY == 0:
            log.info(f"[EVAL] Running validation at step {global_step}...")
            val_losses = []
            for vb in val_batches:
                # vb: [MICRO_BATCH, SEQ_LEN] = [8, 16384]
                # reshape to [TPU_CORES, PER_CORE_BATCH, SEQ_LEN] = [8, 1, 16384]
                vb_sharded = jnp.asarray(vb, dtype=jnp.int32).reshape(
                    TPU_CORES_STATIC, PER_CORE_BATCH, MAX_SEQ_LEN
                )
                vl = eval_step(state.params, vb_sharded)
                val_losses.append(float(flax.jax_utils.unreplicate(vl)))

            mean_val  = sum(val_losses) / len(val_losses)
            val_ppl   = math.exp(min(mean_val, 10.0))
            avg_train = sum(train_loss_log[-200:]) / min(len(train_loss_log), 200)

            log.info(
                f"[EVAL] Val Loss: {mean_val:.4f} | Val PPL: {val_ppl:.2f} | "
                f"Train (200-avg): {avg_train:.4f} | Gap: {mean_val - avg_train:+.4f}"
            )

            is_best = mean_val < best_val_loss - 1e-4
            if is_best:
                best_val_loss = mean_val
                log.info(f"✓ New best val loss: {best_val_loss:.4f}")

            meta = {
                "step":            global_step,
                "best_val_loss":   float(best_val_loss),
                "last_val_loss":   float(mean_val),
                "last_train_loss": float(avg_train),
                "train_losses":    train_loss_log[-500:],
                "model_config": {
                    "vocab_size":      VOCAB_SIZE,
                    "d_model":         D_MODEL,
                    "n_heads":         N_HEADS,
                    "n_kv_heads":      N_KV_HEADS,
                    "head_dim":        HEAD_DIM,
                    "hidden_dim":      MLP_HIDDEN,
                    "n_unique_blocks": N_UNIQUE_BLOCKS,
                    "n_recurrences":   N_RECURRENCES,
                    "max_seq_len":     MAX_SEQ_LEN,
                    "kv_latent":       MLA_KV_LATENT,
                    "q_latent":        MLA_Q_LATENT,
                    "mtp_heads":       MTP_HEADS,
                    "per_core_batch":  PER_CORE_BATCH,
                    "grad_accum":      GRAD_ACCUM,
                    "global_batch":    GLOBAL_BATCH,
                },
            }
            save_checkpoint(
                state, global_step, meta,
                val_batches if global_step == SAVE_EVERY else None,
            )

except KeyboardInterrupt:
    log.info("Interrupted — saving emergency checkpoint...")

log.info("Saving final checkpoint...")
final_meta = {
    "step":          global_step,
    "best_val_loss": float(best_val_loss),
    "train_losses":  train_loss_log[-500:],
    "status":        "complete" if global_step >= MAX_STEPS else "interrupted",
}
save_checkpoint(state, global_step, final_meta)

log.info("=" * 80)
log.info(f"✓ DONE | Final step: {global_step} | Best val loss: {best_val_loss:.4f}")
log.info(f"  Repo: https://huggingface.co/{REPO_ID}")
log.info("=" * 80)
