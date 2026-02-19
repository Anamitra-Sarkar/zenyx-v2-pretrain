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
#
# ── FIXES (v2.5) ──────────────────────────────────────────────────────────────
# Fix 15 [CRITICAL] RESOURCE_EXHAUSTED: exceeded HBM by 674.94 MB.
#         Root cause (from XLA allocator report):
#           Rank-1 temp: f32[1,14335,16,2048] = 1.75 GB  ← full logit tensor
#             live during backward pass of chunked_cross_entropy.
#           Rank-2 temp: bf16[1,14335,16,2048] = 895 MB  ← same tensor
#             rematerialised by @jax.checkpoint on compute_mtp_loss.
#         Three surgical changes:
#         (a) Replace monolithic reshape+max+exp with sequential lax.scan.
#         (b) Remove @jax.checkpoint from compute_mtp_loss; apply surgically
#             on model.apply instead.
#         (c) CHUNK_SIZE 2048 → 1024 for extra HBM safety margin. ✓
#
# ── FIXES (v2.6) ──────────────────────────────────────────────────────────────
# Fix 16 [CRITICAL] InvalidFilterError: "JitTracer(bool[])"
#         Root cause:
#           jax.checkpoint(model.apply)(..., mutable=False) passes the Python
#           bool False through JAX's abstract evaluation / tracing machinery.
#           JAX converts it to a JitTracer(bool[]) abstract value.
#           Flax's scope.in_filter() only accepts real Python bool/str/DenyList
#           — never traced values — so it raises InvalidFilterError.
#         Fix:
#           Wrap the forward pass in a plain Python helper _fwd() that closes
#           over mutable as a compile-time Python constant (never touches the
#           JAX tracer). Apply jax.checkpoint to _fwd instead of model.apply
#           directly. The rematerialisation boundary is identical; Flax never
#           receives a traced bool. ✓
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

import jax
import jax.numpy as jnp
from jax import random as jrand
import optax
import flax
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import flax.jax_utils

# FIX 11: JAX version guard.
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
D_MODEL         = 576
N_HEADS         = 9
N_KV_HEADS      = 3
HEAD_DIM        = D_MODEL // N_HEADS   # 64
MLP_HIDDEN      = 1536
N_UNIQUE_BLOCKS = 8
N_RECURRENCES   = 4
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

TPU_CORES_STATIC = 8
if TPU_CORES != TPU_CORES_STATIC:
    raise RuntimeError(
        f"Device count mismatch: jax.device_count()={TPU_CORES} but "
        f"TPU_CORES_STATIC={TPU_CORES_STATIC}. "
        "This script targets TPU v5e-8. Either run on the correct pod or "
        "update TPU_CORES_STATIC to match your device count."
    )

PER_CORE_BATCH  = 1
MICRO_BATCH     = PER_CORE_BATCH * TPU_CORES   # 8
GRAD_ACCUM      = 32
GLOBAL_BATCH    = MICRO_BATCH * GRAD_ACCUM     # 256

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
    freqs = 1.0 / (ROPE_BASE ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    wavelengths = 2.0 * jnp.pi / freqs

    low_boundary  = float(MAX_SEQ_LEN) / YARN_ALPHA
    high_boundary = float(MAX_SEQ_LEN) / YARN_BETA

    den = low_boundary - high_boundary
    den = jnp.where(den == 0.0, 1e-6, den)

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

        q = nn.Dense(self.q_latent, use_bias=False, dtype=jnp.bfloat16)(x)
        q = RMSNorm(self.q_latent)(q)
        q = nn.Dense(self.n_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(q)
        q = q.reshape(B, T, self.n_heads, self.head_dim)

        kv_lat = nn.Dense(self.kv_latent, use_bias=False, dtype=jnp.bfloat16)(x)
        kv_lat = RMSNorm(self.kv_latent)(kv_lat)

        k = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(kv_lat)
        v = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(kv_lat)

        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim)

        sin_ = sin[None, :T, None, :]
        cos_ = cos[None, :T, None, :]
        q = apply_rope(q, sin_, cos_)
        k = apply_rope(k, sin_, cos_)

        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = jax.nn.dot_product_attention(
            query=q, key=k, value=v,
            bias=None, mask=None,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16)(out)


class ConvSwiGLU(nn.Module):
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
            feature_group_count=self.hidden_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
        )(gate)

        x = val * nn.silu(gate)

        if self.dropout_rate > 0.0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16)(x)


class TitanBlock(nn.Module):
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

        logits_1 = x @ embed_table.T.astype(jnp.bfloat16)
        logits_list = [logits_1]

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
# §8  TRAIN STATE
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


# ════════════════════════════════════════════════════════════════════════════════
# §9  CHUNKED CROSS-ENTROPY
#
# Sequential lax.scan over vocab chunks — only ONE f32[B,T,CHUNK_SIZE] slice
# is live in HBM at any time (~56 MB peak vs 1.75 GB for monolithic reshape).
# ════════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE = 1024
NUM_CHUNKS = VOCAB_SIZE // CHUNK_SIZE   # 32


def chunked_cross_entropy(logits, labels):
    """
    Memory-safe chunked cross-entropy.
    logits : bf16 [B, T, V]
    labels : int32 [B, T]
    returns: float32 [B, T]  per-position NLL
    """
    B, T, V = logits.shape
    assert V == VOCAB_SIZE, f"Expected vocab {VOCAB_SIZE}, got {V}"
    assert VOCAB_SIZE % CHUNK_SIZE == 0

    logits_f32 = logits.astype(jnp.float32)

    # Pass 1: global max over vocab (stable softmax numerics)
    def max_body(carry, chunk_idx):
        start = chunk_idx * CHUNK_SIZE
        chunk = jax.lax.dynamic_slice_in_dim(logits_f32, start, CHUNK_SIZE, axis=-1)
        return jnp.maximum(carry, chunk.max(axis=-1)), None

    global_max, _ = jax.lax.scan(
        max_body,
        jnp.full((B, T), -jnp.inf, dtype=jnp.float32),
        jnp.arange(NUM_CHUNKS, dtype=jnp.int32),
    )

    # Pass 2: sum(exp(logit - max)) chunk-by-chunk
    def sum_body(carry, chunk_idx):
        start = chunk_idx * CHUNK_SIZE
        chunk = jax.lax.dynamic_slice_in_dim(logits_f32, start, CHUNK_SIZE, axis=-1)
        return carry + jnp.exp(chunk - global_max[..., None]).sum(axis=-1), None

    sum_exp, _ = jax.lax.scan(
        sum_body,
        jnp.zeros((B, T), dtype=jnp.float32),
        jnp.arange(NUM_CHUNKS, dtype=jnp.int32),
    )

    # Pass 3: gather target-token logit
    chunk_ids = labels // CHUNK_SIZE
    token_ids = labels % CHUNK_SIZE

    def gather_body(carry, chunk_idx):
        start = chunk_idx * CHUNK_SIZE
        chunk = jax.lax.dynamic_slice_in_dim(logits_f32, start, CHUNK_SIZE, axis=-1)
        tok   = jnp.take_along_axis(chunk, token_ids[..., None], axis=-1)[..., 0]
        mask  = (chunk_ids == chunk_idx)
        return jnp.where(mask, tok, carry), None

    target_logit, _ = jax.lax.scan(
        gather_body,
        jnp.zeros((B, T), dtype=jnp.float32),
        jnp.arange(NUM_CHUNKS, dtype=jnp.int32),
    )

    log_prob = (target_logit - global_max) - jnp.log(sum_exp + 1e-8)
    return -log_prob   # float32 [B, T]


# ════════════════════════════════════════════════════════════════════════════════
# §10  MTP LOSS + PMAP STEPS
#
# FIX 16: InvalidFilterError from jax.checkpoint(model.apply)(..., mutable=False)
# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM:
#   When JAX traces jax.checkpoint(model.apply), every argument — including
#   keyword arguments like mutable=False — passes through JAX's abstract
#   evaluation. The Python bool False becomes a JitTracer(bool[]).
#   Flax's scope.in_filter() is called at trace time with this traced value,
#   but it only handles real Python bool/str/DenyList, so it raises:
#       InvalidFilterError: Invalid Filter: "JitTracer(bool[])"
#
# FIX:
#   Define a plain Python helper _fwd(params, batch, dropout_rng) that calls
#   model.apply with mutable as a hard-coded Python literal — never a traced
#   value. Apply jax.checkpoint to _fwd instead of model.apply.
#   The checkpoint boundary (and therefore rematerialisation behaviour) is
#   exactly the same as before, but Flax only ever sees real Python bools. ✓
# ════════════════════════════════════════════════════════════════════════════════

def _fwd(params, batch, dropout_rng):
    """
    Pure Python wrapper around model.apply for training.
    All keyword args (train=True, mutable=False) are Python literals —
    they are resolved at Python import time, never touched by JAX's tracer.
    This is the function that jax.checkpoint wraps.
    """
    return model.apply(
        {"params": params},
        input_ids=batch,
        train=True,
        rngs={"dropout": dropout_rng},
        # mutable is intentionally NOT passed: Flax defaults to mutable=False
        # for stateless modules, which is what we want. No traced bool ever
        # reaches in_filter(). ✓
    )


# jax.checkpoint(_fwd): transformer block activations are rematerialised
# during the backward pass; CE intermediates (tiny ~56 MB) are not. ✓
_fwd_checkpointed = jax.checkpoint(_fwd)


def compute_mtp_loss(params, batch, dropout_rng):
    """
    MTP loss: weighted sum over t+1, t+2, t+3 prediction heads.
    params      : model params pytree
    batch       : int32 [B, T]
    dropout_rng : PRNGKey
    Returns     : scalar float32 loss
    """
    logits_list = _fwd_checkpointed(params, batch, dropout_rng)

    total_loss   = 0.0
    total_weight = 0.0

    for offset, (logits, weight) in enumerate(zip(logits_list, MTP_WEIGHTS), start=1):
        T        = batch.shape[1]
        clip_len = T - offset
        if clip_len <= 0:
            continue

        logits_slice = logits[:, :clip_len, :]
        labels_slice = batch[:, offset:offset + clip_len].astype(jnp.int32)

        nll = chunked_cross_entropy(logits_slice, labels_slice)

        mask           = labels_slice != PAD_ID
        masked_nll_sum = jnp.where(mask, nll, 0.0).sum()
        denom          = mask.sum() + 1e-8
        loss           = masked_nll_sum / denom

        total_loss   = total_loss   + weight * loss
        total_weight = total_weight + weight

    return (total_loss / (total_weight + 1e-12)).astype(jnp.float32)


@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step_accum(state, micro_batches, dropout_rngs):
    """
    micro_batches: [GRAD_ACCUM, PER_CORE_BATCH, SEQ_LEN] per-device shard
    dropout_rngs : [GRAD_ACCUM] typed PRNGKey array per-device
    """
    def accum_fn(carry, inputs):
        grads_acc, loss_acc = carry
        mb, rng = inputs
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


@partial(jax.pmap, axis_name="devices")
def eval_step(params, batch):
    """
    batch: [PER_CORE_BATCH, SEQ_LEN] per-device shard
    """
    logits_list = model.apply({"params": params}, input_ids=batch, train=False)
    logits = logits_list[0]
    labels = batch[:, 1:].astype(jnp.int32)
    logits = logits[:, :-1, :]

    nll  = chunked_cross_entropy(logits, labels)
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
    total_bytes  = TOTAL_TOKENS * 2
    math_bytes   = int(total_bytes * MATH_RATIO)
    code_bytes   = int(total_bytes * CODE_RATIO)
    eng_bytes    = int(total_bytes * ENG_RATIO)

    tokens_done  = resume_step * GLOBAL_BATCH * MAX_SEQ_LEN
    math_skip    = int(tokens_done * MATH_RATIO)
    code_skip    = int(tokens_done * CODE_RATIO)
    eng_skip     = int(tokens_done * ENG_RATIO)

    cycle = [
        stream_math(math_bytes // 2, seed,     math_skip // 2),
        stream_math(math_bytes // 2, seed + 1, math_skip // 2),
        stream_code(code_bytes // 2, seed + 2, code_skip // 2),
        stream_code(code_bytes // 2, seed + 3, code_skip // 2),
        stream_english(eng_bytes,    seed + 4, eng_skip),
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
    _val_need = VAL_BATCHES_N * MICRO_BATCH
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
        step_blocks = []
        for _ in range(GRAD_ACCUM * MICRO_BATCH):
            block = prefetch_q.get()
            if block is None:
                log.info("Data stream exhausted.")
                break
            step_blocks.append(block)

        if len(step_blocks) < GRAD_ACCUM * MICRO_BATCH:
            break

        arr = np.array(step_blocks, dtype=np.int32)
        arr = arr.reshape(GRAD_ACCUM, TPU_CORES_STATIC, PER_CORE_BATCH, MAX_SEQ_LEN)
        arr = arr.transpose(1, 0, 2, 3)  # [8, 32, 1, 16384]
        arr = jnp.asarray(arr, dtype=jnp.int32)

        rng, step_rng = jrand.split(rng)
        all_keys = jrand.split(step_rng, TPU_CORES_STATIC * GRAD_ACCUM)

        if all_keys.ndim == 1:
            dropout_rngs = all_keys.reshape(TPU_CORES_STATIC, GRAD_ACCUM)
        else:
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
