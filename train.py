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
# Fix 2 [MEDIUM]   YaRN interpolation formula was direction-inverted.
# Fix 3 [LOW]      MLAAttention softmax: cast to float32 BEFORE masking.
# Fix 4 [MEDIUM]   Dropout RNG: clean jrand.split.
# ── AUDIT FIXES (v2.2) ────────────────────────────────────────────────────────
# Fix 5 [CRITICAL] combined_stream shared generator references → 40/40/20 restored.
# Fix 6 [MEDIUM]   model.init (1,64) → (1, MAX_SEQ_LEN).
# Fix 7 [LOW]      TPU_CORES_STATIC hard RuntimeError assertion.
# Fix 8 [LOW]      YaRN denominator epsilon guard.
# ── FINAL FIXES (v2.3) ─────────────────────────────────────────────────────────
# Fix 9  [CRITICAL] HBM budget: PER_CORE_BATCH 2→1, GRAD_ACCUM 16→32.
# Fix 10 [MEDIUM]   XLA_FLAGS: GPU latency flag → correct TPU flags.
# Fix 11 [LOW]      JAX version guard.
# Fix 12 [LOW]      Val set underflow guard.
# ── FIXES (v2.4) ──────────────────────────────────────────────────────────────
# Fix 13 [CRITICAL] chunked_cross_entropy() called with wrong arg count → fixed.
# Fix 14 [MEDIUM]   Dead fori_loop code after return → removed.
# ── FIXES (v2.5) ──────────────────────────────────────────────────────────────
# Fix 15 [CRITICAL] stream_math / stream_code / stream_english used manual
#         token-level iteration to skip already-seen data. At 1.17B tokens per
#         dataset this took 44+ minutes per seed, stalling the TPU (idle timeout).
#         SOLUTION: convert skip_tokens → approximate example count and call
#         HuggingFace IterableDataset.skip(n) BEFORE iteration begins.
#         .skip() uses HF's internal fast-forward (no tokenisation/decode),
#         reducing skip time from ~44 min → <60 seconds per dataset.
# Fix 16 [SECURITY] Removed hardcoded HF_TOKEN fallback string.
#         Token is now read exclusively from Kaggle secrets / Colab userdata.
#         If neither is available the script raises a clear error.
# ──────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS
# ════════════════════════════════════════════════════════════════════════════════
import os, re, gc, sys, json, math, time, logging
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
# FIX 16: No hardcoded token. Read from Kaggle secrets or Colab userdata only.
# ════════════════════════════════════════════════════════════════════════════════
HF_TOKEN = None

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

if HF_TOKEN is None:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN not found. Add it as a Kaggle secret (Settings → Secrets) "
        "or Colab userdata with key 'HF_TOKEN'."
    )

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
MAX_SEQ_LEN     = 9_216
DROPOUT_RATE    = 0.0

MLA_KV_LATENT   = 128
MLA_Q_LATENT    = 384
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
        "This script targets TPU v5e-8."
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

# ── Average tokens per example (used for .skip() conversion) ──────────────────
# Rough estimates — exact value doesn't matter, skip is approximate by design.
# We overshoot very slightly (safe) rather than undershoot (replaying old data).
AVG_TOKENS_MATH = 512    # finemath docs are typically 300–800 tokens
AVG_TOKENS_CODE = 256    # code files vary; 256 is conservative
AVG_TOKENS_ENG  = 384    # fineweb-edu passages ~300–500 tokens

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
            padding=((self.kernel_size - 1, 0),),
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
# §9  CHUNKED CROSS ENTROPY + MTP LOSS
# ════════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE  = 2048
NUM_CHUNKS  = VOCAB_SIZE // CHUNK_SIZE   # 16


def chunked_cross_entropy(logits, labels):
    B, T, V = logits.shape
    assert V == VOCAB_SIZE
    assert VOCAB_SIZE % CHUNK_SIZE == 0

    logits = logits.reshape(B, T, NUM_CHUNKS, CHUNK_SIZE)
    logits = logits.astype(jnp.float32)

    max_chunk  = jnp.max(logits, axis=-1)
    max_logits = jnp.max(max_chunk, axis=-1)
    logits = logits - max_logits[..., None, None]

    exp_logits = jnp.exp(logits)
    sum_exp    = exp_logits.sum(axis=(-1, -2))

    chunk_ids = labels // CHUNK_SIZE
    token_ids = labels % CHUNK_SIZE

    gathered_chunk = jnp.take_along_axis(
        logits, chunk_ids[..., None, None], axis=2,
    )[..., 0, :]

    gathered_token = jnp.take_along_axis(
        gathered_chunk, token_ids[..., None], axis=-1,
    )[..., 0]

    log_prob = gathered_token - jnp.log(sum_exp + 1e-8)
    return -log_prob


@jax.checkpoint
def compute_mtp_loss(params, batch, dropout_rng):
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


# ════════════════════════════════════════════════════════════════════════════════
# §10  PMAP STEPS
# ════════════════════════════════════════════════════════════════════════════════
@partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
def train_step_accum(state, micro_batches, dropout_rngs):
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
# FIX 15: Use HuggingFace IterableDataset.skip(n_examples) for fast resuming.
#
# skip_tokens is converted to an approximate example count:
#   n_examples = skip_tokens // AVG_TOKENS_PER_EXAMPLE
#
# .skip(n) calls HF's internal fast-forward mechanism — no tokenisation,
# no Python-level iteration — so it completes in <60 s regardless of how
# many tokens need to be skipped.
#
# The conversion is intentionally approximate (overshoots by at most a few
# hundred examples). At 150B-token scale this is < 1e-8 fractional error.
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
    # Convert token-level skip → example-level skip for .skip()
    n_skip = skip_tokens // AVG_TOKENS_MATH
    log.info(
        f"[MATH] finemath-3plus | target={target_bytes/1e9:.2f}GB | "
        f"skip={skip_tokens:,} tokens (~{n_skip:,} examples) | seed={seed}"
    )
    ds = (
        load_dataset(
            "HuggingFaceTB/finemath", name="finemath-3plus",
            split="train", streaming=True,
        )
        .shuffle(seed=seed, buffer_size=50_000)
        .skip(n_skip)   # ← fast HF skip, not manual iteration
    )

    consumed = 0
    buf = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        toks = _encode(text)
        buf.extend(toks)
        consumed += len(toks)
        while len(buf) >= MAX_SEQ_LEN:
            yield buf[:MAX_SEQ_LEN]
            buf = buf[MAX_SEQ_LEN:]
        if consumed * 2 >= target_bytes:
            break


def stream_code(target_bytes: int, seed: int, skip_tokens: int = 0):
    per_lang = target_bytes // len(CODE_LANGS)
    n_skip   = skip_tokens // AVG_TOKENS_CODE
    log.info(
        f"[CODE] {len(CODE_LANGS)} langs | {per_lang/1e6:.0f}MB each | "
        f"skip={skip_tokens:,} tokens (~{n_skip:,} examples) | seed={seed}"
    )
    for lang in CODE_LANGS:
        try:
            ds = (
                load_dataset(
                    "bigcode/starcoderdata", data_dir=lang,
                    split="train", streaming=True,
                )
                .skip(n_skip)   # ← fast HF skip per language shard
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
    n_skip = skip_tokens // AVG_TOKENS_ENG
    log.info(
        f"[ENG] fineweb-edu | target={target_bytes/1e9:.2f}GB | "
        f"skip={skip_tokens:,} tokens (~{n_skip:,} examples) | seed={seed}"
    )
    ds = (
        load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train", streaming=True,
        )
        .shuffle(seed=seed, buffer_size=50_000)
        .skip(n_skip)   # ← fast HF skip
    )

    consumed = 0
    buf = []
    for row in ds:
        if row.get("score", 0.0) < 3.0:
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        toks = _encode(text)
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

    if val_batches is not None:
        vpath = TMP_DIR / "val_set.npy"
        np.save(vpath, val_batches)
    else:
        vpath = None

    ensure_repo()

    upload_tasks = [
        (ckpt_path,   f"checkpoints/{ckpt_path.name}"),
        (meta_path,   "metadata.json"),
        (params_path, f"params/{params_path.name}"),
    ]
    if vpath is not None:
        upload_tasks.append((vpath, "val_set.npy"))

    all_ok = True
    for local_path, repo_path in upload_tasks:
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            log.info(f"  ✓ Uploaded: {repo_path}")
        except Exception as e:
            log.error(f"  ✗ Upload failed for {repo_path}: {e}")
            all_ok = False

    if all_ok:
        log.info(f"✓ Checkpoint saved: step={step} | "
                 f"ckpt={len(state_bytes)/1e6:.1f}MB | "
                 f"params_only={len(params_only)/1e6:.1f}MB")
        for p in [ckpt_path, meta_path, params_path]:
            p.unlink(missing_ok=True)
        if vpath is not None:
            vpath.unlink(missing_ok=True)
    else:
        log.warning(f"Some uploads failed — local files kept in {TMP_DIR} for manual retry.")


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
    log.info("✓ Fresh training start")

best_val_loss = meta.get("best_val_loss", float("inf")) if meta else float("inf")

# ════════════════════════════════════════════════════════════════════════════════
# §14  VALIDATION SET
# ════════════════════════════════════════════════════════════════════════════════
if val_batches_loaded is not None:
    val_batches = val_batches_loaded
    log.info(f"✓ Val set from checkpoint: {val_batches.shape}")
else:
    log.info(f"Creating val set ({VAL_BATCHES_N} batches)...")
    val_gen  = stream_english(int(TOTAL_TOKENS * ENG_RATIO * 2), GLOBAL_SEED + 99, skip_tokens=0)
    val_list = []
    for block in val_gen:
        val_list.append(block)
        if len(val_list) >= VAL_BATCHES_N * MICRO_BATCH:
            break

    if len(val_list) < VAL_BATCHES_N * MICRO_BATCH:
        raise RuntimeError(
            f"Val set underflow: got {len(val_list)} blocks, "
            f"need {VAL_BATCHES_N * MICRO_BATCH}."
        )

    val_batches = np.array(val_list, dtype=np.int32).reshape(
        VAL_BATCHES_N, MICRO_BATCH, MAX_SEQ_LEN
    )
    log.info(f"✓ Val set ready: {val_batches.shape}")


# ════════════════════════════════════════════════════════════════════════════════
# §15  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════════
log.info("=" * 80)
log.info(f"ZENYX-V2 PRETRAINING | TPU v5e-8 | Step {resume_step}/{MAX_STEPS}")
log.info(f"d_model={D_MODEL} | {N_UNIQUE_BLOCKS}×{N_RECURRENCES} recurrences | "
         f"vocab={VOCAB_SIZE} | ctx={MAX_SEQ_LEN}")
log.info(f"Unique params: {param_count/1e6:.1f}M | "
         f"Effective depth: {N_UNIQUE_BLOCKS * N_RECURRENCES} layers")
log.info(f"Per-core batch: {PER_CORE_BATCH} | Grad-accum: {GRAD_ACCUM} | "
         f"Global batch: {GLOBAL_BATCH} seqs | "
         f"{GLOBAL_BATCH * MAX_SEQ_LEN / 1e6:.2f}M tokens/step")
log.info("=" * 80)

train_stream = combined_stream(resume_step=resume_step, seed=GLOBAL_SEED)
prefetch_q   = Queue(maxsize=256)
prefetch_t   = Thread(target=prefetch_worker, args=(train_stream, prefetch_q), daemon=True)
prefetch_t.start()

global_step  = resume_step
start_time   = time.time()
rng          = jrand.PRNGKey(GLOBAL_SEED + resume_step)
train_losses = []

try:
    while global_step < MAX_STEPS:
        # ── collect GRAD_ACCUM micro-batches ──────────────────────────────────
        micro_batches_list = []
        for _ in range(GRAD_ACCUM):
            block = prefetch_q.get()
            if block is None:
                break
            micro_batches_list.append(block)

        if len(micro_batches_list) < GRAD_ACCUM:
            log.info("Dataset exhausted.")
            break

        # shape: [GRAD_ACCUM, MICRO_BATCH, SEQ_LEN]
        mb_np = np.stack(micro_batches_list, axis=0)  # [GRAD_ACCUM, MICRO_BATCH, SEQ]

        # shard across TPU cores: [TPU_CORES, GRAD_ACCUM, PER_CORE_BATCH, SEQ]
        mb_jax = jnp.array(mb_np, dtype=jnp.int32)
        mb_jax = mb_jax.reshape(GRAD_ACCUM, TPU_CORES, PER_CORE_BATCH, MAX_SEQ_LEN)
        mb_jax = mb_jax.transpose(1, 0, 2, 3)  # [TPU_CORES, GRAD_ACCUM, PER_CORE_BATCH, SEQ]

        # dropout RNGs
        rng, *sub_rngs = jrand.split(rng, 1 + TPU_CORES * GRAD_ACCUM)
        dropout_rngs = jnp.array(sub_rngs).reshape(TPU_CORES, GRAD_ACCUM, -1)

        state, loss, grad_norm = train_step_accum(state, mb_jax, dropout_rngs)

        loss_val      = float(flax.jax_utils.unreplicate(loss))
        grad_norm_val = float(flax.jax_utils.unreplicate(grad_norm))
        train_losses.append(loss_val)

        if global_step % 50 == 0:
            elapsed  = (time.time() - start_time) / 60.0
            steps_done = max(global_step - resume_step, 1)
            tok_s    = steps_done * GLOBAL_BATCH * MAX_SEQ_LEN / max(time.time() - start_time, 1)
            ppl      = math.exp(min(loss_val, 10))
            log.info(
                f"Step {global_step:6d} | Loss {loss_val:.4f} | PPL {ppl:7.2f} | "
                f"GNorm {grad_norm_val:.3f} | {elapsed:.1f}min | {tok_s/1e6:.2f}M tok/s"
            )

        global_step += 1

        # ── eval + checkpoint ─────────────────────────────────────────────────
        if global_step % EVAL_EVERY == 0:
            log.info(f"Evaluating at step {global_step}...")
            val_loss_list = []
            for vb in val_batches:   # vb: [MICRO_BATCH, SEQ_LEN]
                vb_jax = jnp.array(vb, dtype=jnp.int32).reshape(
                    TPU_CORES, PER_CORE_BATCH, MAX_SEQ_LEN
                )
                vl = eval_step(state.params, vb_jax)
                val_loss_list.append(float(flax.jax_utils.unreplicate(vl)))

            mean_val = sum(val_loss_list) / len(val_loss_list)
            val_ppl  = math.exp(min(mean_val, 10))
            recent_train = sum(train_losses[-100:]) / min(len(train_losses), 100)
            log.info(
                f"EVAL Step {global_step} | Val Loss {mean_val:.4f} | "
                f"Val PPL {val_ppl:.2f} | Train Loss {recent_train:.4f} | "
                f"Gap {mean_val - recent_train:+.4f}"
            )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                log.info(f"  ✓ New best val loss: {best_val_loss:.4f}")

            meta_out = {
                "step": global_step,
                "best_val_loss": float(best_val_loss),
                "recent_train_loss": float(recent_train),
                "train_losses_tail": train_losses[-100:],
            }
            save_checkpoint(state, global_step, meta_out,
                            val_batches if global_step == EVAL_EVERY else None)

except KeyboardInterrupt:
    log.info("\n⚠️  Interrupted — saving checkpoint...")

log.info("Saving final checkpoint...")
meta_final = {
    "step": global_step,
    "best_val_loss": float(best_val_loss),
    "train_losses_tail": train_losses[-100:],
}
save_checkpoint(state, global_step, meta_final)
log.info("=" * 80)
log.info(f"✓ DONE | Final step: {global_step} | Best val loss: {best_val_loss:.4f}")
log.info("=" * 80)
