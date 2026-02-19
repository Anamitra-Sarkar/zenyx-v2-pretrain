#!/bin/bash
# ╔══════════════════════════════════════════════════════════╗
# ║  ZENYX-V2 INSTALL — TPU v5e-8 Kaggle Setup             ║
# ╚══════════════════════════════════════════════════════════╝

set -e

echo "═══════════════════════════════════════════"
echo "  ZENYX-V2 TPU v5e-8 Environment Setup"
echo "═══════════════════════════════════════════"

# Step 1: TPU-compatible JAX (MUST be before anything else)
echo "[1/5] Installing JAX with TPU support..."
pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  -q

# Step 2: Flax + Optax
echo "[2/5] Installing Flax + Optax..."
pip install -U flax optax -q

# Step 3: HuggingFace stack
echo "[3/5] Installing HuggingFace stack..."
pip install -U \
  transformers \
  datasets \
  huggingface_hub \
  tokenizers \
  -q

# Step 4: Msgpack
echo "[4/5] Installing msgpack..."
pip install -U msgpack -q

# Step 5: Verify TPU
echo "[5/5] Verifying TPU setup..."
python -c "
import jax
print('JAX:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
assert jax.device_count() == 8, f'Need 8 TPU cores! Got {jax.device_count()}'
print('✓ TPU v5e-8 ready — all 8 cores detected')
"

echo ""
echo "✓ All dependencies installed. Run: python train.py"
echo "═══════════════════════════════════════════"
