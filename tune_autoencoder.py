"""
Autoencoder Weight Tuning Script
=================================
Modifies the final decoder Conv2D layer weights of best_autoencoder1.h5
to produce cleaner, less noisy reconstruction output.

Strategy:
  1. Load the existing autoencoder weights via keras
  2. Get the last Conv2D layer (conv2d_23, sigmoid output -> [0,1])
  3. Blend the existing kernel toward a Gaussian-smoothing 3x3 kernel
     (reduces high-freq noise amplification while preserving structure)
  4. Slightly increase the bias to counter the sigmoid compression that
     causes dark/murky reconstruction
  5. Resave the h5 weights in-place

Run from the project root:
  cd c:\\Users\\Thrisha\\Downloads\\astronomical-ui
  backend\\venv\\Scripts\\python.exe tune_autoencoder.py
"""

import os, sys, time
import numpy as np

# ── 0. Setup paths ────────────────────────────────────────────────────────────
H5_PATH   = "best_autoencoder1.h5"
SAVE_PATH = "best_autoencoder1.h5"   # overwrite in-place

print(f"[*] Loading model from: {H5_PATH}")
t0 = time.time()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="builtins", name="combined_loss")
def combined_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@keras.saving.register_keras_serializable(package="builtins", name="psnr")
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

@keras.saving.register_keras_serializable(package="builtins", name="ssim")
def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

model = keras.saving.load_model(H5_PATH, compile=False)
print(f"[+] Loaded in {time.time()-t0:.1f}s — layers: {len(model.layers)}")

# ── 1. Identify target layer (last conv2d - sigmoid output) ───────────────────
target_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, keras.layers.Conv2D):
        target_layer = layer
        print(f"[+] Target layer: {layer.name}  filters={layer.filters}  "
              f"kernel={layer.kernel_size}  activation={layer.activation.__name__}")
        break

if target_layer is None:
    print("[-] No Conv2D found — aborting")
    sys.exit(1)

# ── 2. Get current weights ────────────────────────────────────────────────────
weights = target_layer.get_weights()
print(f"[+] Weight tensors: {[w.shape for w in weights]}")

kernel = weights[0].copy()   # shape: [3, 3, 32, 3]
bias   = weights[1].copy() if len(weights) > 1 else None

print(f"    kernel stats — min={kernel.min():.4f}  max={kernel.max():.4f}  "
      f"mean={kernel.mean():.4f}  std={kernel.std():.4f}")
if bias is not None:
    print(f"    bias   stats — min={bias.min():.4f}  max={bias.max():.4f}  "
          f"mean={bias.mean():.4f}")

# ── 3. Build a Gaussian-smoothing prototype kernel ────────────────────────────
# 3×3 Gaussian kernel (σ~0.85) — promotes smooth output, kills ringing
gauss_3x3 = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125 ],
    [0.0625, 0.125, 0.0625],
], dtype=np.float32)

# Build a full [3,3,in_ch,out_ch] kernel that applies the Gaussian per-channel
# For each output channel, use the same smooth kernel across input channels
in_ch  = kernel.shape[2]   # 32
out_ch = kernel.shape[3]   # 3
gauss_kernel = np.zeros_like(kernel)
for o in range(out_ch):
    for i in range(in_ch):
        gauss_kernel[:, :, i, o] = gauss_3x3 / in_ch  # normalised contribution

# ── 4. Blend: 20% toward Gaussian (smooth) + 80% original (structure) ────────
BLEND = 0.20
kernel_tuned = (1.0 - BLEND) * kernel + BLEND * gauss_kernel

# ── 5. Lift bias to brighten dark reconstructions ─────────────────────────────
# The sigmoid compresses outputs; a small positive offset brightens them
BIAS_LIFT = 0.04
if bias is not None:
    bias_tuned = bias + BIAS_LIFT
    print(f"[+] Bias after lift — mean={bias_tuned.mean():.4f}")
else:
    bias_tuned = None

print(f"[+] Kernel after blend — min={kernel_tuned.min():.4f}  "
      f"max={kernel_tuned.max():.4f}  std={kernel_tuned.std():.4f}")

# ── 6. Write tuned weights back ───────────────────────────────────────────────
new_weights = [kernel_tuned]
if bias_tuned is not None:
    new_weights.append(bias_tuned)
target_layer.set_weights(new_weights)

# ── 7. Quick sanity: forward pass on a noise-free test image ─────────────────
test_in  = np.random.uniform(0.3, 0.7, (1, 128, 128, 3)).astype(np.float32)
test_out = model(test_in, training=False).numpy()
print(f"[+] Sanity forward pass — output range [{test_out.min():.3f}, {test_out.max():.3f}]  "
      f"mean={test_out.mean():.3f}  std={test_out.std():.4f}")

if test_out.std() < 0.001:
    print("[!] WARNING: output collapsed — aborting save, model may be broken")
    sys.exit(1)

# ── 8. Save ───────────────────────────────────────────────────────────────────
print(f"[*] Saving tuned model -> {SAVE_PATH}")
model.save(SAVE_PATH)
print(f"[+] Done in {time.time()-t0:.1f}s total")
print("[+] Restart the FastAPI backend to pick up the new weights.")
