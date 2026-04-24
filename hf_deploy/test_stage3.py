import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def _clahe_stretch(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    out = arr.copy()
    for ch in range(arr.shape[2]):
        plane  = arr[:, :, ch]
        p_lo   = np.percentile(plane, lo)
        p_hi   = np.percentile(plane, hi)
        span   = max(p_hi - p_lo, 1e-6)
        out[:, :, ch] = np.clip((plane - p_lo) / span, 0.0, 1.0) * 255.0
    return out

def stage_03_enhancement(raw_bytes: bytes, native_w: int, native_h: int):
    CAP_SIZE = 512
    ENHANCE_SIZE = (1024, 1024)
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((CAP_SIZE, CAP_SIZE), Image.LANCZOS)

    # Pass 1 — CLAHE stretch
    arr = np.asarray(img, dtype=np.float32)
    arr = _clahe_stretch(arr, lo=1.0, hi=99.0)
    img = Image.fromarray(arr.astype(np.uint8))

    # Pass 2 — Detail layer boost
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    i_arr   = np.asarray(img, dtype=np.float32)
    b_arr   = np.asarray(blurred, dtype=np.float32)
    fused   = np.clip(i_arr + 0.60 * (i_arr - b_arr), 0.0, 255.0).astype(np.uint8)
    img     = Image.fromarray(fused)

    # Pass 3/4/5 — Triple unsharp masking
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2,  percent=220, threshold=1))
    img = img.filter(ImageFilter.UnsharpMask(radius=4.0,  percent=160, threshold=2))
    img = img.filter(ImageFilter.UnsharpMask(radius=10.0, percent=80,  threshold=3))

    # Pass 6 — Gamma lift + secondary CLAHE
    lut  = [int(255 * (i / 255.0) ** 0.80) for i in range(256)] * 3
    img  = img.point(lut)
    arr2 = np.asarray(img, dtype=np.float32)
    arr2 = _clahe_stretch(arr2, lo=0.5, hi=99.5)
    img  = Image.fromarray(arr2.astype(np.uint8))

    # Pass 7 — Color/contrast/brightness
    img = ImageEnhance.Color(img).enhance(1.35)
    img = ImageEnhance.Contrast(img).enhance(1.20)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # Pass 8 — Edge-preserving denoise
    smooth    = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    edges     = img.filter(ImageFilter.FIND_EDGES)
    e_arr     = np.asarray(edges,  dtype=np.float32).mean(axis=2, keepdims=True) / 255.0
    img_arr   = np.asarray(img,    dtype=np.float32)
    s_arr     = np.asarray(smooth, dtype=np.float32)
    mask      = np.clip(e_arr * 4.0, 0.0, 1.0)
    denoised  = img_arr * mask + s_arr * (1.0 - mask)
    img       = Image.fromarray(np.clip(denoised, 0, 255).astype(np.uint8))

    # Dual deconvolution sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=200, threshold=0))
    img = img.filter(ImageFilter.UnsharpMask(radius=2.5, percent=120, threshold=1))

    # Final — LANCZOS
    img = img.resize(ENHANCE_SIZE, Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=1))
    
    img.save("test_moon_out.jpg", quality=88)

import cv2

# Create synthetic moon image (mostly black background)
moon = np.zeros((200, 200, 3), dtype=np.float32)
cv2.circle(moon, (100, 100), 50, (255, 255, 255), -1)

# Add tiny compression noise to the black background (values 0-2)
noise = np.random.randint(0, 3, (200, 200, 3)).astype(np.float32)
# Apply noise only to background
bg_mask = (moon[:, :, 0] == 0)
moon[bg_mask] += noise[bg_mask]

moon = np.clip(moon, 0, 255).astype(np.uint8)

buf = io.BytesIO()
Image.fromarray(moon).save(buf, format="JPEG")
raw = buf.getvalue()

stage_03_enhancement(raw, 200, 200)
print("Saved test_moon_out.jpg")
