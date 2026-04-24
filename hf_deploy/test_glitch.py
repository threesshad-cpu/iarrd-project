import io
import base64
import numpy as np
from PIL import Image, ImageFilter

def test_glitch():
    # create a dummy image (RGB)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :50, 0] = 255  # red left half
    img = Image.fromarray(arr)
    
    # Pass 8 logic
    smooth    = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    edges     = img.filter(ImageFilter.FIND_EDGES)
    e_arr     = np.asarray(edges,  dtype=np.float32).mean(axis=2, keepdims=True) / 255.0
    img_arr   = np.asarray(img,    dtype=np.float32)
    s_arr     = np.asarray(smooth, dtype=np.float32)
    mask      = np.clip(e_arr * 4.0, 0.0, 1.0)
    denoised  = img_arr * mask + s_arr * (1.0 - mask)
    
    img_out = Image.fromarray(np.clip(denoised, 0, 255).astype(np.uint8))
    img_out.save("test_glitch_out.png")
    print("Pass 8 done")

test_glitch()
