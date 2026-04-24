import cv2
import numpy as np

# Create an image that has varying values
arr = np.random.randint(0, 255, (100, 100, 3)).astype(np.float32) / 255.0

c = 0
plane = arr[:, :, c]

try:
    blur = cv2.GaussianBlur(plane, (5, 5), 1.5)
    print("Shape:", blur.shape)
    print("Contiguous:", blur.flags['C_CONTIGUOUS'])
except Exception as e:
    print("Exception:", e)
