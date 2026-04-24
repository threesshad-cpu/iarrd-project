from PIL import Image
import numpy as np

img = Image.open("test_moon_out.jpg")
arr = np.asarray(img)
# The image should be mostly a moon in the center and dark background.
# If background noise got blown up, the corners will be bright/noisy.
corner = arr[:20, :20]
print(f"Corner mean: {corner.mean()}")
print(f"Corner max: {corner.max()}")
print(f"Corner min: {corner.min()}")
print(f"Corner std: {corner.std()}")
