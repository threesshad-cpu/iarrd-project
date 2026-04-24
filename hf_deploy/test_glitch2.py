from PIL import Image
import numpy as np

img = Image.open("../downloaded_moon.jpg")
arr = np.asarray(img, dtype=np.float32)

# Calculate adjacent pixel differences (horizontal and vertical)
diff_x = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
diff_y = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()

print(f"Mean pixel diff X: {diff_x:.2f}")
print(f"Mean pixel diff Y: {diff_y:.2f}")

# Check variance per channel
for c in range(3):
    print(f"Ch {c} mean: {arr[:,:,c].mean():.2f}, std: {arr[:,:,c].std():.2f}")
