import cv2
import numpy as np

# Load the user's second screenshot
screenshot_path = r"C:\Users\Thrisha\.gemini\antigravity\brain\9a15ead7-fc94-4743-b574-4942fc538186\media__1777036271742.png"
img = cv2.imread(screenshot_path)

if img is None:
    print("Could not load screenshot!")
else:
    print(f"Loaded screenshot: {img.shape}")
    
    # We want to crop to the main big image in the center.
    # The UI has a big image at the top. Let's crop roughly the top middle.
    h, w = img.shape[:2]
    # Crop center top (e.g., y from 15% to 40%, x from 30% to 70%)
    crop = img[int(h*0.15):int(h*0.4), int(w*0.3):int(w*0.7)]
    
    # Calculate color variance to see if it's rainbow or grayscale
    b, g, r = cv2.split(crop)
    print("Crop R mean:", r.mean(), "std:", r.std())
    print("Crop G mean:", g.mean(), "std:", g.std())
    print("Crop B mean:", b.mean(), "std:", b.std())
    
    # Check if R, G, B are perfectly identical (grayscale noise)
    diff_rg = np.abs(r.astype(np.float32) - g.astype(np.float32)).mean()
    diff_rb = np.abs(r.astype(np.float32) - b.astype(np.float32)).mean()
    print("Diff R-G:", diff_rg)
    print("Diff R-B:", diff_rb)
    
    # Calculate vertical vs horizontal gradients to detect vertical banding
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    print("Horizontal Gradient (X) Energy:", np.abs(gx).mean())
    print("Vertical Gradient (Y) Energy:", np.abs(gy).mean())
