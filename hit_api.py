import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io

url = "https://threessha-iarrd-backend.hf.space/api/analyze"

moon = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(moon, (100, 100), 50, (200, 200, 200), -1)
moon = cv2.GaussianBlur(moon, (5, 5), 0)
noise = np.random.randint(0, 10, (200, 200, 3), dtype=np.uint8)
moon = cv2.add(moon, noise)

buf = io.BytesIO()
Image.fromarray(moon).save(buf, format="JPEG")
files = {"file": ("moon.jpg", buf.getvalue(), "image/jpeg")}

print("Sending request...")
res = requests.post(url, files=files)

print("Status:", res.status_code)
if res.status_code == 200:
    data = res.json()
    b64_img = data["enhancement"]["image_b64"]
    with open("downloaded_moon.jpg", "wb") as f:
        f.write(base64.b64decode(b64_img))
    print("Saved downloaded_moon.jpg")
else:
    print(res.text)
