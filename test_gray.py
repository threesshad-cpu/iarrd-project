import requests
import io
from PIL import Image

# Grayscale image
img = Image.new("L", (319, 180), color=100)
buf = io.BytesIO()
img.save(buf, format="JPEG")
buf.seek(0)

try:
    resp = requests.post("http://localhost:8000/api/analyze", files={"file": ("moon.jpg", buf, "image/jpeg")})
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(e)
