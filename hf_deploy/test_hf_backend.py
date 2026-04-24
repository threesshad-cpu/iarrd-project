import requests
import io
from PIL import Image

img = Image.new('RGB', (200, 200), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)

try:
    r = requests.post(
        "http://127.0.0.1:7864/api/analyze", 
        files={"file": ("dummy.jpg", buf, "image/jpeg")}
    )
    print("Status:", r.status_code)
    print("Response snippet:", r.text[:500])
except Exception as e:
    print("Error:", e)
