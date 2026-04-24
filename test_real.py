import requests

# Download a moon image from NASA APOD or wikimedia
url = "https://upload.wikimedia.org/wikipedia/commons/e/e1/FullMoon2010.jpg"
r = requests.get(url)
with open("test_moon.jpg", "wb") as f:
    f.write(r.content)

# Send to backend
try:
    resp = requests.post("http://localhost:8000/api/analyze", files={"file": ("test_moon.jpg", open("test_moon.jpg", "rb"), "image/jpeg")})
    print(f"Status: {resp.status_code}")
    print(resp.text[:500])
except Exception as e:
    print(e)
