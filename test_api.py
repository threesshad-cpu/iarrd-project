import requests

try:
    with open("test_moon.jpg", "rb") as f:
        r = requests.post("http://localhost:8000/api/analyze", files={"file": f})
    print("Status:", r.status_code)
    print("Text:", r.text)
except Exception as e:
    print("Exception:", str(e))
