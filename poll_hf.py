import urllib.request, json, time

print("Polling HF Space for rebuild completion...")
for i in range(16):
    time.sleep(15)
    elapsed = (i + 1) * 15
    try:
        r = urllib.request.urlopen("https://threessha-iarrd-backend.hf.space/health", timeout=10)
        d = json.loads(r.read())
        status = d.get("status", "?")
        models = d.get("models_loaded", [])
        version = d.get("version", "?")
        print(f"  [{elapsed}s] status={status}  models={models}  version={version}")
        if status == "ready" and models:
            print("SUCCESS: HF Space is LIVE and READY with fixed code!")
            break
    except Exception as e:
        print(f"  [{elapsed}s] not ready yet ({e})")
else:
    print("Timeout — space may still be building. Check https://huggingface.co/spaces/threessha/iarrd-backend")
