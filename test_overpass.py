import time
import requests
import roads

q = '[out:json][timeout:25];way["highway"](10.985,76.425,10.990,76.435);out ids 5;'
for url in roads.OVERPASS_URLS:
    host = url.split("/")[2]
    t = time.time()
    try:
        r = requests.post(url, data={"data": q}, headers=roads.HEADERS, timeout=30)
        print(f"{host}: HTTP {r.status_code} in {time.time() - t:.1f}s  {r.text[:60]!r}")
    except Exception as e:
        print(f"{host}: FAILED after {time.time() - t:.1f}s - {type(e).__name__}")
