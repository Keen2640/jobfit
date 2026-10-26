import requests, time, statistics
url = "http://localhost:8000/predict"
body = {"title": "Registered Nurse",
        "description": "Provide patient care in a hospital. RN license and 2 years experience required. " * 10}
for _ in range(10): requests.post(url, json=body)          # warm-up
ts = []
for _ in range(100):
    t = time.perf_counter(); requests.post(url, json=body).raise_for_status()
    ts.append((time.perf_counter() - t) * 1000)
print("median %.1f ms | p95 %.1f ms" % (statistics.median(ts), sorted(ts)[94]))
