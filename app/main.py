import time, logging, statistics
from collections import Counter, deque
import joblib, pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.metrics.pairwise import linear_kernel
from src.features import build_X

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("jobfit")

app = FastAPI(title="JobFit")
art = joblib.load("artifacts/xgb.joblib")
rk = joblib.load("artifacts/rank.joblib")

STATS = {"requests": 0, "errors": 0}
LATENCY = deque(maxlen=1000)      # last 1000 request latencies (ms)
CONFIDENCE = deque(maxlen=1000)   # last 1000 top-1 probabilities
LABELS = Counter()                # predicted label mix

class Posting(BaseModel):
    title: str
    description: str

class RankReq(BaseModel):
    resume: str
    top_k: int = 5

@app.middleware("http")
async def track(request: Request, call_next):
    t = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        log.exception("unhandled error on %s", request.url.path)
        raise
    ms = (time.perf_counter() - t) * 1000
    if request.url.path in ("/predict", "/rank"):
        STATS["requests"] += 1
        LATENCY.append(ms)
        if resp.status_code >= 400:
            STATS["errors"] += 1
    log.info("%s %s %s %.1fms", request.method, request.url.path, resp.status_code, ms)
    return resp

@app.get("/health")
def health(): return {"ok": True}

@app.get("/stats")
def stats():
    lat = sorted(LATENCY)
    return {
        "requests": STATS["requests"],
        "errors": STATS["errors"],
        "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
        "mean_confidence": round(sum(CONFIDENCE) / len(CONFIDENCE), 4) if CONFIDENCE else None,
        "label_counts": dict(LABELS),
    }

@app.post("/predict")
def predict(p: Posting):
    X = build_X(art["vec"], pd.DataFrame([p.model_dump()]))
    proba = art["clf"].predict_proba(X)[0]
    top = proba.argsort()[::-1][:3]
    labels = art["le"].inverse_transform(top)
    LABELS[labels[0]] += 1
    CONFIDENCE.append(float(proba[top[0]]))
    return {"label": labels[0],
            "top3": [{"label": l, "prob": round(float(proba[i]), 4)} for l, i in zip(labels, top)]}

@app.post("/rank")
def rank(r: RankReq):
    q = rk["vec"].transform([r.resume.lower()])
    sims = linear_kernel(q, rk["X"]).ravel()
    idx = sims.argsort()[::-1][:r.top_k]
    m = rk["meta"].iloc[idx]
    return [{"job_id": int(a), "title": b, "category": c, "score": round(float(s), 4)}
            for a, b, c, s in zip(m.job_id, m.title, m.label, sims[idx])]
