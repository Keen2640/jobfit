import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import linear_kernel
from src.features import build_X

app = FastAPI(title="JobFit")
art = joblib.load("artifacts/xgb.joblib")
rk = joblib.load("artifacts/rank.joblib")

class Posting(BaseModel):
    title: str
    description: str

class RankReq(BaseModel):
    resume: str
    top_k: int = 5

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(p: Posting):
    X = build_X(art["vec"], pd.DataFrame([p.model_dump()]))
    proba = art["clf"].predict_proba(X)[0]
    top = proba.argsort()[::-1][:3]
    labels = art["le"].inverse_transform(top)
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
