import re
import pandas as pd
import scipy.sparse as sp

KW = ["python","sql","java","excel","aws","salesforce","nurse","degree","license",
      "certification","remote","bachelor","clearance","cpa","driver","manager","sales"]

def text_of(df):
    return (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()

def engineered(df):
    t, d = df["title"].fillna(""), df["description"].fillna("")
    out = pd.DataFrame({
        "desc_len": d.str.len(), "desc_words": d.str.split().str.len(),
        "title_len": t.str.len(), "title_words": t.str.split().str.len(),
        "n_lines": d.str.count("\n"), "n_digits": d.str.count(r"\d"),
        "has_salary": d.str.contains(r"\$\s?\d", regex=True).astype(int),
        "upper_ratio": d.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
    })
    low = text_of(df)
    for k in KW:
        out["kw_" + k] = low.str.count(r"\b" + re.escape(k) + r"\b")
    return out.astype(float)

def build_X(vec, df, fit=False):
    tf = vec.fit_transform(text_of(df)) if fit else vec.transform(text_of(df))
    return sp.hstack([tf, sp.csr_matrix(engineered(df).values)]).tocsr()
