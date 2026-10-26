import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import load
from src.features import text_of

df = load()
df = df.sample(min(8000, len(df)), random_state=0).reset_index(drop=True)
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3,
                      sublinear_tf=True, stop_words="english")
X = vec.fit_transform(text_of(df))
joblib.dump({"vec": vec, "X": X, "meta": df[["job_id", "title", "label"]]},
            "artifacts/rank.joblib", compress=3)
