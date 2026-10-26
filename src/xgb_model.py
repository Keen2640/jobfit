import joblib, xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from src.data import load, split
from src.features import build_X

df = load(); tr, va, te = split(df)
print("total/train/val/test:", len(df), len(tr), len(va), len(te))
vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5,
                      sublinear_tf=True, stop_words="english")
le = LabelEncoder().fit(df.label)
Xtr, Xva, Xte = build_X(vec, tr, fit=True), build_X(vec, va), build_X(vec, te)
ytr, yva, yte = le.transform(tr.label), le.transform(va.label), le.transform(te.label)

clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, subsample=0.8,
                        colsample_bytree=0.5, tree_method="hist", eval_metric="mlogloss",
                        early_stopping_rounds=20, n_jobs=-1)
clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr),
        eval_set=[(Xva, yva)], verbose=50)

for name, X, y in [("val", Xva, yva), ("TEST", Xte, yte)]:
    p = clf.predict(X)
    print(name, "acc=%.4f macroF1=%.4f" % (accuracy_score(y, p), f1_score(y, p, average="macro")))
joblib.dump({"vec": vec, "le": le, "clf": clf}, "artifacts/xgb.joblib", compress=3)
