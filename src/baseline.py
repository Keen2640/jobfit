from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data import load, split
from src.features import text_of

df = load(); tr, va, te = split(df)
print("total/train/val/test:", len(df), len(tr), len(va), len(te))
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3,
                      sublinear_tf=True, stop_words="english")
Xtr = vec.fit_transform(text_of(tr))
clf = LogisticRegression(max_iter=1000, C=5, class_weight="balanced").fit(Xtr, tr.label)

for name, d in [("val", va), ("TEST", te)]:
    p = clf.predict(vec.transform(text_of(d)))
    print(name, "acc=%.4f macroF1=%.4f" % (accuracy_score(d.label, p), f1_score(d.label, p, average="macro")))
print(classification_report(te.label, p))
