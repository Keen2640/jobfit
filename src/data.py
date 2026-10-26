import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

def load(min_count=300):
    eng = create_engine("postgresql+psycopg2://localhost/jobfit")
    df = pd.read_sql("SELECT job_id, title, description, label FROM training_data", eng)
    df = df[df.label != "OTHR"]
    vc = df.label.value_counts()
    return df[df.label.isin(vc[vc >= min_count].index)].reset_index(drop=True)

def split(df):  # 70 / 15 / 15, stratified, fixed seed
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=42)
    return tr, va, te
