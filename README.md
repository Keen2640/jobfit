# JobFit: Job-Posting Classifier and Ranker

Predicts a job posting's skill category and ranks postings by similarity to a resume, using SQL analysis, classical ML and a Dockerized FastAPI service.

## Dataset
LinkedIn Job Postings (2023-2024): https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- Raw rows: 123,849. Loaded into PostgreSQL: 123,842 postings.
- Label: the single skill category per posting (`skill_abr`). Postings with several categories, the `OTHR` class, and classes under 300 rows are dropped, leaving 40,914 labeled postings in 16 classes.

## Splits
70/15/15 stratified, seed 42: train 28,639 / val 6,137 / test 6,138.

## Results (held-out test set)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 86.22% | 0.7478 |
| XGBoost (TF-IDF + engineered features) | 87.03% | 0.7558 |

- The classes are imbalanced (health care is about 39% of the test set), so macro-F1 is lower than accuracy.
- SQL: key query went from 24.4 ms to 7.2 ms after adding `idx_postings_company` (last of 3 runs, one company).
- API: 8.2 ms median latency (p95 13.3 ms) over 100 requests, against the Docker container on a MacBook, localhost.

## How to run
1. `brew install python@3.11 postgresql@16 libomp`, start Postgres, then `createdb jobfit`
2. `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Download the dataset into `data/` (Kaggle CLI)
4. `python -m src.load_db`, then `psql jobfit -f sql/queries.sql`
5. `python -m src.baseline`, `python -m src.xgb_model`, `python -m src.make_rank`
6. `docker build -t jobfit . && docker run -p 8000:8000 jobfit`
7. `curl localhost:8000/predict -H 'content-type: application/json' -d '{"title":"Registered Nurse","description":"RN license required."}'`

## Endpoints
- `GET /health`
- `POST /predict`: posting title and description in, predicted category and top 3 out
- `POST /rank`: resume text in, top-k similar postings out

## Limitations
- Labels come from LinkedIn's own tags.
- `/rank` is plain TF-IDF cosine similarity on an 8,000-posting sample.
