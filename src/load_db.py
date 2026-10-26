import pandas as pd
from sqlalchemy import create_engine, text

eng = create_engine("postgresql+psycopg2://localhost/jobfit")
D = "data/"
post = pd.read_csv(D + "postings.csv")
print("RAW ROWS:", len(post))

keep = ["job_id","company_id","title","description","location","work_type",
        "formatted_experience_level","min_salary","max_salary","med_salary","views"]
postings = (post[[c for c in keep if c in post.columns]]
            .dropna(subset=["title","description"]).drop_duplicates("job_id"))
postings["company_id"] = postings["company_id"].astype("Int64")

companies = pd.read_csv(D + "companies/companies.csv")[["company_id","name"]].drop_duplicates("company_id")
skills = pd.read_csv(D + "mappings/skills.csv")
job_skills = pd.read_csv(D + "jobs/job_skills.csv").drop_duplicates()
job_skills = job_skills[job_skills.job_id.isin(postings.job_id)]

for name, df in [("postings",postings),("companies",companies),("skills",skills),("job_skills",job_skills)]:
    df.to_sql(name, eng, if_exists="replace", index=False, chunksize=5000)

with eng.begin() as c:
    c.execute(text("ALTER TABLE postings ADD PRIMARY KEY (job_id)"))
    c.execute(text("ALTER TABLE companies ADD PRIMARY KEY (company_id)"))
    c.execute(text("ALTER TABLE skills ADD PRIMARY KEY (skill_abr)"))
    c.execute(text("""CREATE OR REPLACE VIEW training_data AS
        SELECT p.job_id, p.title, p.description, js.skill_abr AS label
        FROM postings p JOIN job_skills js USING (job_id)
        WHERE p.job_id IN (SELECT job_id FROM job_skills GROUP BY job_id HAVING COUNT(*) = 1)"""))
print("postings loaded:", len(postings))
