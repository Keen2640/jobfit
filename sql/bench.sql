EXPLAIN (ANALYZE)
WITH mine AS (SELECT job_id FROM postings WHERE company_id = :cid),
cnt AS (
  SELECT s.skill_name, COUNT(*) AS n
  FROM mine JOIN job_skills js USING (job_id) JOIN skills s USING (skill_abr)
  GROUP BY 1
)
SELECT *, RANK() OVER (ORDER BY n DESC) FROM cnt;
