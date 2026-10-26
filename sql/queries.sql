-- Q1: top 5 skill categories per experience level (CTE + RANK)
WITH counts AS (
  SELECT p.formatted_experience_level AS lvl, s.skill_name, COUNT(*) AS n
  FROM postings p
  JOIN job_skills js ON js.job_id = p.job_id
  JOIN skills s ON s.skill_abr = js.skill_abr
  WHERE p.formatted_experience_level IS NOT NULL
  GROUP BY 1, 2
), ranked AS (
  SELECT *, RANK() OVER (PARTITION BY lvl ORDER BY n DESC) AS rnk FROM counts
)
SELECT * FROM ranked WHERE rnk <= 5 ORDER BY lvl, rnk;

-- Q2: top companies with cumulative share of all postings (running SUM window)
WITH c AS (
  SELECT co.name, COUNT(*) AS n
  FROM postings p JOIN companies co USING (company_id) GROUP BY co.name
)
SELECT name, n,
       ROUND(100.0 * SUM(n) OVER (ORDER BY n DESC, name) / SUM(n) OVER (), 2) AS cumulative_pct
FROM c ORDER BY n DESC LIMIT 20;

-- Q3: top 3 highest-paying postings per skill category (ROW_NUMBER)
WITH paid AS (
  SELECT s.skill_name, p.title, p.max_salary,
         ROW_NUMBER() OVER (PARTITION BY s.skill_name ORDER BY p.max_salary DESC) AS rn
  FROM job_skills js
  JOIN postings p ON p.job_id = js.job_id
  JOIN skills s ON s.skill_abr = js.skill_abr
  WHERE p.max_salary BETWEEN 10000 AND 1000000
)
SELECT * FROM paid WHERE rn <= 3 ORDER BY skill_name, rn;

-- Q4: salary percentile within each skill category (PERCENT_RANK + AVG window)
WITH base AS (
  SELECT js.skill_abr, p.job_id, p.title, p.max_salary
  FROM job_skills js JOIN postings p USING (job_id)
  WHERE p.max_salary BETWEEN 10000 AND 1000000
)
SELECT skill_abr, title, max_salary,
       ROUND(AVG(max_salary) OVER (PARTITION BY skill_abr)) AS category_avg,
       ROUND(PERCENT_RANK() OVER (PARTITION BY skill_abr ORDER BY max_salary)::numeric, 3) AS pct_rank
FROM base ORDER BY pct_rank DESC LIMIT 25;

-- Q5: work-type mix within each experience level (partitioned SUM)
WITH t AS (
  SELECT formatted_experience_level AS lvl, work_type, COUNT(*) AS n
  FROM postings WHERE formatted_experience_level IS NOT NULL GROUP BY 1, 2
)
SELECT lvl, work_type, n,
       ROUND(100.0 * n / SUM(n) OVER (PARTITION BY lvl), 1) AS pct
FROM t ORDER BY lvl, n DESC;
