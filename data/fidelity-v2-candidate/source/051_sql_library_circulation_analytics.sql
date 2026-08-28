-- PostgreSQL analytical workbook for a public-library circulation system.
-- The script assumes normalized operational tables and builds reporting views
-- without changing source transactions.

BEGIN;

CREATE SCHEMA IF NOT EXISTS analytics;

-- Operational assumptions:
-- members(member_id, home_branch_id, joined_at, status)
-- items(item_id, title_id, owning_branch_id, item_type, acquired_at, retired_at)
-- titles(title_id, isbn, title, primary_language, subject_code)
-- loans(loan_id, item_id, member_id, checkout_branch_id, checkout_at,
--       due_at, returned_at, renewal_count)
-- holds(hold_id, title_id, member_id, pickup_branch_id, placed_at,
--       filled_at, cancelled_at)
-- branches(branch_id, branch_name, region_code)
-- calendar(calendar_date, fiscal_year, fiscal_month, week_start)

CREATE OR REPLACE VIEW analytics.loan_facts AS
SELECT
    l.loan_id,
    l.item_id,
    i.title_id,
    l.member_id,
    l.checkout_branch_id,
    i.owning_branch_id,
    l.checkout_at,
    l.due_at,
    l.returned_at,
    l.renewal_count,
    i.item_type,
    t.primary_language,
    t.subject_code,
    CASE
        WHEN l.returned_at IS NULL AND CURRENT_TIMESTAMP > l.due_at THEN TRUE
        WHEN l.returned_at > l.due_at THEN TRUE
        ELSE FALSE
    END AS ever_overdue,
    CASE
        WHEN l.returned_at IS NULL THEN NULL
        ELSE EXTRACT(EPOCH FROM (l.returned_at - l.checkout_at)) / 86400.0
    END AS loan_duration_days
FROM loans AS l
JOIN items AS i ON i.item_id = l.item_id
JOIN titles AS t ON t.title_id = i.title_id;

COMMENT ON VIEW analytics.loan_facts IS
'Stable loan-level reporting projection with item and title dimensions.';

-- 1. Monthly circulation by checkout branch, split by item type.
WITH monthly AS (
    SELECT
        date_trunc('month', checkout_at)::date AS month_start,
        checkout_branch_id,
        item_type,
        COUNT(*) AS loans,
        COUNT(DISTINCT member_id) AS unique_borrowers
    FROM analytics.loan_facts
    GROUP BY 1, 2, 3
)
SELECT
    m.month_start,
    b.branch_name,
    m.item_type,
    m.loans,
    m.unique_borrowers,
    ROUND(m.loans::numeric / NULLIF(m.unique_borrowers, 0), 2) AS loans_per_borrower
FROM monthly AS m
JOIN branches AS b ON b.branch_id = m.checkout_branch_id
ORDER BY m.month_start, b.branch_name, m.item_type;

-- 2. Branch-to-branch imbalance: items owned by one branch but borrowed elsewhere.
WITH movement AS (
    SELECT
        owning_branch_id,
        checkout_branch_id,
        COUNT(*) AS loan_count
    FROM analytics.loan_facts
    WHERE checkout_at >= CURRENT_DATE - INTERVAL '180 days'
    GROUP BY owning_branch_id, checkout_branch_id
),
totals AS (
    SELECT owning_branch_id, SUM(loan_count) AS all_loans
    FROM movement
    GROUP BY owning_branch_id
)
SELECT
    ob.branch_name AS owning_branch,
    cb.branch_name AS checkout_branch,
    m.loan_count,
    ROUND(100.0 * m.loan_count / NULLIF(t.all_loans, 0), 1) AS pct_of_owner_loans
FROM movement AS m
JOIN totals AS t USING (owning_branch_id)
JOIN branches AS ob ON ob.branch_id = m.owning_branch_id
JOIN branches AS cb ON cb.branch_id = m.checkout_branch_id
WHERE m.owning_branch_id <> m.checkout_branch_id
ORDER BY pct_of_owner_loans DESC, owning_branch, checkout_branch;

-- 3. Hold fulfillment latency by pickup branch.
WITH completed AS (
    SELECT
        pickup_branch_id,
        EXTRACT(EPOCH FROM (filled_at - placed_at)) / 86400.0 AS wait_days
    FROM holds
    WHERE filled_at IS NOT NULL
      AND filled_at >= CURRENT_DATE - INTERVAL '365 days'
),
ranked AS (
    SELECT
        pickup_branch_id,
        wait_days,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY wait_days)
            OVER (PARTITION BY pickup_branch_id) AS p50,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY wait_days)
            OVER (PARTITION BY pickup_branch_id) AS p90
    FROM completed
)
SELECT DISTINCT
    b.branch_name,
    ROUND(r.p50::numeric, 2) AS median_wait_days,
    ROUND(r.p90::numeric, 2) AS p90_wait_days
FROM ranked AS r
JOIN branches AS b ON b.branch_id = r.pickup_branch_id
ORDER BY p90_wait_days DESC;

-- 4. Titles with unusually high holds relative to circulating copies.
WITH copy_counts AS (
    SELECT
        title_id,
        COUNT(*) FILTER (WHERE retired_at IS NULL) AS active_copies
    FROM items
    GROUP BY title_id
),
open_holds AS (
    SELECT
        title_id,
        COUNT(*) AS hold_count
    FROM holds
    WHERE filled_at IS NULL
      AND cancelled_at IS NULL
    GROUP BY title_id
)
SELECT
    t.title_id,
    t.title,
    c.active_copies,
    COALESCE(h.hold_count, 0) AS open_holds,
    ROUND(
        COALESCE(h.hold_count, 0)::numeric / NULLIF(c.active_copies, 0),
        2
    ) AS holds_per_copy
FROM titles AS t
JOIN copy_counts AS c ON c.title_id = t.title_id
LEFT JOIN open_holds AS h ON h.title_id = t.title_id
WHERE c.active_copies > 0
  AND COALESCE(h.hold_count, 0) >= 3
ORDER BY holds_per_copy DESC, open_holds DESC
LIMIT 100;

-- 5. Cohort retention: members who joined in a month and borrowed again in
-- months 1, 3, and 6 after their join month.
WITH cohorts AS (
    SELECT
        member_id,
        date_trunc('month', joined_at)::date AS cohort_month
    FROM members
    WHERE joined_at >= CURRENT_DATE - INTERVAL '24 months'
),
activity AS (
    SELECT DISTINCT
        member_id,
        date_trunc('month', checkout_at)::date AS activity_month
    FROM loans
),
flags AS (
    SELECT
        c.member_id,
        c.cohort_month,
        BOOL_OR(a.activity_month = (c.cohort_month + INTERVAL '1 month')::date) AS m1,
        BOOL_OR(a.activity_month = (c.cohort_month + INTERVAL '3 months')::date) AS m3,
        BOOL_OR(a.activity_month = (c.cohort_month + INTERVAL '6 months')::date) AS m6
    FROM cohorts AS c
    LEFT JOIN activity AS a ON a.member_id = c.member_id
    GROUP BY c.member_id, c.cohort_month
)
SELECT
    cohort_month,
    COUNT(*) AS members,
    ROUND(100.0 * AVG(m1::int), 1) AS retained_m1_pct,
    ROUND(100.0 * AVG(m3::int), 1) AS retained_m3_pct,
    ROUND(100.0 * AVG(m6::int), 1) AS retained_m6_pct
FROM flags
GROUP BY cohort_month
ORDER BY cohort_month;

-- 6. Overdue-rate comparison must account for still-open loans.
WITH eligible AS (
    SELECT
        checkout_branch_id,
        checkout_at::date AS checkout_date,
        ever_overdue
    FROM analytics.loan_facts
    WHERE due_at < CURRENT_TIMESTAMP
      AND checkout_at >= CURRENT_DATE - INTERVAL '365 days'
)
SELECT
    b.branch_name,
    COUNT(*) AS matured_loans,
    COUNT(*) FILTER (WHERE e.ever_overdue) AS overdue_loans,
    ROUND(100.0 * AVG(e.ever_overdue::int), 2) AS overdue_pct
FROM eligible AS e
JOIN branches AS b ON b.branch_id = e.checkout_branch_id
GROUP BY b.branch_name
ORDER BY overdue_pct DESC;

-- 7. Collection-use concentration by subject using a simple Herfindahl-like
-- measure across branches. Higher value means circulation is more concentrated
-- in a small number of branches.
WITH subject_branch AS (
    SELECT
        subject_code,
        checkout_branch_id,
        COUNT(*)::numeric AS loans
    FROM analytics.loan_facts
    WHERE checkout_at >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY subject_code, checkout_branch_id
),
shares AS (
    SELECT
        subject_code,
        checkout_branch_id,
        loans,
        loans / SUM(loans) OVER (PARTITION BY subject_code) AS share
    FROM subject_branch
)
SELECT
    subject_code,
    COUNT(*) AS active_branches,
    ROUND(SUM(share * share), 4) AS circulation_concentration
FROM shares
GROUP BY subject_code
HAVING SUM(loans) >= 100
ORDER BY circulation_concentration DESC;

-- 8. Inventory candidates for review: active copies with no circulation for
-- three years, excluding recently acquired items and selected item types.
SELECT
    i.item_id,
    i.title_id,
    t.title,
    b.branch_name AS owning_branch,
    i.item_type,
    i.acquired_at,
    MAX(l.checkout_at) AS last_checkout_at
FROM items AS i
JOIN titles AS t ON t.title_id = i.title_id
JOIN branches AS b ON b.branch_id = i.owning_branch_id
LEFT JOIN loans AS l ON l.item_id = i.item_id
WHERE i.retired_at IS NULL
  AND i.acquired_at < CURRENT_DATE - INTERVAL '3 years'
  AND i.item_type NOT IN ('local_history', 'reference', 'archive')
GROUP BY i.item_id, i.title_id, t.title, b.branch_name, i.item_type, i.acquired_at
HAVING COALESCE(MAX(l.checkout_at), TIMESTAMP '1900-01-01')
       < CURRENT_TIMESTAMP - INTERVAL '3 years'
ORDER BY last_checkout_at NULLS FIRST, i.acquired_at;

COMMIT;
