-- Monthly acquisition-cohort retention (share of the cohort purchasing in month k).
-- Note: the extract starts 2010-12-01, so the 2010-12 "cohort" also contains every
-- pre-existing customer (left-censoring) and is not a true acquisition cohort.
WITH purchase_months AS (
    SELECT DISTINCT CustomerID, date_trunc('month', InvoiceDate) AS order_month
    FROM clean_transactions
    WHERE NOT IsCancel
),
cohorts AS (
    SELECT CustomerID, MIN(order_month) AS cohort_month
    FROM purchase_months
    GROUP BY CustomerID
),
activity AS (
    SELECT
        c.cohort_month,
        date_diff('month', c.cohort_month, pm.order_month) AS period,
        COUNT(DISTINCT pm.CustomerID)                      AS active_customers
    FROM purchase_months pm
    JOIN cohorts c USING (CustomerID)
    GROUP BY 1, 2
)
SELECT
    cohort_month,
    period,
    active_customers,
    active_customers / FIRST_VALUE(active_customers) OVER (
        PARTITION BY cohort_month ORDER BY period
    ) AS retention
FROM activity
ORDER BY cohort_month, period;
