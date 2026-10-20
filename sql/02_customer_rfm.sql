-- Customer-level RFM features, net of returns.
-- Parameter: $snapshot (DATE) - the day after the last invoice in the extract.
WITH purchases AS (
    SELECT
        CustomerID,
        MIN(InvoiceDate)            AS first_purchase,
        MAX(InvoiceDate)            AS last_purchase,
        COUNT(DISTINCT InvoiceNo)   AS frequency,
        SUM(Revenue)                AS gross_revenue
    FROM clean_transactions
    WHERE NOT IsCancel
    GROUP BY CustomerID
),
returns AS (
    SELECT CustomerID, -SUM(Revenue) AS returned_value
    FROM clean_transactions
    WHERE IsCancel
    GROUP BY CustomerID
)
SELECT
    p.CustomerID,
    date_diff('day', CAST(p.last_purchase AS DATE), CAST($snapshot AS DATE))   AS recency_days,
    p.frequency,
    p.gross_revenue - COALESCE(r.returned_value, 0)                           AS monetary,
    p.gross_revenue,
    COALESCE(r.returned_value, 0)                                             AS returned_value,
    date_diff('day', CAST(p.first_purchase AS DATE), CAST($snapshot AS DATE))  AS tenure_days
FROM purchases p
LEFT JOIN returns r USING (CustomerID)
ORDER BY p.CustomerID;
