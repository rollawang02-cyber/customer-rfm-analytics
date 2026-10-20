-- Clean line-level transactions.
-- Input : raw_transactions (the CSV schema, InvoiceDate already a TIMESTAMP)
-- Output: view clean_transactions with Revenue and IsCancel
-- Rules mirror rfm.data.CLEANING_STEPS; tests/test_sql_parity.py keeps them in sync.
CREATE OR REPLACE VIEW clean_transactions AS
WITH dedup AS (
    SELECT DISTINCT * FROM raw_transactions
)
SELECT
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    InvoiceDate,
    UnitPrice,
    CustomerID,
    Country,
    Quantity * UnitPrice              AS Revenue,
    starts_with(InvoiceNo, 'C')       AS IsCancel
FROM dedup
WHERE CustomerID IS NOT NULL
  AND UnitPrice > 0
  AND StockCode NOT IN ('POST', 'DOT', 'M', 'm', 'C2', 'D', 'S',
                        'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'B', 'PADS')
  AND NOT starts_with(StockCode, 'gift_');
