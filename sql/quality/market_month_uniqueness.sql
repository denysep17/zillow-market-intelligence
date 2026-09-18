-- Returns violations only. A valid mart should return zero rows.
SELECT
    market_id,
    month,
    COUNT(*) AS row_count
FROM mart_market_monthly
GROUP BY 1, 2
HAVING COUNT(*) > 1;
