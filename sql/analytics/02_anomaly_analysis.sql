-- QUESTION: Which venues produce the most severe physics anomalies (unexpected ball behavior)?
-- Demonstrates: Multi-table JOINs, Conditional Aggregation, Percentage Calculations

WITH VenueAnomalies AS (
    SELECT 
        m.venue,
        COUNT(a.delivery_id) as total_anomalies,
        AVG(a.anomaly_score_if) as avg_severity,
        SUM(CASE WHEN a.anomaly_type = 'extreme_deviation' THEN 1 ELSE 0 END) as extreme_count
    FROM bi.fact_anomalies a
    JOIN bi.fact_deliveries f ON a.delivery_id = f.delivery_id
    JOIN bi.dim_match m ON f.match_id = m.match_id
    GROUP BY 1
)
SELECT 
    venue,
    total_anomalies,
    ROUND(avg_severity, 3) as avg_severity,
    extreme_count,
    ROUND((extreme_count::numeric / total_anomalies) * 100, 1) as pct_extreme
FROM VenueAnomalies
WHERE total_anomalies > 10
ORDER BY avg_severity DESC;
