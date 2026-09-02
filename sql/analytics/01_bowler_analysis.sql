-- QUESTION: Who are the most consistent and lethal bowlers?
-- Demonstrates: CTEs, Window Functions (RANK), Aggregations, JOINs, HAVING clauses

WITH BowlerStats AS (
    SELECT 
        b.bowler,
        b.bowling_style,
        COUNT(f.delivery_id) as total_deliveries,
        AVG(f.ball_speed_kmh) as avg_speed_kmh,
        STDDEV(f.ball_speed_kmh) as speed_variability,
        AVG(f.lateral_swing) as avg_swing_m
    FROM bi.fact_deliveries f
    JOIN bi.dim_bowler b ON f.bowler_id = b.bowler_id
    GROUP BY 1, 2
    HAVING COUNT(f.delivery_id) >= 50
)
SELECT 
    bowler,
    bowling_style,
    total_deliveries,
    ROUND(avg_speed_kmh, 2) as avg_speed,
    ROUND(speed_variability, 2) as speed_variability,
    ROUND(avg_swing_m, 3) as avg_swing,
    RANK() OVER(ORDER BY avg_speed_kmh DESC) as speed_rank,
    RANK() OVER(ORDER BY avg_swing_m DESC) as swing_rank
FROM BowlerStats
ORDER BY speed_rank ASC;