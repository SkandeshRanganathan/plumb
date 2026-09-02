# PLUMB Power BI DAX Measures

To maintain a clean and professional portfolio dashboard, all DAX measures should be placed in a dedicated measure table.

## Creating a Measure Table
1. In Power BI, go to the **Home** ribbon and click **Enter Data**.
2. Name the table `_Measures` (the underscore keeps it at the top of the list) and click **Load**.
3. Right-click this new table in the Data pane to add the following New Measures.

## 1. Core Volume Metrics
```dax
Total Deliveries = COUNT(fact_deliveries[delivery_id])

Total Wides = CALCULATE(COUNT(fact_deliveries[delivery_id]), fact_deliveries[is_wide] = 1)

Wide Percentage = DIVIDE([Total Wides], [Total Deliveries], 0)
```

## 2. Physics & Delivery Metrics
```dax
Avg Speed (kph) = AVERAGE(fact_deliveries[ball_speed_kmh])

Max Speed (kph) = MAX(fact_deliveries[ball_speed_kmh])

Avg Lateral Swing (m) = AVERAGE(fact_deliveries[lateral_swing])

Speed Variability (Std Dev) = STDEV.S(fact_deliveries[ball_speed_kmh])
```

## 3. Anomaly Intelligence (Machine Learning Metrics)
```dax
Total Anomalies = COUNT(fact_anomalies[delivery_id])

Anomaly Rate = DIVIDE([Total Anomalies], [Total Deliveries], 0)

Avg Anomaly Severity = AVERAGE(fact_anomalies[anomaly_score_if])

Extreme Anomaly Count = CALCULATE([Total Anomalies], fact_anomalies[anomaly_type] = "extreme_deviation")
```

## How to Use These Measures
Because we built a Star Schema, these measures are **fully dynamic**. If you put `Avg Speed (kph)` in a Card visual, it will show the global average. But if you click on "Shaun Tait" in a Bowler slicer, the `Avg Speed (kph)` card will instantly recalculate to only show Shaun Tait's average speed!
