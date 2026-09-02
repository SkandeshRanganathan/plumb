# PLUMB Business Intelligence & Analytics Data Dictionary

This document outlines the Star Schema and analytical tables generated for the Power BI Dashboard and SQL Analytics engine.

## 1. Fact Tables

### `fact_deliveries.parquet`
**Grain:** 1 row = 1 individual cricket delivery bowled in a match.
- `delivery_id` (String) [Primary Key]: Unique identifier for the ball bowled.
- `match_id` (Int) [Foreign Key]: Links to `dim_match`.
- `bowler_id` (Int) [Foreign Key]: Links to `dim_bowler`.
- `batter_id` (Int): Identifier for the facing batsman.
- `over_num` (Int): The over number in the innings.
- `ball_in_over` (Int): The ball number within the over.
- `ball_speed_kmh` (Float): Release speed in kilometers per hour.
- `pitch_x` / `pitch_y` (Float): Coordinates of where the ball pitched on the wicket.
- `stumps_x` / `stumps_y` (Float): Coordinates of where the ball passed the stumps.
- `lateral_swing` (Float): Total lateral movement in the air (meters).
- `runs` (Float): Total runs scored off the delivery.
- `is_wide` / `is_no_ball` (Int): Boolean flags (1 = True, 0 = False) indicating extras.

### `fact_anomalies.parquet`
**Grain:** 1 row = 1 anomalous physics delivery detected by the ML engine.
- `delivery_id` (String) [Foreign Key]: Links to `fact_deliveries`.
- `anomaly_score_if` (Float): Isolation Forest severity score. Negative values indicate severe anomalies.
- `anomaly_type` (String): Categorical string indicating the type of anomaly (e.g. `extreme_deviation`, `unusual_speed`).

## 2. Dimension Tables

### `dim_bowler.parquet`
**Grain:** 1 row = 1 unique bowler.
- `bowler_id` (Int) [Primary Key]: Unique identifier.
- `bowler` (String): Full name of the bowler.
- `bowling_style` (String): Primary bowling technique (e.g. `FAST_SEAM`, `OFF_SPIN`).
- `right_armed_bowl` (Int): Boolean flag (1 = Right Arm, 0 = Left Arm).

### `dim_match.parquet`
**Grain:** 1 row = 1 unique match.
- `match_id` (Int) [Primary Key]: Unique identifier for the match.
- `venue` (String): Stadium name.
- `city` / `country` (String): Location data.
- `match_date` (String): Date the match was played.
- `format` (String): The format of the match (e.g. `T20`, `ODI`, `Test`).
- `weather_id` (Int) [Foreign Key]: Links to `dim_weather`.

### `dim_weather.parquet`
**Grain:** 1 row = 1 unique match weather state.
- `weather_id` (Int) [Primary Key]: Links to `dim_match`.
- `temperature_c` (Float): Air temperature in Celsius.
- `humidity_pct` (Float): Relative humidity percentage.
- `wind_speed_kmh` (Float): Wind speed.
- `cloud_cover_pct` (Float): Cloud cover percentage.
- `pitch_type` (String): Scraped classification of the pitch surface (e.g. `Standard Pitch`, `Dusty`).
