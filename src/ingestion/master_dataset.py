"""
master_dataset.py  –  MODULE 1-H
Orchestrates the full data pipeline and generates the master research dataset.

Pipeline:
  1. Ingest + clean HawkeyeStats          → hawkeye_clean.parquet
  2. Join with CricSheet (venue/date)     → hawkeye_with_venue.parquet
  3. Fetch weather (Open-Meteo)           → hawkeye_with_weather.parquet
  4. Add ball-state features              → hawkeye_with_ball_state.parquet
  5. Build bowler profiles (train-only)   → bowler_profiles.parquet
  6. Merge bowler profile features        → master_dataset.parquet
  7. Generate data dictionary             → data_dictionary.csv
  8. Generate missing data report         → missing_data_report.csv
  9. Generate data lineage table          → dataset_lineage.csv

The master_dataset.parquet is the SINGLE source of truth for all experiments.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from config import DATA_MASTER, DATA_PROCESSED, DATA_BOWLER, RANDOM_SEED

DATA_MASTER.mkdir(parents=True, exist_ok=True)


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data dictionary describing every column in the master dataset."""
    descriptions = {
        "delivery_id":           "Unique delivery identifier (format_matchId_delivery_string)",
        "format":                "Match format: IPL_Men/ODI_Men/Test_Men/IPL_Women/ODI_Women/Test_Women",
        "match_id":              "Match identifier (from HawkeyeStats; format-specific)",
        "delivery_str":          "Original delivery string: innings.over.ball",
        "innings":               "Innings number (1-4 for Tests, 1-2 for limited overs)",
        "over_num":              "Over number within innings (0-indexed)",
        "ball_in_over":          "Ball number within over (1-6, plus extras)",
        "bowler":                "Bowler name (from HawkeyeStats)",
        "bowler_id":             "Bowler identifier (from HawkeyeStats/ICC API)",
        "bowling_style":         "Bowling style: FAST_SEAM/MEDIUM_SEAM/OFF_SPIN/ORTHODOX/LEG_SPIN/UNORTHODOX",
        "right_armed_bowl":      "True if bowler is right-armed",
        "batter":                "Batter name",
        "batter_id":             "Batter identifier",
        "right_handed_bat":      "True if batter is right-handed",
        "non_striker":           "Non-striker name",
        "matchup":               "Batter-bowler handedness matchup: RH_vs_RA, LH_vs_LA, etc.",
        "ball_speed_ms":         "Ball release speed in metres/second (NaN if invalid or missing)",
        "ball_speed_kmh":        "Ball release speed in km/h (derived from ball_speed_ms)",
        "pitch_x":               "Lateral position of ball bounce (metres from pitch centre; negative = off-side for RHB)",
        "pitch_y":               "Length of ball from bowler's stumps to bounce point (metres)",
        "stumps_x":              "Lateral position of ball as it passes batter's stumps (metres)",
        "stumps_y":              "Height of ball at batter's stumps (metres above ground)",
        "field_x":               "Fielding position X coordinate after shot (zone; sparse)",
        "field_y":               "Fielding position Y coordinate after shot (zone; sparse)",
        "lateral_swing":         "Lateral movement: stumps_x - pitch_x (metres; positive = away from stumps centre)",
        "stumps_off_centre":     "Absolute lateral distance of ball from stumps centre (|stumps_x|)",
        "pitch_off_centre":      "Absolute lateral distance of bounce from pitch centre (|pitch_x|)",
        "length_from_batter":    "Distance from batter's stumps where ball pitched (20.12 - pitch_y)",
        "delivery_type":         "Delivery length classification: full_toss/yorker/full/good_length/short_of_length/short",
        "height_class":          "Ball height at stumps: below_knee/knee/hip/waist/chest/head",
        "ball_age_overs":        "Continuous ball age in overs (over_num + ball_in_over/6)",
        "ball_age_since_replacement": "Age since last ball replacement (resets at over 80 for Test)",
        "ball_id_within_innings":"Ball entity within innings: A (first 80 overs) / B (80+ in Test)",
        "ball_phase":            "Ball condition phase: new_ball/swinging/worn/old/second_new_ball etc.",
        "ball_type":             "Ball brand/type: SG/Dukes/Kookaburra/Kookaburra_White/Kookaburra_Pink (rule-based)",
        "is_new_ball_period":    "1 if ball < 10 overs old or 80-90 overs (second new ball period)",
        "runs":                  "Total runs scored on this delivery",
        "batter_runs":           "Runs scored by the batter",
        "bowler_runs":           "Runs conceded by the bowler",
        "extras":                "Extras code: Wd/Nb/Lb/B/NbB/NbLb/WdB",
        "is_wide":               "1 if delivery was a wide",
        "is_no_ball":            "1 if delivery was a no-ball",
        "is_leg_bye":            "1 if delivery was a leg-bye",
        "is_bye":                "1 if delivery was a bye",
        "dismissal_details":     "Dismissal mode (sparse; ~95% null)",
        "venue":                 "Venue/stadium name (from CricSheet join; may be NULL)",
        "city":                  "City (from CricSheet join)",
        "country":               "Country (from CricSheet join or city mapping)",
        "match_date":            "Match date (from CricSheet join)",
        "venue_join_confidence": "Confidence of venue join: HIGH/MEDIUM/LOW/NULL",
        "temperature_c":         "Temperature at match start in °C (from Open-Meteo; may be NULL)",
        "humidity_pct":          "Relative humidity % (from Open-Meteo)",
        "wind_speed_kmh":        "Wind speed in km/h (from Open-Meteo)",
        "wind_direction_deg":    "Wind direction in degrees (from Open-Meteo)",
        "precipitation_mm":      "Precipitation in mm (from Open-Meteo)",
        "pressure_hpa":          "Atmospheric pressure in hPa (from Open-Meteo)",
        "cloud_cover_pct":       "Cloud cover % (from Open-Meteo)",
        "weather_available":     "1 if weather data was successfully retrieved",
        "pitch_type":            "Pitch surface classification (from external sources; mostly NULL)",
        "roll_speed_1ov":        "Rolling avg speed (km/h) over last 1 over (ball-state feature)",
        "roll_speed_5ov":        "Rolling avg speed (km/h) over last 5 overs (ball-state feature)",
        "roll_speed_10ov":       "Rolling avg speed (km/h) over last 10 overs (ball-state feature)",
        "roll_swing_1ov":        "Rolling avg lateral swing over last 1 over",
        "roll_swing_5ov":        "Rolling avg lateral swing over last 5 overs",
        "roll_swing_10ov":       "Rolling avg lateral swing over last 10 overs",
        "roll_bounce_1ov":       "Rolling avg stumps_y over last 1 over",
        "roll_bounce_5ov":       "Rolling avg stumps_y over last 5 overs",
        "speed_decline_trend":   "Speed trend: recent avg - early avg (negative = slowing ball)",
        "swing_trend":           "Swing trend: recent avg - early avg",
        "roughness_proxy":       "Rolling std of lateral swing (proxy for ball roughness/unpredictability)",
        "ball_condition":        "Discrete ball condition: NEW/LIGHTLY_WORN/MODERATELY_WORN/WORN/OLD",
        "bp_career_avg_speed_kmh": "Bowler profile: career average speed (km/h) — from training split only",
        "bp_career_speed_cv":    "Bowler profile: coefficient of variation of speed (consistency)",
        "bp_career_avg_pitch_y": "Bowler profile: typical pitch length (metres)",
        "bp_career_avg_lateral_swing": "Bowler profile: typical lateral swing (metres)",
        "bp_career_wide_rate":   "Bowler profile: historical wide rate",
    }

    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        n_unique = df[col].nunique() if df[col].dtype == "object" else "—"
        example = df[col].dropna().iloc[0] if df[col].notna().any() else "—"
        rows.append({
            "column":         col,
            "dtype":          dtype,
            "null_pct":       round(null_pct, 2),
            "n_unique":       n_unique,
            "example":        str(example)[:80],
            "description":    descriptions.get(col, ""),
        })
    return pd.DataFrame(rows)


def build_lineage_table() -> pd.DataFrame:
    """Build the data lineage table documenting all sources and transformations."""
    lineage = [
        {"source_dataset": "HawkeyeStats", "source_files": "6 CSV files",
         "deliveries": "1,131,102", "join_method": "primary source",
         "confidence": "HIGH", "fields_provided": "matchId,delivery,bowler,batter,speed,pitchX,pitchY,stumpsX,stumpsY,extras",
         "transformations": "Rename columns, clean coordinates (bounds filtering), derive ball_age/delivery_type/swing/height_class/wide/no_ball"},
        {"source_dataset": "CricSheet", "source_files": "JSON ball-by-ball files",
         "deliveries": "matched to HawkeyeStats", "join_method": "player overlap matching",
         "confidence": "HIGH/MEDIUM/LOW", "fields_provided": "venue,city,country,match_date",
         "transformations": "Parse JSON, build player sets per match, compute overlap score, merge by match_id"},
        {"source_dataset": "Open-Meteo API", "source_files": "Historical archive API",
         "deliveries": "matched by (venue,date)", "join_method": "venue coordinates + date",
         "confidence": "MEDIUM (hourly approximation)", "fields_provided": "temperature,humidity,wind,pressure,cloud_cover,precipitation",
         "transformations": "Fetch hourly data at 10:00 local time, cache per (lat,lon,date)"},
        {"source_dataset": "HawkeyeStats (derived)", "source_files": "Same as above",
         "deliveries": "all", "join_method": "computed within group",
         "confidence": "MEDIUM", "fields_provided": "rolling ball-state features (speed/swing/bounce trends)",
         "transformations": "Sort by match/innings/over/ball, compute causal rolling windows (1/5/10 over)"},
        {"source_dataset": "HawkeyeStats training split", "source_files": "Same",
         "deliveries": "training only", "join_method": "merged by bowler_id",
         "confidence": "HIGH (leakage-free)", "fields_provided": "bowler profile features (career speed/line/length/swing/wide-rate)",
         "transformations": "Group by bowler_id on training split, compute career aggregates, impute sparse bowlers with style median"},
    ]
    return pd.DataFrame(lineage)


def run_full_pipeline(skip_cricsheet: bool = False, skip_weather: bool = False) -> pd.DataFrame:
    """
    Run the complete data pipeline end-to-end.
    Set skip_cricsheet=True or skip_weather=True to skip those API calls
    (useful when running offline or for testing).
    """
    print("\n" + "=" * 65)
    print("  CRICKET BALL INTELLIGENCE — MASTER DATASET PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")

    # ── STEP 1: Ingest + clean HawkeyeStats ──────────────────────────────────
    from ingestion.hawkeye_ingest import run_ingestion
    df = run_ingestion()
    df.to_parquet(DATA_PROCESSED / "hawkeye_clean.parquet", index=False)

    # ── STEP 2: Ball-state features (before join so rolling is clean) ─────────
    from features.ball_state import build_ball_state_features, build_ball_summary_table
    df = build_ball_state_features(df)
    df.to_parquet(DATA_PROCESSED / "hawkeye_with_ball_state.parquet", index=False)

    # ── STEP 3: CricSheet venue/date join ─────────────────────────────────────
    if not skip_cricsheet:
        from ingestion.cricsheet_join import run_cricsheet_join
        df = run_cricsheet_join(df)
        df.to_parquet(DATA_PROCESSED / "hawkeye_with_venue.parquet", index=False)
    else:
        print("[SKIP] CricSheet join")

    # ── STEP 4: Weather fetch ─────────────────────────────────────────────────
    if not skip_weather and not skip_cricsheet:
        from ingestion.weather_fetch import run_weather_fetch
        df = run_weather_fetch(df)
        df.to_parquet(DATA_PROCESSED / "hawkeye_with_weather.parquet", index=False)
    else:
        print("[SKIP] Weather fetch")

    # ── STEP 5: Build bowler profiles from training split ─────────────────────
    from sklearn.model_selection import GroupShuffleSplit
    from features.bowler_profiles import build_bowler_profiles, save_bowler_profiles, merge_bowler_features

    print("\n" + "=" * 65)
    print("MODULE 1-C: Bowler Profile (training split)")
    print("=" * 65)
    groups = df["match_id"].astype(str) + "_" + df["format"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, _ = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    print(f"Training split: {len(train_df):,} rows ({len(train_df['match_id'].unique()):,} matches)")

    profiles = build_bowler_profiles(train_df)
    save_bowler_profiles(profiles)

    # ── STEP 6: Merge bowler features onto full dataset ───────────────────────
    df = merge_bowler_features(df, profiles)
    print(f"\n✓ Bowler profile features merged. Total columns: {df.shape[1]}")

    # ── STEP 7: Final master dataset ──────────────────────────────────────────
    master_path = DATA_MASTER / "master_dataset.parquet"
    df.to_parquet(master_path, index=False)
    print(f"\n✓ Master dataset saved: {master_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Valid trajectory rows: {df.get('has_trajectory', pd.Series(dtype=int)).sum():,}")

    # ── STEP 8: Generate data dictionary ─────────────────────────────────────
    print("\nGenerating data dictionary ...")
    data_dict = build_data_dictionary(df)
    dict_path = DATA_MASTER / "data_dictionary.csv"
    data_dict.to_csv(dict_path, index=False)
    print(f"✓ Data dictionary saved: {dict_path}  ({len(data_dict)} columns documented)")

    # ── STEP 9: Missing data report ───────────────────────────────────────────
    miss = df.isnull().mean().reset_index()
    miss.columns = ["column", "missing_fraction"]
    miss["missing_pct"] = (miss["missing_fraction"] * 100).round(2)
    miss["total_null"] = (miss["missing_fraction"] * len(df)).astype(int)
    miss_path = DATA_MASTER / "missing_data_report.csv"
    miss.to_csv(miss_path, index=False)
    print(f"✓ Missing data report: {miss_path}")

    # ── STEP 10: Data lineage ─────────────────────────────────────────────────
    lineage = build_lineage_table()
    lineage_path = DATA_MASTER / "dataset_lineage.csv"
    lineage.to_csv(lineage_path, index=False)
    print(f"✓ Lineage table: {lineage_path}")

    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build master cricket dataset")
    parser.add_argument("--skip-cricsheet", action="store_true",
                        help="Skip CricSheet download/join (offline mode)")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Skip Open-Meteo weather fetch")
    args = parser.parse_args()

    df = run_full_pipeline(
        skip_cricsheet=args.skip_cricsheet,
        skip_weather=args.skip_weather
    )
