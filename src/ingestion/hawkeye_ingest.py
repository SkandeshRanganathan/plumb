"""
hawkeye_ingest.py  –  MODULE 1-A
Ingest, validate, and clean all 6 HawkeyeStats CSV files into a single
unified DataFrame. Derives primary features and saves a parquet file.
"""

import sys
from pathlib import Path

# Make sure project src is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from config import (
    HAWKEYE_FILES, DATA_PROCESSED, COORD_BOUNDS,
    DELIVERY_LENGTHS, STUMPS_HEIGHT_CLASSES,
    NEW_BALL_OVERS, BALL_PHASES, BALL_TYPE_RULES,
    PITCH_LENGTH_M
)

# ── Ensure output dir exists ─────────────────────────────────────────────────
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────────────────

def classify_delivery_length(pitch_y: float) -> str:
    """Classify delivery length from pitchY (metres from bowler's stumps)."""
    for (lo, hi), label in DELIVERY_LENGTHS.items():
        if lo <= pitch_y < hi:
            return label
    return "unknown"


def classify_height(stumps_y: float) -> str:
    """Classify ball height at batter's stumps."""
    for (lo, hi), label in STUMPS_HEIGHT_CLASSES.items():
        if lo <= stumps_y < hi:
            return label
    return "unknown"


def get_ball_phase(over_num: float, format_: str) -> str:
    """
    Return ball phase label based on over number.
    Test cricket has a second new ball at over 80.
    T20/ODI: single ball per innings, so simplified.
    """
    if "Test" in format_:
        for (lo, hi), label in BALL_PHASES.items():
            if lo <= over_num < hi:
                return label
        return "unknown"
    else:
        # For T20 (20 overs) and ODI (50 overs) use simplified phasing
        if over_num < 6:
            return "powerplay"
        elif over_num < 16 if "IPL" in format_ else over_num < 40:
            return "middle"
        else:
            return "death"


def get_ball_type(format_: str, country: str = None) -> str:
    """Rule-based ball type. Country from venue join (may be None)."""
    key = (format_, country or "default")
    if key in BALL_TYPE_RULES:
        return BALL_TYPE_RULES[key]
    # Fallback to format-level default
    fallback = (format_, "default")
    return BALL_TYPE_RULES.get(fallback, "unknown")


def parse_delivery_string(delivery_series: pd.Series):
    """
    Parse 'innings.over.ball' delivery strings.
    Returns DataFrame with columns: innings, over_num, ball_in_over.
    """
    split = delivery_series.str.split(".", expand=True)
    result = pd.DataFrame({
        "innings":      pd.to_numeric(split[0], errors="coerce"),
        "over_num":     pd.to_numeric(split[1], errors="coerce"),
        "ball_in_over": pd.to_numeric(split[2], errors="coerce"),
    })
    return result


def compute_derived_features(df: pd.DataFrame, format_: str) -> pd.DataFrame:
    """
    Derive all features that can be computed purely from HawkeyeStats.
    No external data needed.
    """
    # Ball speed in km/h
    df["ball_speed_kmh"] = np.where(
        df["ball_speed_ms"] > 0,
        df["ball_speed_ms"] * 3.6,
        np.nan
    )

    # Ball age: continuous (overs.fraction)
    df["ball_age_overs"] = df["over_num"] + (df["ball_in_over"] / 6.0)

    # In Test, 2nd new ball at over 80; assign ball_id accordingly
    if "Test" in format_:
        df["ball_id_within_innings"] = np.where(df["over_num"] < 80, "A", "B")
        df["ball_age_since_replacement"] = np.where(
            df["over_num"] < 80,
            df["ball_age_overs"],
            df["ball_age_overs"] - 80.0
        )
    else:
        df["ball_id_within_innings"] = "A"
        df["ball_age_since_replacement"] = df["ball_age_overs"]

    # Ball phase
    df["ball_phase"] = df["over_num"].apply(lambda x: get_ball_phase(x, format_))

    # Is new ball period? (first 10 overs in any format; 80+ in Test = second new ball)
    df["is_new_ball_period"] = (
        (df["over_num"] < 10) |
        (("Test" in format_) & (df["over_num"] >= 80) & (df["over_num"] < 90))
    ).astype(int)

    # Delivery length classification from pitchY
    df["delivery_type"] = df["pitch_y"].apply(
        lambda y: classify_delivery_length(y) if pd.notna(y) else "unknown"
    )

    # Ball height at stumps
    df["height_class"] = df["stumps_y"].apply(
        lambda y: classify_height(y) if pd.notna(y) else "unknown"
    )

    # Lateral swing: pitchX → stumpsX delta
    # Positive = movement away from centre (away from stumps centre-line)
    df["lateral_swing"] = np.where(
        df[["pitch_x", "stumps_x"]].notna().all(axis=1),
        df["stumps_x"] - df["pitch_x"],
        np.nan
    )

    # Distance from stump centre (|stumpsX|)
    df["stumps_off_centre"] = df["stumps_x"].abs()

    # Distance from pitch centre
    df["pitch_off_centre"] = df["pitch_x"].abs()

    # Effective length (distance from batter's stumps where ball pitches)
    df["length_from_batter"] = np.where(
        df["pitch_y"].notna(),
        PITCH_LENGTH_M - df["pitch_y"],
        np.nan
    )

    # Wide label (from extras field)
    df["is_wide"]    = df["extras"].str.contains("Wd", na=False).astype(int)
    df["is_no_ball"] = df["extras"].str.contains("Nb", na=False).astype(int)
    df["is_leg_bye"] = df["extras"].str.contains("Lb", na=False).astype(int)
    df["is_bye"]     = df["extras"].fillna("").str.match(r"^B$").astype(int)

    # [OK] Ball-state features added. Ball condition distribution:
    df["ball_type"] = get_ball_type(format_)

    # Handedness encoding
    df["batter_is_right"] = df["right_handed_bat"].astype(bool)
    df["bowler_is_right"]  = df["right_armed_bowl"].astype(bool)

    # Match-up type
    df["matchup"] = (
        df["batter_is_right"].map({True: "RH", False: "LH"}) + "_vs_" +
        df["bowler_is_right"].map({True: "RA", False: "LA"})
    )

    # Coordinate validity flag
    df["has_pitch_xy"]   = (df["pitch_x"].notna() & df["pitch_y"].notna()).astype(int)
    df["has_stumps_xy"]  = (df["stumps_x"].notna() & df["stumps_y"].notna()).astype(int)
    df["has_speed"]      = (df["ball_speed_ms"] > 0).astype(int)
    df["has_trajectory"] = (df["has_pitch_xy"] & df["has_stumps_xy"] & df["has_speed"]).astype(int)

    return df


def load_and_clean_one(format_: str, filepath: Path) -> pd.DataFrame:
    """Load a single HawkeyeStats CSV, clean, and derive primary features."""
    print(f"  Loading {format_} from {filepath.name} ...")

    df = pd.read_csv(filepath, low_memory=False)

    # ── Rename columns to snake_case ─────────────────────────────────────────
    rename = {
        "matchId":          "match_id",
        "delivery":         "delivery_str",
        "ball":             "ball_num_seq",
        "batter":           "batter",
        "batterId":         "batter_id",
        "rightHandedBat":   "right_handed_bat",
        "nonStriker":       "non_striker",
        "nonStrikerId":     "non_striker_id",
        "bowler":           "bowler",
        "bowlerId":         "bowler_id",
        "rightArmedBowl":   "right_armed_bowl",
        "bowlingStyle":     "bowling_style",
        "ballSpeed":        "ball_speed_ms",
        "dismissalDetails": "dismissal_details",
        "runs":             "runs",
        "batterRuns":       "batter_runs",
        "bowlerRuns":       "bowler_runs",
        "extras":           "extras",
        "pitchX":           "pitch_x",
        "pitchY":           "pitch_y",
        "stumpsX":          "stumps_x",
        "stumpsY":          "stumps_y",
        "fieldX":           "field_x",
        "fieldY":           "field_y",
    }
    df = df.rename(columns=rename)

    # ── Add format column ─────────────────────────────────────────────────────
    df["format"] = format_

    # ── Force numeric types on coordinate columns ─────────────────────────────
    numeric_cols = ["ball_speed_ms", "pitch_x", "pitch_y",
                    "stumps_x", "stumps_y", "field_x", "field_y",
                    "batter_id", "bowler_id", "non_striker_id",
                    "runs", "batter_runs", "bowler_runs",
                    "right_handed_bat", "right_armed_bowl"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure bool-like columns are stored as float (1.0/0.0/NaN) not object
    # to avoid PyArrow mixed-type errors when saving to parquet
    for bool_col in ["right_handed_bat", "right_armed_bowl"]:
        if bool_col in df.columns:
            df[bool_col] = pd.to_numeric(df[bool_col], errors="coerce").astype("float32")


    # ── Parse delivery string ─────────────────────────────────────────────────
    parsed = parse_delivery_string(df["delivery_str"])
    df = pd.concat([df, parsed], axis=1)

    # ── Apply coordinate bounds cleaning ─────────────────────────────────────
    cb = COORD_BOUNDS
    df.loc[df["pitch_x"].abs() > cb["pitchX_max"], "pitch_x"] = np.nan
    df.loc[df["pitch_y"] < cb["pitchY_min"],        "pitch_y"] = np.nan
    df.loc[df["pitch_y"] > cb["pitchY_max"],        "pitch_y"] = np.nan
    df.loc[df["stumps_x"].abs() > cb["stumpsX_max"], "stumps_x"] = np.nan
    df.loc[df["stumps_y"] < cb["stumpsY_min"],       "stumps_y"] = np.nan
    df.loc[df["stumps_y"] > cb["stumpsY_max"],       "stumps_y"] = np.nan
    # Invalidate speed sentinel (-1) and out-of-range values
    df.loc[
        (df["ball_speed_ms"] < cb["speed_min_ms"]) |
        (df["ball_speed_ms"] > cb["speed_max_ms"]),
        "ball_speed_ms"
    ] = np.nan

    # ── Create unique delivery_id ─────────────────────────────────────────────
    df["delivery_id"] = (
        format_ + "_" +
        df["match_id"].astype(str) + "_" +
        df["delivery_str"].astype(str)
    )

    # ── Derive features ───────────────────────────────────────────────────────
    df = compute_derived_features(df, format_)

    # ── Placeholder columns for external join (filled later) ─────────────────
    for col in ["venue", "city", "country", "match_date",
                "temperature_c", "humidity_pct", "wind_speed_kmh",
                "wind_direction_deg", "cloud_cover_pct",
                "precipitation_mm", "pressure_hpa",
                "pitch_type", "pitch_condition",
                "venue_join_confidence", "weather_available"]:
        if col not in df.columns:
            df[col] = np.nan if col not in ["venue", "city", "country",
                                             "match_date", "pitch_type",
                                             "pitch_condition",
                                             "venue_join_confidence"] else None

    return df


def run_ingestion() -> pd.DataFrame:
    """
    Main entry point: load all 6 HawkeyeStats CSVs, clean, derive features,
    and save to data/processed/hawkeye_clean.parquet.
    """
    print("=" * 65)
    print("MODULE 1-A: HawkeyeStats Ingestion & Cleaning")
    print("=" * 65)

    all_dfs = []
    for format_, filepath in HAWKEYE_FILES.items():
        if not filepath.exists():
            print(f"  [WARN] {format_}: file not found at {filepath}")
            continue
        df = load_and_clean_one(format_, filepath)
        all_dfs.append(df)
        print(f"  [OK] {format_}: {len(df):,} rows loaded")

    print()
    print("Concatenating all formats ...")
    master = pd.concat(all_dfs, ignore_index=True, sort=False)

    print(f"Total rows (all formats): {len(master):,}")
    print(f"Rows with valid trajectory: {master['has_trajectory'].sum():,}")
    print(f"Wides: {master['is_wide'].sum():,}")
    print(f"No-balls: {master['is_no_ball'].sum():,}")
    print()

    # ── Column ordering ───────────────────────────────────────────────────────
    front_cols = [
        "delivery_id", "format", "match_id", "delivery_str",
        "innings", "over_num", "ball_in_over",
        "bowler", "bowler_id", "bowling_style",
        "right_armed_bowl", "batter", "batter_id", "right_handed_bat",
        "non_striker", "non_striker_id", "matchup",
        "ball_speed_ms", "ball_speed_kmh",
        "pitch_x", "pitch_y", "stumps_x", "stumps_y",
        "field_x", "field_y",
        "lateral_swing", "stumps_off_centre", "pitch_off_centre",
        "length_from_batter",
        "delivery_type", "height_class",
        "ball_age_overs", "ball_age_since_replacement",
        "ball_id_within_innings", "ball_phase", "ball_type",
        "is_new_ball_period",
        "runs", "batter_runs", "bowler_runs", "extras",
        "is_wide", "is_no_ball", "is_leg_bye", "is_bye",
        "dismissal_details",
        "batter_is_right", "bowler_is_right",
        "has_pitch_xy", "has_stumps_xy", "has_speed", "has_trajectory",
        "venue", "city", "country", "match_date",
        "temperature_c", "humidity_pct", "wind_speed_kmh",
        "wind_direction_deg", "cloud_cover_pct",
        "precipitation_mm", "pressure_hpa",
        "pitch_type", "pitch_condition",
        "venue_join_confidence", "weather_available",
        "ball_type",
    ]
    # Keep only columns that actually exist, then add any extras at end
    existing_front = [c for c in front_cols if c in master.columns]
    remaining = [c for c in master.columns if c not in existing_front]
    master = master[existing_front + remaining]
    # Drop duplicate ball_type if it appeared twice from the ordering above
    master = master.loc[:, ~master.columns.duplicated()]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_parquet = DATA_PROCESSED / "hawkeye_clean.parquet"
    out_csv_sample = DATA_PROCESSED / "hawkeye_clean_sample.csv"
    master.to_parquet(out_parquet, index=False)
    master.head(5000).to_csv(out_csv_sample, index=False)

    print(f"Saved: {out_parquet}")
    print(f"Saved sample: {out_csv_sample}")
    print()

    # ── Missing value report ──────────────────────────────────────────────────
    miss = master.isnull().mean().reset_index()
    miss.columns = ["column", "missing_fraction"]
    miss["missing_pct"] = (miss["missing_fraction"] * 100).round(2)
    miss_path = DATA_PROCESSED / "missing_data_report.csv"
    miss.to_csv(miss_path, index=False)
    print("Missing data report:")
    key_cols = ["ball_speed_ms", "pitch_x", "pitch_y", "stumps_x", "stumps_y",
                "bowling_style", "venue", "country", "temperature_c"]
    print(miss[miss["column"].isin(key_cols)].to_string(index=False))
    print(f"\nFull report saved: {miss_path}")
    print()

    return master


if __name__ == "__main__":
    df = run_ingestion()
    print(f"\nDone. Final shape: {df.shape}")
    print("Columns:", list(df.columns))
