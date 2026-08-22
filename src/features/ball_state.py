"""
ball_state.py  –  MODULE 1-G / MODULE 5
Builds the evolving ball-state representation.

Design:
  - For each (match_id, innings), group deliveries in order.
  - Compute rolling statistics over the last N deliveries of that ball.
  - Assign a discrete ball phase and a continuous latent state proxy.
  - Handle Test ball replacement at over 80 (hard reset to new ball).

CRITICAL: All rolling features use only PAST deliveries of the same ball.
No future information is ever used (respects prediction timestamp).

Latent Ball State:
  Rather than a neural embedding (requires GPU, reserved for Phase 2),
  we proxy ball state via a set of rolling statistics that capture
  deterioration signatures:
    - speed_decline_10:  avg_speed(last 10) - avg_speed(first 10)
    - swing_trend_10:    avg_swing(last 10) - avg_swing(first 10)  
    - bounce_change_10:  avg_stumps_y(last 10) - avg_stumps_y(first 10)
    - roughness_proxy:   rolling std of lateral swing (higher = more irregular)
    - age_since_replace: continuous overs since last ball replacement
    - phase:             discrete category
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from config import DATA_BALL_STATE, DATA_PROCESSED, BALL_PHASES

DATA_BALL_STATE.mkdir(parents=True, exist_ok=True)

WINDOW_SHORT = 6    # 1 over
WINDOW_MED   = 30   # 5 overs
WINDOW_LONG  = 60   # 10 overs


def assign_ball_id(group: pd.DataFrame) -> pd.DataFrame:
    """
    Within a single (match_id, innings) group (sorted by over_num, ball_in_over),
    assign ball_entity_id: 'A' for first 80 overs, 'B' for 80+ (Test only).
    Pandas 2.x groupby.apply may drop groupby key columns inside the function.
    We use a safe fallback by checking column existence.
    """
    group = group.sort_values(["over_num", "ball_in_over"]).reset_index(drop=True)

    # Safely get format (may be dropped by groupby in pandas 2.x)
    format_ = "unknown"
    if "format" in group.columns and len(group) > 0:
        format_ = str(group["format"].iloc[0])

    if "Test" in format_:
        group["ball_entity_id"] = np.where(group["over_num"] < 80, "A", "B")
        group["age_since_replace"] = np.where(
            group["over_num"] < 80,
            group["ball_age_overs"],
            group["ball_age_overs"] - 80.0
        )
    else:
        group["ball_entity_id"] = "A"
        group["age_since_replace"] = group["ball_age_overs"]
    return group



def compute_rolling_ball_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute causal rolling statistics within a single (match_id, innings, ball_entity)
    group. Uses only past deliveries — .shift(1) prevents current delivery leakage.
    """
    g = group.sort_values(["over_num", "ball_in_over"]).reset_index(drop=True)

    speed   = g["ball_speed_kmh"]
    swing   = g["lateral_swing"]
    bounce  = g["stumps_y"]
    pitch_x = g["pitch_x"]

    # Rolling means (shift(1) = exclude current delivery)
    for w, suffix in [(WINDOW_SHORT, "1ov"), (WINDOW_MED, "5ov"), (WINDOW_LONG, "10ov")]:
        g[f"roll_speed_{suffix}"]   = speed.shift(1).rolling(w, min_periods=1).mean()
        g[f"roll_swing_{suffix}"]   = swing.shift(1).rolling(w, min_periods=1).mean()
        g[f"roll_bounce_{suffix}"]  = bounce.shift(1).rolling(w, min_periods=1).mean()
        g[f"roll_pitchx_{suffix}"]  = pitch_x.shift(1).rolling(w, min_periods=1).mean()
        g[f"roll_swing_std_{suffix}"] = swing.shift(1).rolling(w, min_periods=1).std()

    # Trend features: recent minus early
    # Speed decline (negative = ball is slowing — sign of deterioration)
    early_speed = speed.rolling(WINDOW_LONG, min_periods=1).mean().shift(WINDOW_LONG)
    g["speed_decline_trend"] = g.get(f"roll_speed_10ov", np.nan) - early_speed

    # Swing trend (change in lateral movement over time)
    early_swing = swing.rolling(WINDOW_LONG, min_periods=1).mean().shift(WINDOW_LONG)
    g["swing_trend"] = g.get(f"roll_swing_10ov", np.nan) - early_swing

    # Roughness proxy: rolling std of swing (higher = more unpredictable movement)
    g["roughness_proxy"] = swing.shift(1).rolling(WINDOW_MED, min_periods=3).std()

    # Cumulative swing direction consistency
    # (positive swing fraction = how often ball swings away from off stump)
    g["swing_direction_consistency"] = (
        swing.gt(0).shift(1).rolling(WINDOW_MED, min_periods=3).mean()
    )

    # Speed entropy proxy (delivery-to-delivery variation)
    g["speed_variation_1ov"] = speed.shift(1).rolling(WINDOW_SHORT, min_periods=2).std()

    return g


def classify_ball_condition(row: pd.Series) -> str:
    """
    Assign a discrete ball condition label from continuous features.
    This is a heuristic proxy; the neural latent model (Phase 2) replaces this.
    """
    age = row.get("age_since_replace", 0) or 0
    speed_decline = row.get("speed_decline_trend", 0) or 0
    roughness = row.get("roughness_proxy", 0) or 0

    if age < 10:
        return "NEW"
    elif age < 25 and speed_decline > -2:
        return "LIGHTLY_WORN"
    elif age < 50:
        if roughness > 0.15:
            return "MODERATELY_WORN"
        return "MODERATELY_WORN"
    elif age < 70:
        if roughness > 0.25 or speed_decline < -3:
            return "WORN"
        return "WORN"
    else:
        return "OLD"


def build_ball_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the full delivery DataFrame:
      1. Sort within each (match_id, innings).
      2. Assign ball entity IDs.
      3. Compute rolling statistics per ball entity.
      4. Classify ball condition.
    Returns DataFrame with ~20 new ball-state feature columns.
    """
    print("  Building ball-state features ...")

    # Sort globally: match → innings → over → ball
    df = df.sort_values(
        ["match_id", "format", "innings", "over_num", "ball_in_over"],
        na_position="last"
    ).reset_index(drop=True)

    # Step 1: Assign ball_entity_id per (match, innings) — vectorized (avoid pandas 2.x groupby apply column drop)
    df["ball_entity_id"] = np.where(
        (df["format"].str.contains("Test", na=False)) & (df["over_num"] >= 80), "B", "A"
    )
    df["age_since_replace"] = np.where(
        df["ball_entity_id"] == "B",
        df["ball_age_overs"] - 80.0,
        df["ball_age_overs"]
    )

    # Step 2: Compute rolling features per (match, innings, ball_entity_id)
    keys = ["match_id", "format", "innings", "ball_entity_id"]
    original_keys = df[keys].copy()
    
    df = df.groupby(keys, group_keys=False).apply(compute_rolling_ball_features)
    
    # Restore keys if dropped by pandas 2.x groupby.apply
    for col in keys:
        if col not in df.columns:
            df[col] = original_keys[col]

    # Step 3: Classify discrete ball condition
    df["ball_condition"] = df.apply(classify_ball_condition, axis=1)
    df["ball_condition_confidence"] = np.where(
        df["age_since_replace"].notna() &
        df["roll_speed_10ov"].notna(),
        "MEDIUM", "LOW"
    )

    print("  [OK] Ball-state features added. Ball condition distribution:")
    print(df["ball_condition"].value_counts().to_string())

    return df


def build_ball_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the BALL_STATE summary table:
    One row per (match_id, innings, ball_entity_id) summarising
    the entire lifecycle of that ball.
    """
    summary = df.groupby(
        ["match_id", "format", "innings", "ball_entity_id"]
    ).agg(
        total_deliveries    = ("delivery_id", "count"),
        start_over          = ("over_num",    "min"),
        end_over            = ("over_num",    "max"),
        ball_type           = ("ball_type",   "first"),
        avg_speed_first10   = ("ball_speed_kmh",
                               lambda x: x.iloc[:10].mean() if len(x) >= 5 else np.nan),
        avg_speed_last10    = ("ball_speed_kmh",
                               lambda x: x.iloc[-10:].mean() if len(x) >= 5 else np.nan),
        speed_decline       = ("speed_decline_trend", "last"),
        avg_swing_first10   = ("lateral_swing",
                               lambda x: x.iloc[:10].mean() if len(x) >= 5 else np.nan),
        avg_swing_last10    = ("lateral_swing",
                               lambda x: x.iloc[-10:].mean() if len(x) >= 5 else np.nan),
        roughness_peak      = ("roughness_proxy", "max"),
        avg_bounce          = ("stumps_y",    "mean"),
        final_condition     = ("ball_condition", "last"),
    ).reset_index()

    summary["speed_decline_rate"] = (
        (summary["avg_speed_last10"] - summary["avg_speed_first10"]) /
        summary["total_deliveries"].clip(lower=1)
    )
    summary["swing_decline_rate"] = (
        (summary["avg_swing_last10"] - summary["avg_swing_first10"]) /
        summary["total_deliveries"].clip(lower=1)
    )
    return summary


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 1-G / MODULE 5: Ball State Modelling")
    print("=" * 65)

    parquet_path = DATA_PROCESSED / "hawkeye_clean.parquet"
    if not parquet_path.exists():
        print("Run hawkeye_ingest.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    df = build_ball_state_features(df)

    # Save enriched
    out = DATA_PROCESSED / "hawkeye_with_ball_state.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out}")

    # Save ball summary
    ball_summary = build_ball_summary_table(df)
    bs_out = DATA_BALL_STATE / "ball_state_summary.parquet"
    ball_summary.to_parquet(bs_out, index=False)
    ball_summary.to_csv(DATA_BALL_STATE / "ball_state_summary.csv", index=False)
    print(f"Saved: {bs_out}  ({len(ball_summary):,} ball entities)")
    print("\nSample ball summary:")
    print(ball_summary.head(5).to_string(index=False))
