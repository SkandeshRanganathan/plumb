"""
bowler_profiles.py  –  MODULE 1-C
Builds per-bowler historical profile features from the training split ONLY.
No test-set data is ever used — this prevents data leakage.

Profile features (per bowler, computed from training deliveries):
  - career_avg_speed_kmh, career_speed_std, career_speed_cv
  - career_avg_pitch_x, career_avg_pitch_y
  - career_avg_stumps_x, career_avg_stumps_y
  - career_avg_lateral_swing
  - career_wide_rate, career_no_ball_rate
  - career_deliveries
  - delivery_type_distribution (dict → encoded as separate columns)
  - avg_speed_by_over_bucket  (paced as fatigue/effort changes over spell)
  - bowling_style (mode)
  - formats_played

Ball-state signal features (rolling per-match, computed online — no leakage):
  - last_N_deliveries_avg_speed (within the SAME match/innings)
  - last_N_deliveries_avg_swing
  These are computed in the feature engineering step, not here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from config import DATA_BOWLER, DATA_PROCESSED, RANDOM_SEED

DATA_BOWLER.mkdir(parents=True, exist_ok=True)


def build_bowler_profiles(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bowler profile from training split deliveries.
    Returns a DataFrame indexed by bowler_id with profile columns.

    IMPORTANT: Only call this with the TRAINING SPLIT DataFrame.
    Never pass the full dataset to avoid leakage.
    """
    print("  Building bowler profiles from training split ...")

    required = {"bowler_id", "bowler", "bowling_style",
                "ball_speed_kmh", "pitch_x", "pitch_y",
                "stumps_x", "stumps_y", "lateral_swing",
                "is_wide", "is_no_ball", "delivery_type",
                "over_num", "format"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    valid_speed = train_df[train_df["ball_speed_kmh"].notna()]
    valid_traj  = train_df[train_df["has_trajectory"] == 1] if "has_trajectory" in train_df.columns else train_df

    # ── Base aggregates ───────────────────────────────────────────────────────
    base = train_df.groupby("bowler_id").agg(
        bowler_name          = ("bowler",          "first"),
        bowling_style        = ("bowling_style",   lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
        is_right_armed       = ("right_armed_bowl","first"),
        career_deliveries    = ("bowler_id",       "count"),
        career_wide_rate     = ("is_wide",         "mean"),
        career_no_ball_rate  = ("is_no_ball",      "mean"),
        formats_played       = ("format",          lambda x: list(x.unique())),
    ).reset_index()

    # ── Speed stats ───────────────────────────────────────────────────────────
    speed_stats = valid_speed.groupby("bowler_id").agg(
        career_avg_speed_kmh  = ("ball_speed_kmh", "mean"),
        career_speed_std      = ("ball_speed_kmh", "std"),
        career_speed_min      = ("ball_speed_kmh", "min"),
        career_speed_max      = ("ball_speed_kmh", "max"),
        career_speed_q25      = ("ball_speed_kmh", lambda x: x.quantile(0.25)),
        career_speed_q75      = ("ball_speed_kmh", lambda x: x.quantile(0.75)),
        career_speed_samples  = ("ball_speed_kmh", "count"),
    ).reset_index()
    speed_stats["career_speed_cv"] = (
        speed_stats["career_speed_std"] / speed_stats["career_avg_speed_kmh"]
    ).fillna(0).round(4)

    # ── Trajectory stats ──────────────────────────────────────────────────────
    traj_stats = valid_traj.groupby("bowler_id").agg(
        career_avg_pitch_x        = ("pitch_x",       "mean"),
        career_avg_pitch_y        = ("pitch_y",       "mean"),
        career_pitch_x_std        = ("pitch_x",       "std"),
        career_pitch_y_std        = ("pitch_y",       "std"),
        career_avg_stumps_x       = ("stumps_x",      "mean"),
        career_avg_stumps_y       = ("stumps_y",      "mean"),
        career_avg_lateral_swing  = ("lateral_swing", "mean"),
        career_swing_std          = ("lateral_swing", "std"),
        career_traj_samples       = ("pitch_x",       "count"),
    ).reset_index()

    # ── Delivery type distribution ────────────────────────────────────────────
    dtype_dist = (
        train_df.groupby(["bowler_id", "delivery_type"])
        .size()
        .reset_index(name="cnt")
    )
    dtype_total = dtype_dist.groupby("bowler_id")["cnt"].transform("sum")
    dtype_dist["pct"] = dtype_dist["cnt"] / dtype_total
    dtype_pivot = dtype_dist.pivot_table(
        index="bowler_id", columns="delivery_type", values="pct", fill_value=0.0
    ).add_prefix("dtype_pct_").reset_index()

    # ── Speed by over bucket (fatigue model) ─────────────────────────────────
    # Bucket overs into: powerplay (0-5), middle (6-14), death (15-20/40-49)
    def over_bucket(over):
        if over < 6:   return "pp"
        elif over < 15: return "mid"
        else:           return "death"
    train_speed = train_df[train_df["ball_speed_kmh"].notna()].copy()
    train_speed["over_bucket"] = train_speed["over_num"].apply(over_bucket)
    speed_by_phase = (
        train_speed.groupby(["bowler_id", "over_bucket"])["ball_speed_kmh"]
        .mean()
        .reset_index()
        .pivot_table(index="bowler_id", columns="over_bucket",
                     values="ball_speed_kmh", fill_value=np.nan)
        .add_prefix("avg_speed_")
        .reset_index()
    )

    # ── Merge all ─────────────────────────────────────────────────────────────
    profiles = base.copy()
    for extra in [speed_stats, traj_stats, dtype_pivot, speed_by_phase]:
        profiles = profiles.merge(extra, on="bowler_id", how="left")

    # ── Fill missing numeric stats with global style medians ─────────────────
    numeric_cols = profiles.select_dtypes(include=np.number).columns
    style_medians = profiles.groupby("bowling_style")[numeric_cols].transform("median")
    for col in numeric_cols:
        profiles[col] = profiles[col].fillna(style_medians[col])

    # ── Format lists as JSON strings for CSV compatibility ────────────────────
    profiles["formats_played"] = profiles["formats_played"].apply(
        lambda x: ",".join(sorted(set(x))) if isinstance(x, list) else str(x)
    )

    profiles = profiles.round(5)
    print(f"  ✓ Built profiles for {len(profiles):,} unique bowlers")
    return profiles


def save_bowler_profiles(profiles: pd.DataFrame) -> Path:
    """Save bowler profile table to parquet and CSV."""
    out_parquet = DATA_BOWLER / "bowler_profiles.parquet"
    out_csv     = DATA_BOWLER / "bowler_profiles.csv"
    profiles.to_parquet(out_parquet, index=False)
    profiles.to_csv(out_csv, index=False)
    print(f"  Saved: {out_parquet}")
    print(f"  Saved: {out_csv}")
    return out_parquet


def load_bowler_profiles() -> pd.DataFrame:
    """Load pre-computed bowler profiles."""
    path = DATA_BOWLER / "bowler_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError("Run build_bowler_profiles() first.")
    return pd.read_parquet(path)


def merge_bowler_features(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bowler profile features back onto the delivery DataFrame.
    Adds columns prefixed 'bp_' (bowler profile) to avoid confusion with
    delivery-level features.
    """
    profile_cols = [c for c in profiles.columns if c != "bowler_id"]
    bp_rename = {c: f"bp_{c}" for c in profile_cols}
    profiles_bp = profiles.rename(columns=bp_rename)
    merged = df.merge(profiles_bp, on="bowler_id", how="left")
    return merged


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 1-C: Bowler Profile Builder")
    print("=" * 65)

    # Load cleaned data
    parquet_path = DATA_PROCESSED / "hawkeye_clean.parquet"
    if not parquet_path.exists():
        print("Run hawkeye_ingest.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    # Build on TRAINING split (80% stratified by format+match)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    groups = df["match_id"].astype(str) + "_" + df["format"]
    train_idx, _ = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx]

    print(f"  Training split: {len(train_df):,} deliveries from "
          f"{train_df['match_id'].nunique():,} matches")

    profiles = build_bowler_profiles(train_df)
    save_bowler_profiles(profiles)

    # Sample output
    print("\n  Sample bowler profiles (top 5 by deliveries):")
    cols = ["bowler_name", "bowling_style", "career_deliveries",
            "career_avg_speed_kmh", "career_avg_pitch_y",
            "career_avg_lateral_swing", "career_wide_rate"]
    available = [c for c in cols if c in profiles.columns]
    print(profiles.nlargest(5, "career_deliveries")[available].to_string(index=False))
