"""
anomaly_detection.py  –  MODULE 10
Unusual / Mystery Delivery Detection

Stage 1: Anomaly detector
  - Uses Isolation Forest and reconstruction error on trajectory features
  - Identifies deliveries where actual trajectory significantly differs
    from BOTH the physics prediction AND the bowler's historical profile

Stage 2: Anomaly classifier (if anomaly detected)
  - Attempts to assign: knuckleball / cutter / slower_ball /
    unusual_swing / unusual_seam / UNKNOWN
  - UNKNOWN is a valid output (open-set recognition)

Key signal: residual = actual - physics_predicted
  Large |residual_stumps_x| = unexpected lateral movement (swing/seam anomaly)
  Large |residual_stumps_y| = unexpected height (unusual bounce, top-spin etc.)
  Speed much lower than bowler average = slower ball / knuckleball candidate
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from config import DATA_MASTER, MODELS_SAVED, EXPERIMENTS, RANDOM_SEED

MODELS_SAVED.mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "results").mkdir(parents=True, exist_ok=True)


ANOMALY_FEATURES = [
    # Core trajectory
    "ball_speed_kmh",
    "pitch_x", "pitch_y",
    "stumps_x", "stumps_y",
    "lateral_swing",
    # Residual from physics model
    "residual_stumps_x", "residual_stumps_y",
    "residual_pitch_x",  "residual_pitch_y",
    # Ball state rolling features
    "roll_speed_5ov",
    "roll_swing_5ov",
    "roll_bounce_5ov",
    "roughness_proxy",
    # Bowler deviation from profile
    "speed_vs_career_avg",   # derived below
    "swing_vs_career_avg",   # derived below
    "pitch_y_vs_career_avg", # derived below
]


def build_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery-level deviation from bowler career averages.
    These are the primary anomaly signals.
    """
    df["speed_vs_career_avg"]    = df["ball_speed_kmh"] - df.get("bp_career_avg_speed_kmh", df["ball_speed_kmh"])
    df["swing_vs_career_avg"]    = df["lateral_swing"]  - df.get("bp_career_avg_lateral_swing", pd.Series(0, index=df.index))
    df["pitch_y_vs_career_avg"]  = df["pitch_y"]        - df.get("bp_career_avg_pitch_y", pd.Series(6.0, index=df.index))
    df["speed_z_score"] = (
        df["speed_vs_career_avg"] /
        df.get("bp_career_speed_std", pd.Series(5.0, index=df.index)).replace(0, 5.0)
    )
    return df


def train_isolation_forest(X_train: pd.DataFrame,
                             contamination: float = 0.05) -> object:
    """Train Isolation Forest for anomaly detection."""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=min(len(X_train.columns), 8),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train)
    return clf


def detect_anomalies(df: pd.DataFrame,
                      clf=None,
                      threshold_sigma: float = 3.0) -> pd.DataFrame:
    """
    Stage 1: Detect anomalous deliveries.
    Two-method ensemble:
      Method A: Isolation Forest score < threshold
      Method B: Residual-based z-score (|residual| > N*sigma of bowler-style distribution)

    Returns df with added columns:
      anomaly_score, anomaly_if_flag, anomaly_residual_flag, anomaly_flag (ensemble)
    """
    df = build_deviation_features(df)

    avail_feats = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[avail_feats].fillna(0.0)

    if clf is None:
        print("  Training Isolation Forest ...")
        # Train only on rows with valid trajectory
        valid_mask = df["stumps_x"].notna() & df["ball_speed_kmh"].notna()
        X_train = X[valid_mask]
        clf = train_isolation_forest(X_train)

    # Isolation Forest: predict = -1 is anomaly
    if_scores = clf.score_samples(X)   # more negative = more anomalous
    df["anomaly_score_if"] = if_scores
    df["anomaly_if_flag"]  = (clf.predict(X) == -1).astype(int)

    # Residual z-score method
    # Compute per bowling-style residual distribution
    if "residual_stumps_x" in df.columns and "residual_stumps_y" in df.columns:
        style_stats = df.groupby("bowling_style")[["residual_stumps_x","residual_stumps_y"]].agg(
            ["mean","std"]
        )
        style_stats.columns = ["rx_mean","rx_std","ry_mean","ry_std"]
        df = df.join(style_stats, on="bowling_style", rsuffix="_sty")
        df["rx_std"].fillna(0.3, inplace=True)
        df["ry_std"].fillna(0.2, inplace=True)
        df["z_rx"] = (df["residual_stumps_x"] - df["rx_mean"]).abs() / df["rx_std"].clip(lower=0.01)
        df["z_ry"] = (df["residual_stumps_y"] - df["ry_mean"]).abs() / df["ry_std"].clip(lower=0.01)
        df["anomaly_residual_flag"] = ((df["z_rx"] > threshold_sigma) |
                                        (df["z_ry"] > threshold_sigma)).astype(int)
        df["anomaly_residual_score"] = np.sqrt(df["z_rx"]**2 + df["z_ry"]**2)
    else:
        df["anomaly_residual_flag"]  = 0
        df["anomaly_residual_score"] = 0.0

    # Ensemble: flag if EITHER method flags
    df["anomaly_flag"] = ((df["anomaly_if_flag"] == 1) |
                           (df["anomaly_residual_flag"] == 1)).astype(int)

    n_anomalies = df["anomaly_flag"].sum()
    total = len(df)
    print(f"  Anomaly detection: {n_anomalies:,} flagged ({100*n_anomalies/total:.2f}%)")
    return df, clf


def classify_anomaly_type(row: pd.Series) -> str:
    """
    Stage 2: Heuristic classification of anomaly type.
    Returns UNKNOWN if no confident pattern is found.
    This is intentionally open-set — an unexpected delivery
    SHOULD be labelable as UNKNOWN.
    """
    speed_z = row.get("speed_z_score", 0) or 0
    rx = abs(row.get("residual_stumps_x", 0) or 0)
    ry = abs(row.get("residual_stumps_y", 0) or 0)
    swing = row.get("lateral_swing", 0) or 0
    pitch_y = row.get("pitch_y", 8) or 8
    style = str(row.get("bowling_style", ""))

    # Slower ball: speed significantly below bowler's average
    if speed_z < -2.5:
        if "FAST" in style or "MEDIUM" in style:
            return "slower_ball"

    # Knuckleball: slower ball + unusual trajectory
    if speed_z < -2.0 and (rx > 0.15 or ry > 0.15):
        return "knuckleball_candidate"

    # Unusual lateral swing (away from bowler's typical direction)
    if rx > 0.25 and abs(swing) > 0.3:
        if "FAST" in style or "SEAM" in style:
            return "unusual_swing_seam"

    # Unusual bounce height
    if ry > 0.20:
        if row.get("pitch_y", 8) > 11:
            return "unusual_bounce_short"
        else:
            return "unusual_bounce_full"

    # Cutter: seam/swing bowler, faster than normal slow ball, lateral deviation
    if "FAST" in style and speed_z > -1.5 and rx > 0.15:
        return "cutter_candidate"

    # If residual is large but nothing else matches
    if rx > 0.3 or ry > 0.25:
        return "UNKNOWN"

    return "UNKNOWN"


def run_anomaly_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full anomaly detection pipeline.
    Returns (enriched_df, anomaly_summary_df)
    """
    print("=" * 65)
    print("MODULE 10: Anomaly / Unusual Delivery Detection")
    print("=" * 65)

    # Require physics residuals
    if "residual_stumps_x" not in df.columns:
        print("  [WARN] Physics residuals not computed. Run physics model first.")
        print("  Running anomaly detection on raw trajectory features only ...")

    df, clf = detect_anomalies(df)

    # Stage 2: classify anomalies
    anomaly_mask = df["anomaly_flag"] == 1
    if anomaly_mask.sum() > 0:
        df.loc[anomaly_mask, "anomaly_type"] = df[anomaly_mask].apply(
            classify_anomaly_type, axis=1
        )
        df.loc[~anomaly_mask, "anomaly_type"] = "normal"
    else:
        df["anomaly_type"] = "normal"

    # Save Isolation Forest model
    clf_path = MODELS_SAVED / "isolation_forest.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    # Summary table
    anomaly_df = df[df["anomaly_flag"] == 1][[
        "delivery_id", "bowler", "bowling_style",
        "ball_speed_kmh", "ball_age_overs",
        "residual_stumps_x" if "residual_stumps_x" in df.columns else "lateral_swing",
        "anomaly_score_if", "anomaly_residual_score",
        "anomaly_type", "speed_z_score"
    ]].sort_values("anomaly_residual_score", ascending=False)

    summary_path = EXPERIMENTS / "results" / "anomaly_detections.csv"
    anomaly_df.to_csv(summary_path, index=False)
    print(f"\n  Anomaly breakdown by type:")
    print(df["anomaly_type"].value_counts().to_string())
    print(f"\n  Saved {len(anomaly_df):,} anomalous deliveries to {summary_path}")

    return df, anomaly_df


if __name__ == "__main__":
    master_path = DATA_MASTER / "master_dataset.parquet"
    if not master_path.exists():
        print("Run master_dataset.py first.")
        sys.exit(1)
    df = pd.read_parquet(master_path)
    print(f"Loaded: {df.shape}")
    df, anomaly_df = run_anomaly_detection(df)
    df.to_parquet(DATA_MASTER / "master_with_anomalies.parquet", index=False)
    print("\nSample anomalous deliveries:")
    print(anomaly_df.head(10).to_string(index=False))
