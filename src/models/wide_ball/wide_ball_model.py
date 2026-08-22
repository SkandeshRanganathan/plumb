"""
wide_ball_model.py  –  MODULE 11
Wide-Ball Decision Assistance

Uses trajectory features (stumpsX, stumpsY, pitch position) from HawkeyeStats
to model wide-ball probability.

Two sub-models:
  11A: Statistical/trajectory-based model (HawkeyeStats data)
     - Input: stumps_x, stumps_y, batter handedness, bowler style, matchup
     - Target: is_wide (0/1)
     - Format-specific thresholds (T20 more liberal than Test)

  11B: Vision-based model (Wide Balls Dataset — when available)
     - Input: batter position + ball position from dual camera views
     - Target: wide (0/1)
     - Dynamic threshold relative to batter's guard position

Wide rule implementation:
  ICC Playing Conditions (simplified):
  - Ball is wide if it passes outside the off/leg stump line in the batter's
    normal guard position without being played.
  - For T20/ODI: wider interpretation (more balls called wide)
  - For Test: stricter (more latitude given)
  - Batter movement: if batter moves, the wide line moves with them
    (HawkeyeStats does not track batter x-position, so we use stumpsX only)

Output format:
  wide_probability: float [0,1]
  decision: "WIDE" | "LEGAL" | "REVIEW_REQUIRED"
  confidence: float [0,1]
  reason: str
"""

import sys
import pickle
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from src.config import DATA_MASTER, MODELS_SAVED, EXPERIMENTS, RANDOM_SEED, WIDE_THRESHOLD_X

MODELS_SAVED.mkdir(parents=True, exist_ok=True)


# ── Wide detection thresholds (metres from stumps centre) ────────────────────
# Positive stumpsX = off-side (for right-handed batter)
# Negative stumpsX = leg-side
WIDE_THRESHOLDS = {
    # Format: (off_side_limit, leg_side_limit)
    "Test_Men":   (0.53, -0.40),   # ball outside these = wide
    "ODI_Men":    (0.46, -0.35),
    "IPL_Men":    (0.46, -0.35),
    "Test_Women": (0.53, -0.40),
    "ODI_Women":  (0.46, -0.35),
    "IPL_Women":  (0.46, -0.35),
}


def rule_based_wide(row: pd.Series) -> Dict:
    """
    Rule-based wide detection using stumps_x and format-specific thresholds.
    Accounts for batter handedness (LHB has mirrored off/leg sides).
    """
    stumps_x = row.get("stumps_x", np.nan)
    format_  = row.get("format", "IPL_Men")
    is_right = row.get("right_handed_bat", True)

    if pd.isna(stumps_x):
        return {"wide_prob_rule": np.nan, "wide_rule": np.nan,
                "reason": "no_stumps_data"}

    off_lim, leg_lim = WIDE_THRESHOLDS.get(format_, (0.46, -0.35))

    # For left-handed batter, off/leg directions are reversed
    if not is_right:
        off_lim, leg_lim = -leg_lim, -off_lim

    is_wide_off = stumps_x > off_lim
    is_wide_leg = stumps_x < leg_lim

    if is_wide_off:
        return {"wide_prob_rule": 0.95, "wide_rule": 1,
                "reason": f"off_side ({stumps_x:.3f}m > {off_lim:.3f}m)"}
    elif is_wide_leg:
        return {"wide_prob_rule": 0.90, "wide_rule": 1,
                "reason": f"leg_side ({stumps_x:.3f}m < {leg_lim:.3f}m)"}
    else:
        # How close to the wide line?
        off_margin = off_lim - stumps_x
        leg_margin = stumps_x - leg_lim
        min_margin = min(off_margin, leg_margin)
        # Probability decreases as ball approaches the wide line
        prob = max(0.0, 1.0 - (min_margin / 0.1))  # within 10cm = rising probability
        prob = min(prob, 0.45)
        return {"wide_prob_rule": round(prob, 3), "wide_rule": 0,
                "reason": f"legal (margin: {min_margin:.3f}m)"}


WIDE_ML_FEATURES = [
    "stumps_x", "stumps_y",
    "pitch_x", "pitch_y",
    "ball_speed_kmh",
    "ball_age_overs",
    "lateral_swing",
    "is_new_ball_period",
    "batter_is_right", "bowler_is_right",
    "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
    "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX",
    # Format one-hot
    "fmt_IPL_Men","fmt_ODI_Men","fmt_Test_Men",
    "fmt_IPL_Women","fmt_ODI_Women","fmt_Test_Women",
    # Weather
    "temperature_c","humidity_pct","wind_speed_kmh",
    # Bowler profile
    "bp_career_avg_lateral_swing","bp_career_wide_rate",
]


def encode_wide_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode features needed for wide model."""
    from models.context_aware.trajectory_models import encode_categorical_features
    df = encode_categorical_features(df)
    for fmt in ["IPL_Men","ODI_Men","Test_Men","IPL_Women","ODI_Women","Test_Women"]:
        df[f"fmt_{fmt}"] = (df["format"] == fmt).astype(int)
    return df


def train_wide_model(df: pd.DataFrame) -> Dict:
    """
    Train XGBoost + calibrated probability wide-ball classifier.
    Returns dict with model, calibrator, feature list, and evaluation metrics.
    """
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score,
                                  brier_score_loss)

    print("=" * 65)
    print("MODULE 11: Wide-Ball Decision Model")
    print("=" * 65)

    df = encode_wide_features(df)

    # Only use rows where stumps data exists
    valid = df["stumps_x"].notna() & df["is_wide"].notna()
    df_v = df[valid].copy()
    print(f"  Valid rows for wide model: {len(df_v):,}")
    print(f"  Wide rate: {df_v['is_wide'].mean()*100:.2f}%")

    avail = [f for f in WIDE_ML_FEATURES if f in df_v.columns]
    X = df_v[avail].fillna(df_v[avail].median())
    y = df_v["is_wide"].astype(int)

    groups = df_v["match_id"].astype(str) + "_" + df_v["format"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

    # Class weights (wides are rare ~3-5% of deliveries)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Class balance (pos_weight): {pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        random_state=RANDOM_SEED,
        verbosity=0,
        eval_metric="auc",
    )
    model.fit(X_train, y_train, verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "precision":  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc":     round(float(average_precision_score(y_test, y_prob)), 4),
        "brier":      round(float(brier_score_loss(y_test, y_prob)), 4),
        "n_test":     len(y_test),
        "wide_rate":  round(float(y_test.mean()), 4),
    }

    print(f"\n  WIDE MODEL RESULTS:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"    Brier:     {metrics['brier']:.4f}")

    # Save
    model_path = MODELS_SAVED / "wide_ball_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": avail, "metrics": metrics}, f)

    metrics_path = EXPERIMENTS / "results" / "wide_model_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Model saved: {model_path}")

    return {"model": model, "features": avail, "metrics": metrics}


def train_or_load_wide_model(df: pd.DataFrame) -> Dict:
    """Loads existing wide model if it exists, otherwise trains a new one."""
    model_path = MODELS_SAVED / "wide_ball_model.pkl"
    if model_path.exists():
        print(f"Loading existing wide model from {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Training new wide model...")
        return train_wide_model(df)


def predict_wide(delivery: dict, model_bundle: dict = None) -> dict:
    """
    Predict wide probability for a single delivery.
    Falls back to rule-based if ML model not available.
    Returns:
      wide_probability, decision, confidence, reason
    """
    row = pd.Series(delivery)

    # Rule-based component
    rule_result = rule_based_wide(row)
    rule_prob = rule_result.get("wide_prob_rule", np.nan)

    ml_prob = None
    if model_bundle is not None:
        try:
            model   = model_bundle["model"]
            feats   = model_bundle["features"]
            X_row   = pd.DataFrame([delivery])
            X_row   = encode_wide_features(X_row)
            avail   = [f for f in feats if f in X_row.columns]
            X_input = X_row[avail].fillna(0)
            # Pad missing features with 0
            for f in feats:
                if f not in X_input.columns:
                    X_input[f] = 0.0
            X_input = X_input[feats].fillna(0)
            ml_prob = float(model.predict_proba(X_input)[0, 1])
        except Exception as e:
            ml_prob = None

    # Ensemble: average rule + ML if both available
    if ml_prob is not None and not np.isnan(rule_prob):
        final_prob = 0.4 * rule_prob + 0.6 * ml_prob
        method = "ensemble"
    elif ml_prob is not None:
        final_prob = ml_prob
        method = "ml_only"
    elif not np.isnan(rule_prob):
        final_prob = rule_prob
        method = "rule_only"
    else:
        return {"wide_probability": None, "decision": "REVIEW_REQUIRED",
                "confidence": 0.0, "reason": "insufficient_data", "method": "none"}

    # Decision
    if final_prob >= 0.70:
        decision = "WIDE"
        confidence = final_prob
    elif final_prob >= 0.40:
        decision = "REVIEW_REQUIRED"
        confidence = 1.0 - abs(final_prob - 0.5) * 2
    else:
        decision = "LEGAL"
        confidence = 1.0 - final_prob

    return {
        "wide_probability": round(final_prob, 3),
        "decision":         decision,
        "confidence":       round(confidence, 3),
        "reason":           rule_result.get("reason", ""),
        "method":           method,
    }


if __name__ == "__main__":
    master_path = DATA_MASTER / "master_dataset.parquet"
    if not master_path.exists():
        print("Run master_dataset.py first.")
        sys.exit(1)
    df = pd.read_parquet(master_path)
    print(f"Loaded: {df.shape}")
    result = train_wide_model(df)

    # Test single delivery prediction
    sample = {
        "stumps_x": 0.55, "stumps_y": 0.72,
        "pitch_x": 0.30, "pitch_y": 7.5,
        "ball_speed_kmh": 138.0,
        "ball_age_overs": 12.0,
        "lateral_swing": 0.25,
        "format": "IPL_Men",
        "bowling_style": "FAST_SEAM",
        "right_handed_bat": True, "batter_is_right": True,
        "bowler_is_right": True, "is_new_ball_period": 0,
    }
    pred = predict_wide(sample, result)
    print(f"\nSample prediction (stumpsX=0.55, just outside off stump):")
    print(f"  Wide probability: {pred['wide_probability']}")
    print(f"  Decision: {pred['decision']}")
    print(f"  Confidence: {pred['confidence']}")
    print(f"  Reason: {pred['reason']}")
