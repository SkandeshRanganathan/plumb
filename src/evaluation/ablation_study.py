"""
ablation_study.py  –  MODULE 14 (Ablation)
Runs the 12-model ablation study to isolate which contextual features
actually improve trajectory prediction.

Models:
  1.  Trajectory only (speed + pitchXY)
  2.  + Venue/country
  3.  + Pitch type
  4.  + Weather
  5.  + Ball age
  6.  + Ball state (rolling features)
  7.  + Bowler profile
  8.  + All context (Model B)
  9.  Physics only
  10. ML only (pure data-driven, all features)
  11. Physics + ML residual
  12. + Ball type

Each model predicts stumps_x and stumps_y on the held-out test set.
Results are saved as a comparison table.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from config import DATA_MASTER, MODELS_SAVED, EXPERIMENTS, RANDOM_SEED

EXPERIMENTS.mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "results").mkdir(parents=True, exist_ok=True)


# ── Ablation feature sets ─────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "M1_traj_only": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
    ],
    "M2_plus_venue": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        # Venue encoded as country one-hot
        "country_India","country_England","country_Australia",
        "country_South Africa","country_New Zealand","country_Pakistan",
        "country_Sri Lanka","country_West Indies","country_Bangladesh",
        "country_UAE","country_Zimbabwe",
    ],
    "M3_plus_pitch": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "pitch_type_flat","pitch_type_seaming","pitch_type_turning","pitch_type_bouncy",
    ],
    "M4_plus_weather": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "temperature_c","humidity_pct","wind_speed_kmh",
        "wind_direction_deg","cloud_cover_pct","pressure_hpa",
    ],
    "M5_plus_ball_age": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "ball_age_overs","ball_age_since_replacement","is_new_ball_period",
        "btype_SG","btype_Dukes","btype_Kookaburra",
        "btype_Kookaburra_White","btype_Kookaburra_Pink",
    ],
    "M6_plus_ball_state": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "ball_age_overs","ball_age_since_replacement","is_new_ball_period",
        "roll_speed_5ov","roll_speed_10ov",
        "roll_swing_5ov","roll_swing_10ov",
        "roll_bounce_5ov","speed_decline_trend","swing_trend","roughness_proxy",
    ],
    "M7_plus_bowler": [
        "ball_speed_kmh", "pitch_x", "pitch_y",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "ball_age_overs",
        "bp_career_avg_speed_kmh","bp_career_speed_cv",
        "bp_career_avg_pitch_y","bp_career_avg_pitch_x",
        "bp_career_avg_lateral_swing","bp_career_swing_std",
        "bp_career_wide_rate","bp_career_avg_stumps_y",
    ],
    "M8_all_context": [
        # Same as FEATURES_CONTEXT_AWARE in trajectory_models.py
        "ball_speed_kmh", "pitch_x", "pitch_y", "ball_age_overs",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "batter_is_right","bowler_is_right",
        "ball_age_since_replacement","is_new_ball_period",
        "roll_speed_5ov","roll_speed_10ov",
        "roll_swing_5ov","roll_swing_10ov","roll_bounce_5ov",
        "speed_decline_trend","swing_trend","roughness_proxy",
        "btype_SG","btype_Dukes","btype_Kookaburra",
        "btype_Kookaburra_White","btype_Kookaburra_Pink",
        "bp_career_avg_speed_kmh","bp_career_speed_cv",
        "bp_career_avg_pitch_y","bp_career_avg_pitch_x",
        "bp_career_avg_lateral_swing","bp_career_swing_std",
        "bp_career_wide_rate","bp_career_avg_stumps_y",
        "temperature_c","humidity_pct","wind_speed_kmh",
        "wind_direction_deg","cloud_cover_pct","pressure_hpa",
        "country_India","country_England","country_Australia",
        "country_South Africa","country_New Zealand","country_Pakistan",
        "country_Sri Lanka","country_West Indies","country_Bangladesh",
    ],
    "M12_plus_ball_type": [
        "ball_speed_kmh", "pitch_x", "pitch_y", "ball_age_overs",
        "style_FAST_SEAM","style_MEDIUM_SEAM","style_OFF_SPIN",
        "style_ORTHODOX","style_LEG_SPIN","style_UNORTHODOX","style_SEAM",
        "btype_SG","btype_Dukes","btype_Kookaburra",
        "btype_Kookaburra_White","btype_Kookaburra_Pink",
        "ball_age_since_replacement","is_new_ball_period",
    ],
}


def encode_for_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all categorical features needed for ablation study."""
    from models.context_aware.trajectory_models import encode_categorical_features
    df = encode_categorical_features(df)

    # Country one-hot
    countries = ["India","England","Australia","South Africa","New Zealand",
                 "Pakistan","Sri Lanka","West Indies","Bangladesh","UAE","Zimbabwe"]
    for c in countries:
        col = f"country_{c}"
        df[col] = (df.get("country", pd.Series(dtype=str)) == c).astype(int)

    # Pitch type one-hot (sparse — mostly NULL, will be 0)
    pitch_types = ["flat","seaming","turning","bouncy"]
    for pt in pitch_types:
        col = f"pitch_type_{pt}"
        if "pitch_type" in df.columns:
            df[col] = df["pitch_type"].str.lower().str.contains(pt, na=False).astype(int)
        else:
            df[col] = 0

    return df


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Run all ablation models and return comparison table."""
    from sklearn.model_selection import GroupShuffleSplit
    import xgboost as xgb

    print("=" * 65)
    print("ABLATION STUDY: Feature Contribution Analysis")
    print("=" * 65)

    df = encode_for_ablation(df)

    # Match-level split
    groups = df["match_id"].astype(str) + "_" + df["format"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_val_idx]
    test_df  = df.iloc[test_idx]

    valid_train = (train_df["stumps_x"].notna() & train_df["pitch_x"].notna() &
                   train_df["ball_speed_kmh"].notna() & train_df["stumps_y"].notna())
    valid_test  = (test_df["stumps_x"].notna()  & test_df["pitch_x"].notna()  &
                   test_df["ball_speed_kmh"].notna() & test_df["stumps_y"].notna())

    train_clean = train_df[valid_train]
    test_clean  = test_df[valid_test]
    print(f"  Train: {len(train_clean):,}  Test: {len(test_clean):,}")

    all_results = []
    TARGETS = ["stumps_x", "stumps_y"]

    for model_key, feature_cols in ABLATION_CONFIGS.items():
        # Keep only features that exist in the DataFrame
        avail_feats = [c for c in feature_cols if c in df.columns]
        if len(avail_feats) < 2:
            print(f"  [SKIP] {model_key}: only {len(avail_feats)} features available")
            continue

        print(f"\n  {model_key} ({len(avail_feats)} features)")

        for target in TARGETS:
            X_train = train_clean[avail_feats].fillna(train_clean[avail_feats].median())
            y_train = train_clean[target]
            X_test  = test_clean[avail_feats].fillna(train_clean[avail_feats].median())
            y_test  = test_clean[target]

            model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=RANDOM_SEED, verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)

            residuals = y_test - y_pred
            mae  = float(np.abs(residuals).mean())
            rmse = float(np.sqrt((residuals**2).mean()))
            r2   = float(1 - (residuals**2).sum() / ((y_test - y_test.mean())**2).sum())

            row = {
                "model":        model_key,
                "n_features":   len(avail_feats),
                "target":       target,
                "mae_m":        round(mae, 5),
                "rmse_m":       round(rmse, 5),
                "r2":           round(r2, 4),
                "n_test":       len(y_test),
            }
            all_results.append(row)
            print(f"    {target}: MAE={mae:.4f}m  RMSE={rmse:.4f}m  R²={r2:.4f}")

            # Save model
            save_path = MODELS_SAVED / f"ablation_{model_key}_{target}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({"model": model, "features": avail_feats}, f)

    results_df = pd.DataFrame(all_results)
    out_path = EXPERIMENTS / "results" / "ablation_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("ABLATION SUMMARY (stumps_x)")
    print("=" * 65)
    sx = results_df[results_df["target"] == "stumps_x"].sort_values("rmse_m")
    print(sx[["model","n_features","mae_m","rmse_m","r2"]].to_string(index=False))
    print(f"\nResults saved: {out_path}")
    return results_df


if __name__ == "__main__":
    master_path = DATA_MASTER / "master_dataset.parquet"
    if not master_path.exists():
        print("Run master_dataset.py first.")
        sys.exit(1)
    df = pd.read_parquet(master_path)
    print(f"Loaded: {df.shape}")
    run_ablation(df)
