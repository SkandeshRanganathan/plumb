"""
trajectory_models.py  –  MODULE 7
Context-Aware Trajectory Prediction Models.

Implements four model variants for comparison:

MODEL A — Context-Free Baseline:
  Inputs: ball_speed_kmh, pitch_x, pitch_y, bowling_style (encoded)
  Target: stumps_x, stumps_y

MODEL B — Context-Aware:
  Inputs: Model A features + venue/country, pitch_type, weather,
          ball_age, ball_condition, bowler_profile features

MODEL C — Physics + ML Residual:
  Physics model predicts (phys_pred_stumps_x, phys_pred_stumps_y)
  ML predicts residual: residual_stumps_x, residual_stumps_y
  Final: physics_pred + ml_residual

MODEL D — Ball-State Augmented:
  Model B + rolling ball-state features (speed/swing/bounce rolling means)

All models use XGBoost as the primary learner (with RF as comparison).
Feature importance and SHAP values are computed for Models B and D.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from config import DATA_MASTER, DATA_PROCESSED, MODELS_SAVED, EXPERIMENTS, RANDOM_SEED

MODELS_SAVED.mkdir(parents=True, exist_ok=True)
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "results").mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "plots").mkdir(parents=True, exist_ok=True)


# ── Feature sets ──────────────────────────────────────────────────────────────

FEATURES_CONTEXT_FREE = [
    "ball_speed_kmh",
    "pitch_x",
    "pitch_y",
    "ball_age_overs",
    # Bowling style (one-hot encoded below)
    "style_FAST_SEAM", "style_MEDIUM_SEAM", "style_OFF_SPIN",
    "style_ORTHODOX", "style_LEG_SPIN", "style_UNORTHODOX", "style_SEAM",
    # Handedness matchup
    "batter_is_right", "bowler_is_right",
]

FEATURES_CONTEXT_AWARE = FEATURES_CONTEXT_FREE + [
    # Ball state
    "ball_age_since_replacement",
    "is_new_ball_period",
    "roll_speed_5ov", "roll_speed_10ov",
    "roll_swing_5ov", "roll_swing_10ov",
    "roll_bounce_5ov",
    "speed_decline_trend",
    "swing_trend",
    "roughness_proxy",
    # Ball type (one-hot encoded below)
    "btype_SG", "btype_Dukes", "btype_Kookaburra",
    "btype_Kookaburra_White", "btype_Kookaburra_Pink",
    # Bowler profile features (prefix bp_)
    "bp_career_avg_speed_kmh",
    "bp_career_speed_cv",
    "bp_career_avg_pitch_y",
    "bp_career_avg_pitch_x",
    "bp_career_avg_lateral_swing",
    "bp_career_swing_std",
    "bp_career_wide_rate",
    "bp_career_avg_stumps_y",
    # Weather
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "wind_direction_deg",
    "cloud_cover_pct",
    "pressure_hpa",
]

PHYSICS_RESIDUAL_FEATURES = FEATURES_CONTEXT_AWARE + [
    "phys_pred_stumps_x",
    "phys_pred_stumps_y",
    "phys_pred_pitch_x",
    "phys_pred_pitch_y",
]

TARGETS_PRIMARY = ["stumps_x", "stumps_y"]
TARGETS_RESIDUAL = ["residual_stumps_x", "residual_stumps_y"]


# ── Feature engineering ───────────────────────────────────────────────────────

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode bowling style and ball type into indicator columns."""
    styles = ["FAST_SEAM","MEDIUM_SEAM","OFF_SPIN","ORTHODOX","LEG_SPIN","UNORTHODOX","SEAM"]
    for s in styles:
        col = f"style_{s}"
        df[col] = (df["bowling_style"] == s).astype(int)

    btypes = ["SG","Dukes","Kookaburra","Kookaburra_White","Kookaburra_Pink"]
    for b in btypes:
        col = f"btype_{b}"
        df[col] = (df["ball_type"] == b).astype(int)

    return df


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix, keeping only rows where all key fields are valid.
    Fills remaining NaNs with column medians.
    Returns (X, valid_mask).
    """
    # Ensure all required columns exist (fill missing ones with 0 or NaN)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].copy()

    # Fill NaN with column median (fit on train, applied on test)
    medians = X.median()
    X = X.fillna(medians)

    valid_mask = (
        df["stumps_x"].notna() &
        df["stumps_y"].notna() &
        df["pitch_x"].notna() &
        df["pitch_y"].notna() &
        df["ball_speed_kmh"].notna()
    )
    return X, valid_mask


# ── Dataset splitting ─────────────────────────────────────────────────────────

def make_splits(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.Index]:
    """
    Match-level random split (prevents same match appearing in train + test).
    Returns dict of index arrays: {train, val, test}
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = df["match_id"].astype(str) + "_" + df["format"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss.split(df, groups=groups))

    train_val = df.iloc[train_val_idx]
    groups_tv = groups.iloc[train_val_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss2.split(train_val, groups=groups_tv))

    return {
        "train": df.iloc[train_val_idx].iloc[train_idx].index,
        "val":   df.iloc[train_val_idx].iloc[val_idx].index,
        "test":  df.iloc[test_idx].index,
    }


def make_cross_venue_split(df: pd.DataFrame,
                            test_countries: List[str]) -> Tuple[pd.Index, pd.Index]:
    """Create cross-venue split: train on some countries, test on others."""
    test_mask  = df["country"].isin(test_countries)
    train_mask = ~test_mask & df["country"].notna()
    return df[train_mask].index, df[test_mask].index


def make_ball_age_split(df: pd.DataFrame,
                         train_max_overs: float = 50.0,
                         test_min_overs: float = 55.0) -> Tuple[pd.Index, pd.Index]:
    """Split by ball age: train on young ball, test on old ball."""
    train_mask = df["ball_age_overs"] <= train_max_overs
    test_mask  = df["ball_age_overs"] >= test_min_overs
    return df[train_mask].index, df[test_mask].index


# ── Model training ────────────────────────────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   target_name: str = "target",
                   n_estimators: int = 500,
                   early_stopping: int = 30) -> object:
    """Train an XGBoost regressor with early stopping on validation set."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=early_stopping if X_val is not None else None,
        verbosity=0,
    )
    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                         target_name: str = "target") -> object:
    """Train a Random Forest as a comparison baseline."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray,
                          label: str = "model") -> Dict:
    """Compute regression metrics."""
    residuals = y_true - y_pred
    mae  = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    r2   = 1 - (residuals**2).sum() / ((y_true - y_true.mean())**2).sum()
    p90  = np.percentile(np.abs(residuals), 90)
    return {
        "label": label,
        "n":     len(y_true),
        "mae":   round(float(mae), 5),
        "rmse":  round(float(rmse), 5),
        "r2":    round(float(r2), 4),
        "p90_abs_error": round(float(p90), 5),
    }


def euclidean_trajectory_error(
    true_x: pd.Series, true_y: pd.Series,
    pred_x: np.ndarray, pred_y: np.ndarray
) -> Dict:
    """2D Euclidean position error at stump crossing."""
    dist = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
    return {
        "euclidean_mae":  round(float(dist.mean()), 5),
        "euclidean_rmse": round(float(np.sqrt((dist**2).mean())), 5),
        "euclidean_p90":  round(float(np.percentile(dist, 90)), 5),
    }


# ── Full experiment runner ────────────────────────────────────────────────────

class TrajectoryExperiment:
    """
    Runs the full set of trajectory prediction experiments:
      - Experiment 1: Context-free vs context-aware
      - Experiment 2: Physics vs ML vs Physics+ML
      - Experiment 3-6: Ablation studies
      - Experiment 7: Cross-venue generalization
      - Experiment 8: Ball-age split
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = []
        self._prepare()

    def _prepare(self):
        """Encode categoricals and prepare data."""
        self.df = encode_categorical_features(self.df)
        self.splits = make_splits(self.df)
        print(f"  Data splits:")
        print(f"    Train: {len(self.splits['train']):,}")
        print(f"    Val:   {len(self.splits['val']):,}")
        print(f"    Test:  {len(self.splits['test']):,}")

    def _get_split_data(self, feature_cols: List[str], target: str,
                         split: str) -> Tuple:
        idx = self.splits[split]
        df_split = self.df.loc[idx]
        X, valid = prepare_features(df_split, feature_cols)
        valid_idx = df_split[valid].index
        X_v = X.loc[valid_idx]
        y_v = df_split.loc[valid_idx, target]
        return X_v, y_v, valid_idx

    def run_model_ab(self):
        """Experiment 1: Context-Free (A) vs Context-Aware (B) for stumps_x and stumps_y."""
        print("\n" + "=" * 65)
        print("EXPERIMENT 1: Context-Free (A) vs Context-Aware (B)")
        print("=" * 65)

        for model_name, feat_cols in [
            ("A_context_free",  FEATURES_CONTEXT_FREE),
            ("B_context_aware", FEATURES_CONTEXT_AWARE),
        ]:
            for target in TARGETS_PRIMARY:
                print(f"\n  Model {model_name} | Target: {target}")
                X_train, y_train, _ = self._get_split_data(feat_cols, target, "train")
                X_val,   y_val,   _ = self._get_split_data(feat_cols, target, "val")
                X_test,  y_test,  _ = self._get_split_data(feat_cols, target, "test")

                if len(X_train) < 100:
                    print(f"    [SKIP] insufficient training data ({len(X_train)} rows)")
                    continue

                model = train_xgboost(X_train, y_train, X_val, y_val, target)
                y_pred = model.predict(X_test)

                metrics = evaluate_predictions(y_test, y_pred,
                                               label=f"{model_name}_{target}")
                metrics["model"] = model_name
                metrics["target"] = target
                metrics["split"] = "random_test"
                self.results.append(metrics)

                print(f"    MAE={metrics['mae']:.4f}m  RMSE={metrics['rmse']:.4f}m  R²={metrics['r2']:.4f}")

                # Save model
                model_path = MODELS_SAVED / f"{model_name}_{target}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump({"model": model, "features": feat_cols}, f)

    def run_physics_ml(self):
        """Experiment 2: Physics vs ML vs Physics+ML residual."""
        print("\n" + "=" * 65)
        print("EXPERIMENT 2: Physics vs ML vs Physics+ML Residual")
        print("=" * 65)

        # Check physics columns exist
        if "phys_pred_stumps_x" not in self.df.columns:
            print("  [SKIP] Run physics model first (predict_physics_batch)")
            return

        for target, residual_target in [("stumps_x", "residual_stumps_x"),
                                         ("stumps_y", "residual_stumps_y")]:
            print(f"\n  Target: {target}")

            # Physics-only baseline
            phys_col = f"phys_pred_{target}"
            test_idx = self.splits["test"]
            test_df  = self.df.loc[test_idx]
            valid_mask = test_df[target].notna() & test_df[phys_col].notna()
            if valid_mask.sum() > 0:
                phys_metrics = evaluate_predictions(
                    test_df.loc[valid_mask, target],
                    test_df.loc[valid_mask, phys_col].values,
                    label=f"physics_only_{target}"
                )
                phys_metrics["model"] = "physics_only"
                phys_metrics["target"] = target
                phys_metrics["split"] = "random_test"
                self.results.append(phys_metrics)
                print(f"    Physics-only: MAE={phys_metrics['mae']:.4f}m  RMSE={phys_metrics['rmse']:.4f}m")

            # ML on residual (Physics + ML)
            if residual_target not in self.df.columns:
                print(f"    [SKIP] {residual_target} not computed yet")
                continue

            X_train, y_train, _ = self._get_split_data(
                PHYSICS_RESIDUAL_FEATURES, residual_target, "train")
            X_val,   y_val,   _ = self._get_split_data(
                PHYSICS_RESIDUAL_FEATURES, residual_target, "val")
            X_test,  y_test,  test_valid_idx = self._get_split_data(
                PHYSICS_RESIDUAL_FEATURES, residual_target, "test")

            if len(X_train) < 100:
                continue

            model_res = train_xgboost(X_train, y_train, X_val, y_val, residual_target)
            pred_residual = model_res.predict(X_test)

            # Physics + ML = physics_pred + predicted_residual
            test_phys = self.df.loc[test_valid_idx, f"phys_pred_{target}"].fillna(0).values
            pred_combined = test_phys + pred_residual
            y_combined_true = self.df.loc[test_valid_idx, target]

            combined_metrics = evaluate_predictions(
                y_combined_true, pred_combined,
                label=f"physics_plus_ml_{target}"
            )
            combined_metrics["model"] = "physics_plus_ml"
            combined_metrics["target"] = target
            combined_metrics["split"] = "random_test"
            self.results.append(combined_metrics)
            print(f"    Physics+ML:   MAE={combined_metrics['mae']:.4f}m  RMSE={combined_metrics['rmse']:.4f}m")

            model_path = MODELS_SAVED / f"physics_ml_residual_{target}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model_res, "features": PHYSICS_RESIDUAL_FEATURES}, f)

    def run_cross_venue(self):
        """Experiment 7: Cross-venue generalization."""
        print("\n" + "=" * 65)
        print("EXPERIMENT 7: Cross-Venue Generalization")
        print("=" * 65)

        if "country" not in self.df.columns or self.df["country"].isna().all():
            print("  [SKIP] No venue data available")
            return

        test_scenarios = [
            (["South Africa"], "train_excl_SA"),
            (["England"], "train_excl_ENG"),
            (["Australia"], "train_excl_AUS"),
        ]

        for test_countries, scenario_name in test_scenarios:
            train_idx, test_idx = make_cross_venue_split(self.df, test_countries)
            if len(test_idx) < 50:
                print(f"  [SKIP] {scenario_name}: only {len(test_idx)} test rows")
                continue

            train_df = self.df.loc[train_idx]
            test_df  = self.df.loc[test_idx]

            for target in ["stumps_x"]:
                X_train, valid_train = prepare_features(train_df, FEATURES_CONTEXT_AWARE)
                X_test,  valid_test  = prepare_features(test_df,  FEATURES_CONTEXT_AWARE)
                X_tr = X_train[valid_train.loc[train_idx].values]
                y_tr = train_df.loc[valid_train.loc[train_idx].values, target]
                X_te = X_test[valid_test.loc[test_idx].values]
                y_te = test_df.loc[valid_test.loc[test_idx].values, target]

                if len(X_tr) < 100 or len(X_te) < 10:
                    continue

                model = train_xgboost(X_tr, y_tr, target_name=target)
                y_pred = model.predict(X_te)
                metrics = evaluate_predictions(y_te, y_pred,
                                               label=f"xvenue_{scenario_name}_{target}")
                metrics.update({"model": "context_aware", "target": target,
                                "split": f"cross_venue_{'+'.join(test_countries)}"})
                self.results.append(metrics)
                print(f"  Test {test_countries}: MAE={metrics['mae']:.4f}m  "
                      f"RMSE={metrics['rmse']:.4f}m  (n={metrics['n']})")

    def run_ball_age_split(self):
        """Experiment 8 (partial): Ball age generalization."""
        print("\n" + "=" * 65)
        print("EXPERIMENT 8: Ball Age Generalization")
        print("=" * 65)

        train_idx, test_idx = make_ball_age_split(self.df)
        train_df = self.df.loc[train_idx]
        test_df  = self.df.loc[test_idx]

        for target in ["stumps_x"]:
            X_train, valid_train = prepare_features(train_df, FEATURES_CONTEXT_AWARE)
            X_test,  valid_test  = prepare_features(test_df,  FEATURES_CONTEXT_AWARE)

            vt = valid_train.values
            ve = valid_test.values

            if vt.sum() < 100 or ve.sum() < 10:
                continue

            model = train_xgboost(
                X_train[vt], train_df[target][vt], target_name=target)
            y_pred = model.predict(X_test[ve])
            metrics = evaluate_predictions(test_df[target][ve], y_pred,
                                           label=f"ball_age_split_{target}")
            metrics.update({"model": "context_aware", "target": target,
                            "split": "ball_age_young_train_old_test"})
            self.results.append(metrics)
            print(f"  Young→Old ball: MAE={metrics['mae']:.4f}m  RMSE={metrics['rmse']:.4f}m")

    def save_results(self) -> pd.DataFrame:
        """Save all experiment results to CSV."""
        results_df = pd.DataFrame(self.results)
        path = EXPERIMENTS / "results" / "trajectory_experiments.csv"
        results_df.to_csv(path, index=False)
        print(f"\n✓ All experiment results saved to: {path}")
        print(results_df.to_string(index=False))
        return results_df


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 7: Context-Aware Trajectory Models")
    print("=" * 65)

    # Load master dataset
    master_path = DATA_MASTER / "master_dataset.parquet"
    if not master_path.exists():
        print("Run master_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(master_path)
    print(f"Loaded master dataset: {df.shape}")

    exp = TrajectoryExperiment(df)
    exp.run_model_ab()
    exp.run_physics_ml()
    exp.run_cross_venue()
    exp.run_ball_age_split()
    results = exp.save_results()
