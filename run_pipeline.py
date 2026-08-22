"""
run_pipeline.py  —  Master run script
Executes the full research pipeline end-to-end.
Usage:
  py run_pipeline.py                    # Full pipeline (includes CricSheet + weather)
  py run_pipeline.py --offline          # Skip CricSheet download + weather API
  py run_pipeline.py --skip-cricsheet   # Skip CricSheet only
  py run_pipeline.py --models-only      # Skip data prep, run models only
  py run_pipeline.py --dashboard        # Launch Streamlit dashboard
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from config import DATA_MASTER, DATA_PROCESSED, MODELS_SAVED


def run_data_pipeline(skip_cricsheet: bool = False, skip_weather: bool = False):
    print("\n" + "="*65)
    print("  STEP 1/6 — Hawkeye Ingestion & Feature Derivation")
    print("="*65)
    from ingestion.hawkeye_ingest import run_ingestion
    df = run_ingestion()
    df.to_parquet(DATA_PROCESSED / "hawkeye_clean.parquet", index=False)

    print("\n" + "="*65)
    print("  STEP 2/6 — Ball-State Features")
    print("="*65)
    from features.ball_state import build_ball_state_features, build_ball_summary_table
    df = build_ball_state_features(df)
    df.to_parquet(DATA_PROCESSED / "hawkeye_with_ball_state.parquet", index=False)
    ball_summary = build_ball_summary_table(df)
    ball_summary.to_parquet("data/ball_state/ball_state_summary.parquet", index=False)
    print(f"  Ball summary: {len(ball_summary):,} unique balls tracked")

    if not skip_cricsheet:
        print("\n" + "="*65)
        print("  STEP 3/6 — CricSheet Venue/Date Join")
        print("="*65)
        from ingestion.cricsheet_join import run_cricsheet_join
        df = run_cricsheet_join(df)
        df.to_parquet(DATA_PROCESSED / "hawkeye_with_venue.parquet", index=False)
    else:
        print("  [SKIP] CricSheet join (offline mode)")

    if not skip_weather and not skip_cricsheet:
        print("\n" + "="*65)
        print("  STEP 4/6 — Weather Fetch (Open-Meteo)")
        print("="*65)
        from ingestion.weather_fetch import run_weather_fetch
        df = run_weather_fetch(df)
        df.to_parquet(DATA_PROCESSED / "hawkeye_with_weather.parquet", index=False)
    else:
        print("  [SKIP] Weather fetch")

    print("\n" + "="*65)
    print("  STEP 5/6 — Bowler Profile Table")
    print("="*65)
    from sklearn.model_selection import GroupShuffleSplit
    from features.bowler_profiles import build_bowler_profiles, save_bowler_profiles, merge_bowler_features
    from config import RANDOM_SEED
    groups = df["match_id"].astype(str) + "_" + df["format"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, _ = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    print(f"  Training split: {len(train_df):,} rows, {train_df['match_id'].nunique():,} matches")
    profiles = build_bowler_profiles(train_df)
    save_bowler_profiles(profiles)
    df = merge_bowler_features(df, profiles)

    print("\n" + "="*65)
    print("  STEP 6/6 — Saving Master Dataset + Docs")
    print("="*65)
    DATA_MASTER.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_MASTER / "master_dataset.parquet", index=False)

    # Missing data report
    miss = df.isnull().mean().reset_index()
    miss.columns = ["column", "missing_fraction"]
    miss["missing_pct"] = (miss["missing_fraction"] * 100).round(2)
    miss.to_csv(DATA_MASTER / "missing_data_report.csv", index=False)

    print(f"\n  Master dataset: {df.shape}")
    print(f"  Saved: {DATA_MASTER / 'master_dataset.parquet'}")

    # Key statistics
    valid_traj = df["has_trajectory"].sum() if "has_trajectory" in df.columns else df["stumps_x"].notna().sum()
    print(f"\n  KEY STATS:")
    print(f"    Total deliveries:         {len(df):,}")
    print(f"    Valid trajectory rows:    {valid_traj:,}")
    print(f"    Wide deliveries:          {df['is_wide'].sum() if 'is_wide' in df.columns else '?':,}")
    print(f"    No-ball deliveries:       {df['is_no_ball'].sum() if 'is_no_ball' in df.columns else '?':,}")
    print(f"    Venue join coverage:      "
          f"{(df.get('venue_join_confidence', pd.Series()).isin(['HIGH','MEDIUM'])).sum():,}")
    print(f"    Weather coverage:         "
          f"{df.get('weather_available', pd.Series(0)).sum():,}")
    print(f"    Unique bowlers profiled:  {profiles['bowler_id'].nunique():,}")
    return df


def run_models(df: pd.DataFrame = None):
    if df is None:
        print("  Loading master_dataset.parquet...")
        df = pd.read_parquet(DATA_MASTER / "master_dataset.parquet")
        print(f"  Loaded: {df.shape}")

    print("\n" + "="*65)
    print("  MODEL — Physics Trajectory Baseline")
    print("="*65)
    from models.physics.physics_model import predict_physics_batch, evaluate_physics_model
    sample = df[df["has_trajectory"]==1].head(20000) if "has_trajectory" in df.columns else df.head(20000)
    sample = predict_physics_batch(sample)
    phys_metrics = evaluate_physics_model(sample)
    print("  Physics model metrics:")
    for k, v in phys_metrics.items():
        print(f"    {k}: {v}")

    print("\n" + "="*65)
    print("  MODEL — Context-Aware Trajectory (Ablation)")
    print("="*65)
    from evaluation.ablation_study import run_ablation
    ablation_results = run_ablation(df)

    print("\n" + "="*65)
    print("  MODEL — Wide Ball Decision Assistance")
    print("="*65)
    from models.wide_ball.wide_ball_model import train_wide_model
    wide_bundle = train_wide_model(df)

    print("\n" + "="*65)
    print("  MODEL — Anomaly / Unusual Delivery Detection")
    print("="*65)
    from models.anomaly.anomaly_detection import run_anomaly_detection
    # Add physics residuals to sample first
    if "residual_stumps_x" not in df.columns:
        df_sample = predict_physics_batch(df[df.get("has_trajectory", pd.Series(1, index=df.index))==1].head(50000))
        df.loc[df_sample.index, "residual_stumps_x"] = df_sample.get("residual_stumps_x", None)
        df.loc[df_sample.index, "residual_stumps_y"] = df_sample.get("residual_stumps_y", None)
    df, anomaly_df = run_anomaly_detection(df)

    print("\n" + "="*65)
    print("  ALL MODELS COMPLETE")
    print("="*65)
    return df


def launch_dashboard():
    import subprocess
    import sys
    dashboard_path = ROOT / "src" / "dashboard" / "dashboard.py"
    print(f"  Launching Streamlit dashboard: {dashboard_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket Ball Intelligence System — Master Pipeline")
    parser.add_argument("--offline",          action="store_true", help="Skip CricSheet + weather API calls")
    parser.add_argument("--skip-cricsheet",   action="store_true", help="Skip CricSheet download")
    parser.add_argument("--skip-weather",     action="store_true", help="Skip weather fetch")
    parser.add_argument("--models-only",      action="store_true", help="Run models only (data already prepared)")
    parser.add_argument("--data-only",        action="store_true", help="Run data pipeline only")
    parser.add_argument("--dashboard",        action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
    elif args.models_only:
        run_models()
    elif args.data_only:
        run_data_pipeline(
            skip_cricsheet=args.skip_cricsheet or args.offline,
            skip_weather=args.skip_weather or args.offline
        )
    else:
        df = run_data_pipeline(
            skip_cricsheet=args.skip_cricsheet or args.offline,
            skip_weather=args.skip_weather or args.offline
        )
        if not args.data_only:
            run_models(df)

    print("\nPipeline complete!")
