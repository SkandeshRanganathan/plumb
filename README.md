# Context-Aware Adaptive Cricket Ball Intelligence System

## Project Structure
```
cric/
├── run_pipeline.py              ← Master run script (start here)
├── src/
│   ├── config.py                ← All paths, bounds, constants
│   ├── ingestion/
│   │   ├── hawkeye_ingest.py    ← MODULE 1-A: Load + clean all 6 HawkeyeStats CSVs
│   │   ├── cricsheet_join.py    ← MODULE 1-D: Venue/date join from CricSheet
│   │   ├── weather_fetch.py     ← MODULE 1-E: Open-Meteo weather API
│   │   └── master_dataset.py   ← MODULE 1-H: Pipeline orchestrator
│   ├── features/
│   │   ├── bowler_profiles.py  ← MODULE 1-C: Bowler career profile features
│   │   └── ball_state.py       ← MODULE 1-G: Ball-state rolling features
│   ├── models/
│   │   ├── physics/
│   │   │   └── physics_model.py        ← MODULE 6: Physics trajectory simulation
│   │   ├── context_aware/
│   │   │   └── trajectory_models.py    ← MODULE 7: XGBoost context-aware models
│   │   ├── anomaly/
│   │   │   └── anomaly_detection.py    ← MODULE 10: Unusual delivery detection
│   │   ├── wide_ball/
│   │   │   └── wide_ball_model.py      ← MODULE 11: Wide-ball decision assistance
│   │   └── no_ball/
│   │       └── no_ball_model.py        ← MODULE 12: No-ball classification
│   ├── evaluation/
│   │   └── ablation_study.py   ← Experiments 1-8: Feature ablation
│   ├── explainability/
│   │   └── shap_analysis.py    ← MODULE 13: SHAP explainability
│   └── dashboard/
│       └── dashboard.py        ← MODULE 15: Streamlit research dashboard
├── data/
│   ├── processed/               ← Intermediate parquet files
│   ├── master/                  ← master_dataset.parquet (source of truth)
│   ├── external/
│   │   ├── cricsheet/           ← Downloaded CricSheet JSON files
│   │   └── weather/             ← Cached Open-Meteo responses
│   ├── bowler_profiles/         ← bowler_profiles.parquet + .csv
│   └── ball_state/              ← ball_state_summary.parquet
├── models/
│   └── saved/                   ← All saved .pkl model files
├── experiments/
│   ├── results/                 ← ablation_results.csv, trajectory_experiments.csv
│   └── plots/                   ← shap_*.png and other plots
└── datasets/                    ← Raw data (git-cloned repos)
    ├── hawkeye_stats/           ← 6 HawkeyeStats CSV files
    └── cric360/                 ← Cric360 broadcast images
```

## Quick Start

### 1. Full Pipeline (requires internet for CricSheet + weather)
```bash
py run_pipeline.py
```

### 2. Offline Mode (skip CricSheet + weather API)
```bash
py run_pipeline.py --offline
```

### 3. Data Preparation Only
```bash
py run_pipeline.py --data-only --offline
```

### 4. Models Only (data already prepared)
```bash
py run_pipeline.py --models-only
```

### 5. Launch Research Dashboard
```bash
py run_pipeline.py --dashboard
# OR directly:
py -m streamlit run src/dashboard/dashboard.py
```

## Dataset Summary

| Dataset | Rows | Source |
|---|---|---|
| mensIPLHawkeyeStats.csv | 149,424 | HawkeyeStats GitHub |
| mensODIHawkeyeStats.csv | 439,105 | HawkeyeStats GitHub |
| mensTestHawkeyeStats.csv | 527,165 | HawkeyeStats GitHub |
| womensIPLHawkeyeStats.csv | 2,096 | HawkeyeStats GitHub |
| womensODIHawkeyeStats.csv | 9,403 | HawkeyeStats GitHub |
| womensTestHawkeyeStats.csv | 3,909 | HawkeyeStats GitHub |
| **Total** | **1,131,102** | |
| Valid trajectory rows | 792,366 | Derived |
| Wide deliveries | 15,648 | Labeled |
| No-ball deliveries | 3,290 | Labeled |

## Research Experiments

| # | Experiment | File |
|---|---|---|
| 1 | Context-Free vs Context-Aware | `trajectory_models.py` |
| 2 | Physics vs ML vs Physics+ML | `trajectory_models.py` |
| 3 | Ball age contribution | `ablation_study.py` |
| 4 | Ball state contribution | `ablation_study.py` |
| 5 | Weather contribution | `ablation_study.py` |
| 6 | Pitch/venue contribution | `ablation_study.py` |
| 7 | Cross-venue generalization | `trajectory_models.py` |
| 8 | Ball-age split | `trajectory_models.py` |
| 9 | Delivery classification | `trajectory_models.py` |
| 10 | Anomaly detection | `anomaly_detection.py` |
| 11 | Wide-ball decision assistance | `wide_ball_model.py` |
| 12 | Bowler profiling contribution | `ablation_study.py` |

## Key Technical Decisions
- **Match-level splits**: Never split at delivery level to prevent set contamination
- **Bowler profiles from training split only**: Prevents data leakage
- **Rolling features use .shift(1)**: No current-delivery information in rolling stats
- **NULL/NaN explicit**: Missing fields are never fabricated
- **Physics residuals**: Actual − physics prediction is the core anomaly signal
